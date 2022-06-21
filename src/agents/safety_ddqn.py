from collections import namedtuple
import itertools
import random
import time
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, PackedSequence
from tqdm import tqdm

from agents.agent import Agent
from agents.ddqn import DDQNParams, Basic2dEncoder, QModel
from envs.safe_env import SafetyConstrainedEnv
from replay import EpisodicExperienceReplay, ExperienceReplay
from safety.utils import START_TOKEN, from_one_hot, safety_spec_to_str, get_one_hot_spec

SAFETY_TRANSITION_TUPLE_SIZE = 8

SafetyTransition = namedtuple("SafetyTransition", ("state", "safety_spec", "action", "reward", "next_state", "prog_spec", "violation", "done"))
SafetyState = namedtuple("SafetyState", ["env_state", "formula"])

class SafetyDDQNParams(DDQNParams):
  def __init__(self, num_env_props, *args, q_loss_scale=10, vio_loss_scale=1, recon_loss_scale=0.1, spec_encoding_hidden_size=64, rnn_hidden_size=64, violation_threshold=0.5, chunk_size=4, violation_weighting_alpha=0.25, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_props = num_env_props
    self.q_loss_scale = q_loss_scale
    self.vio_loss_scale = vio_loss_scale
    self.recon_loss_scale = recon_loss_scale
    self.spec_hidden_size = spec_encoding_hidden_size
    self.rnn_hidden_size = rnn_hidden_size
    self.violation_threshold = violation_threshold
    self.chunk_size = chunk_size
    self.violation_weighting_alpha = violation_weighting_alpha


class SpecEncoder(nn.Module):
  def __init__(self, encoding_size, hidden_size, num_layers=1) -> None:
    super().__init__()
    self.lstm = nn.LSTM(input_size=encoding_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)

  def forward(self, encoded_seq: Union[torch.Tensor, PackedSequence]) -> torch.Tensor:
    hiddens, _ = self.lstm(encoded_seq)
    if isinstance(hiddens, PackedSequence):
      hiddens, _ = pad_packed_sequence(hiddens)
    return hiddens.sum(dim=0)

class SpecDecoder(nn.Module):
  def __init__(self, hidden_size, encoding_size, num_props) -> None:
    super().__init__()
    self.hidden_size = hidden_size
    self.encoding_size = encoding_size
    self.num_props = num_props

    self.lstm = nn.LSTM(input_size=encoding_size, hidden_size=hidden_size, batch_first=True)
    self.token_decoder = nn.Sequential(
      nn.Linear(hidden_size, 2 * hidden_size),
      nn.ReLU(),
      nn.Linear(2 * hidden_size, 2 * hidden_size),
      nn.ReLU(),
      nn.Linear(2 * hidden_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, encoding_size),
    )

  def _init_inputs(self, batch_size, device):
    start_encoding = get_one_hot_spec([START_TOKEN], self.num_props).to(device=device)
    start_tensor = torch.zeros((batch_size, 1, self.encoding_size)).to(device=device)
    start_tensor[:] = start_encoding

    c0 = torch.zeros((1, batch_size, self.hidden_size)).to(device=device)

    return start_tensor, c0

  def forward(self, hidden_specs, padded_target_spec, max_len):
    """
    Input:
      - hidden_specs: (batch_size, 2 * spec_hidden_size)
      - padded_target_spec: (batch_size, max_len, encoding_size)
      - target_lens: (batch_size) Target spec lengths including start and eos tokens

    Output:
      - decoded_spec: (batch_size, max_len, encoding_size)
    """
    batch_size = hidden_specs.shape[0]
    h0 = hidden_specs.unsqueeze(0)
    decoded_spec, c0 = self._init_inputs(batch_size, h0.device)

    # No prediction after last EOS token hence splicing
    # lstm_out: (batch_size, max_len, 2 * spec_hidden_size)
    lstm_out, _ = self.lstm(padded_target_spec[:, :-1], (h0, c0))
    flat_out = lstm_out.reshape(batch_size * (max_len - 1), self.hidden_size)

    flat_tokens = self.token_decoder(flat_out)
    pred_tokens = flat_tokens.reshape(batch_size, (max_len - 1), self.encoding_size)

    return torch.cat((decoded_spec, pred_tokens), dim=1)


class SafetyRNN(nn.Module):
  def __init__(self, spec_size, obs_size, action_size, hidden_size=64) -> None:
    super().__init__()
    self.hidden_size = hidden_size
    self.rnn = nn.GRUCell(spec_size + obs_size + action_size, hidden_size)
    self.violation_predictor = nn.Sequential(
      nn.Linear(hidden_size, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.Sigmoid()
    )
    self.spec_progressor = nn.Sequential(
      nn.Linear(hidden_size + spec_size, 128),
      nn.ReLU(),
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, spec_size)
    )
  
  def forward(self, encoded_spec, encoded_obs, action, hidden_state=None):
    """
    Inputs:
      - encoded_spec: (batch_size, 2 * spec_hidden_size)
      - encoded_obs: (batch_size, encoding_size)
      - action: (batch_size, action_size)

    Outputs:
      - p_violation: (batch_size, 1)
      - prog_spec: (batch_size, 2 * spec_hidden_size)
      - new_hidden = (batch_size, hidden_size)
    """
    curr_state = torch.cat((encoded_spec, encoded_obs, action), dim=1)
    batch_size = curr_state.shape[0]
    if hidden_state is None:
      hidden_state = torch.zeros((batch_size, self.hidden_size), device=curr_state.device)

    new_hidden = self.rnn(curr_state, hidden_state)
    p_violation = self.violation_predictor(new_hidden)
    prog_spec = self.spec_progressor(torch.cat((encoded_spec, new_hidden), dim=1))
    return p_violation, prog_spec, new_hidden

class ConditionalAdaptiveShieldHidden(nn.Module):
  def __init__(self, spec_size, obs_size, action_size, truth_hidden_size, hidden_size=64) -> None:
    super().__init__()
    self.hidden_size = hidden_size
    self.rnn = nn.GRUCell(spec_size + truth_hidden_size, hidden_size)
    self.truth_predictor = nn.Sequential(
      nn.Linear(obs_size + action_size, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, truth_hidden_size)
    )

    self.violation_predictor = nn.Sequential(
      nn.Linear(truth_hidden_size + spec_size, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.Sigmoid()
    )

    self.spec_progressor = nn.Sequential(
      nn.Linear(hidden_size, 128),
      nn.ReLU(),
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, spec_size)
    )
  
  def forward(self, encoded_spec, encoded_obs, action, hidden_state=None):
    """
    Inputs:
      - encoded_spec: (batch_size, 2 * spec_hidden_size)
      - encoded_obs: (batch_size, encoding_size)
      - action: (batch_size, action_size)

    Outputs:
      - p_violation: (batch_size, 1)
      - prog_spec: (batch_size, 2 * spec_hidden_size)
      - new_hidden = (batch_size, hidden_size)
    """
    curr_state = torch.cat((encoded_obs, action), dim=1)
    batch_size = curr_state.shape[0]
    approx_truth = self.truth_predictor(curr_state)
    if hidden_state is None:
      hidden_state = torch.zeros((batch_size, self.hidden_size), device=curr_state.device)

    inputs = torch.cat((encoded_spec, approx_truth), dim=1)
    p_violation = self.violation_predictor(inputs)
    new_hidden = self.rnn(inputs, hidden_state)
    prog_spec = self.spec_progressor(new_hidden)
    return p_violation, prog_spec, new_hidden


class ConditionalAdaptiveShield(nn.Module):
  def __init__(self, spec_size, obs_size, action_size, truth_hidden_size) -> None:
    super().__init__()
    self.rnn = nn.GRUCell(truth_hidden_size, spec_size)
    self.truth_predictor = nn.Sequential(
      nn.Linear(obs_size + action_size, 64),
      nn.ReLU(),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, truth_hidden_size)
    )

    self.violation_predictor = nn.Sequential(
      nn.Linear(truth_hidden_size + spec_size, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
      nn.Sigmoid()
    )
  
  def forward(self, encoded_spec, encoded_obs, action):
    """
    Inputs:
      - encoded_spec: (batch_size, 2 * spec_hidden_size)
      - encoded_obs: (batch_size, encoding_size)
      - action: (batch_size, action_size)

    Outputs:
      - p_violation: (batch_size, 1)
      - prog_spec: (batch_size, 2 * spec_hidden_size)
    """
    curr_state = torch.cat((encoded_obs, action), dim=1)
    batch_size = curr_state.shape[0]
    approx_truth = self.truth_predictor(curr_state)

    p_violation = self.violation_predictor(torch.cat((encoded_spec, approx_truth), dim=1))
    prog_spec = self.rnn(approx_truth, encoded_spec)
    return p_violation, prog_spec


class SafetyDDQNAgent(Agent):
  def __init__(self, state_size, action_size, spec_encoding_size, params: SafetyDDQNParams) -> None:
    super().__init__()

    self.state_size = state_size
    self.action_size = np.prod(action_size)
    self.spec_encoding_size = spec_encoding_size
    self.params = params

    if isinstance(state_size, tuple):
      assert params.encoding_size is not None
      assert params.cnn_kernels is not None
      assert params.cnn_kernels is not None
      
      self.multi_dim_input = True
      self._state_encoder = Basic2dEncoder(
          state_size, params.encoding_size, params.cnn_kernels, params.cnn_channels
        ).to(self.params.device)
      input_shape = params.encoding_size
    else:
      self.multi_dim_input = False
      input_shape = state_size

    # Initialise safety specification encoder
    self._spec_encoder = SpecEncoder(spec_encoding_size, params.spec_hidden_size).to(device=params.device)
    self._spec_decoder = SpecDecoder(2 * params.spec_hidden_size, spec_encoding_size, params.num_props).to(device=params.device)
    # self._safety_rnn = SafetyRNN(2 * params.spec_hidden_size, input_shape, self.action_size, params.rnn_hidden_size).to(device=params.device)
    # self._ca_shield = ConditionalAdaptiveShield(2 * params.spec_hidden_size, input_shape, self.action_size, input_shape, hidden_size=params.rnn_hidden_size).to(device=params.device)
    self._ca_shield = ConditionalAdaptiveShield(2 * params.spec_hidden_size, input_shape, self.action_size, input_shape).to(device=params.device)
    self._policy_net = QModel(input_shape + (2 * params.spec_hidden_size), self.action_size, params.nn_sizes).to(params.device)
    self._target_net = QModel(input_shape + (2 * params.spec_hidden_size), self.action_size, params.nn_sizes).to(params.device)
    self._target_net.load_state_dict(self._policy_net.state_dict())
    
    if self.multi_dim_input:
      self._optimizer = torch.optim.Adam(itertools.chain(
        self._state_encoder.parameters(), self._policy_net.parameters(),
        self._spec_encoder.parameters(), self._spec_decoder.parameters(),
        self._ca_shield.parameters()), lr=params.lr)
    else:
      self._optimizer = torch.optim.Adam(itertools.chain(
        self._policy_net.parameters(), self._spec_encoder.parameters(),
        self._spec_decoder.parameters(), self._ca_shield.parameters()), lr=params.lr)
    
    self._exp_replay = EpisodicExperienceReplay(tuple_shape=SAFETY_TRANSITION_TUPLE_SIZE, buffer_size=params.buff_size)

    self.metrics = {
      "episode_rewards": [],
      "train_losses": [],
      "q_losses": [],
      "vio_losses": [],
      "recon_losses": [],
      "cum_num_violations": [],
      "steps": [],
      "metric_titles": {
        "episode_rewards": "Total Training Episode Reward",
        "train_losses": "Loss Function",
        "q_losses": "TD-Learning Loss",
        "vio_losses": "Violation Prediction Loss",
        "recon_losses": "LTL Reconstruction Loss",
        "cum_num_violations": "Cumulative Number of Violations"
      }
    }
  
  def _idx_to_one_hot(self, idx, max_idx):
    vec = torch.zeros((1, max_idx), device=self.params.device, dtype=torch.long)
    vec[0, idx] = 1
    return vec
  
  def _to_one_hot(self, a_vec):
    max_idx = torch.argmax(a_vec)
    a = torch.zeros((1, np.prod(a_vec.shape)), device=self.params.device, dtype=torch.long)
    a[0, max_idx] = 1
    return a

  # def _get_safe_action(self, state: SafetyState, rnn_state: Optional[torch.Tensor] = None, eps=0):
  def _get_safe_action(self, state: SafetyState, eps=0):
    act_randomly = False
    
    if eps > 0:
      sample = random.random()
      if sample < eps:
        act_randomly = True
    
    with torch.no_grad():
      encoded_formula = self._spec_encoder(state.formula)
      if self.multi_dim_input:
        encoded_state = self._state_encoder(state.env_state)
      else:
        encoded_state = state.env_state

      if not act_randomly:
        q_vals = self._policy_net_pass(state)
        best_action_idxs = torch.argsort(q_vals, descending=True).squeeze()

        for i in range(self.action_size):
          proposed_action = self._idx_to_one_hot(best_action_idxs[i], self.action_size)
          # p_violation, prog_spec, new_rnn_state = self._ca_shield(encoded_formula, encoded_state, proposed_action, hidden_state=rnn_state)
          p_violation, prog_spec = self._ca_shield(encoded_formula, encoded_state, proposed_action)
          if p_violation < self.params.violation_threshold:
            # return proposed_action, prog_spec, new_rnn_state
            return proposed_action, prog_spec
      
      # No safe actions or epsilon exploration so act randomly
      rand_action = self._idx_to_one_hot(random.randrange(self.action_size), self.action_size)
      # _, prog_spec, new_rnn_state = self._ca_shield(encoded_formula, encoded_state, rand_action, hidden_state=rnn_state)
      _, prog_spec = self._ca_shield(encoded_formula, encoded_state, rand_action)
      return rand_action, prog_spec

  def _get_action(self, state, eps=0):
    if eps > 0:
      sample = random.random()
      if sample < eps:
        return self._idx_to_one_hot(random.randrange(self.action_size), self.action_size)
    
    with torch.no_grad():
      return self._to_one_hot(self._policy_net_pass(state))

  def choose_action(self, state, info, eps=0):
    tensor_state = state.to(self.params.device)
    safety_spec_strs = safety_spec_to_str(info["prog_formula"] if "prog_formula" in info else "True")
    safety_spec = get_one_hot_spec(safety_spec_strs, info["num_props"] if "prog_formula" in info else 0).to(device=self.params.device)

    comb_state = SafetyState(
      tensor_state if len(tensor_state.shape) == 2 else tensor_state.unsqueeze(0),
      safety_spec.unsqueeze(1)
    )
    return self._get_action(comb_state, eps=eps).squeeze().detach().cpu()

  def _update_target_network(self):
    self._target_net.load_state_dict(self._policy_net.state_dict())
    # if self.multi_dim_input:
    #   self._target_net_encoder.load_state_dict(self._policy_net_encoder.state_dict())

  def _policy_net_pass(self, state: SafetyState):
    encoded_formula = self._spec_encoder(state.formula)

    if self.multi_dim_input:
      encoded_state = self._state_encoder(state.env_state)
      return self._policy_net(torch.cat((encoded_state, encoded_formula), dim=1))
    else:
      return self._policy_net(torch.cat((state.env_state, encoded_formula), dim=1))

  def _target_net_pass(self, state: SafetyState):
    encoded_formula = self._spec_encoder(state.formula)

    if self.multi_dim_input:
      encoded_state = self._state_encoder(state.env_state)
      return self._target_net(torch.cat((encoded_state, encoded_formula), dim=1))
    else:
      return self._target_net(torch.cat((state.env_state, encoded_formula), dim=1))

  def _optimize_model(self):
    if len(self._exp_replay) < self.params.batch_size:
      return None

    transitions = self._exp_replay.get_sample(self.params.batch_size)
    batches = SafetyTransition(*zip(*transitions))

    state_batch = torch.cat(batches.state)
    pack_spec_batch = nn.utils.rnn.pack_sequence(batches.safety_spec, enforce_sorted=False)
    action_batch = torch.cat(batches.action)
    reward_batch = torch.cat(batches.reward)
    next_state_batch = torch.cat(batches.next_state)
    non_terminal_mask = torch.logical_not(torch.tensor(batches.done))

    # Calculate TD Loss

    if torch.sum(non_terminal_mask) > 0:
      non_terminal_next_states = next_state_batch[torch.nonzero(non_terminal_mask, as_tuple=True)]
      # non_terminal_next_states = torch.tensor([next_state_batch[i] for i in range(self.params.batch_size) if non_terminal_mask[i]])
      non_terminal_prog_specs = nn.utils.rnn.pack_sequence(
        [batches.prog_spec[i] for i in range(self.params.batch_size) if non_terminal_mask[i]],
        enforce_sorted=False)
    else:
      non_terminal_next_states = torch.empty(0, self.state_size, device=self.params.device)
      non_terminal_prog_specs = nn.utils.rnn.pack_sequence([torch.empty(0, self.spec_encoding_size, device=self.params.device)])

    safety_state_batch = SafetyState(state_batch, pack_spec_batch)
    state_qs = self._policy_net_pass(safety_state_batch).gather(1, torch.argmax(action_batch, dim=1, keepdim=True))

    non_terminal_next_batch = SafetyState(non_terminal_next_states, non_terminal_prog_specs)
    next_state_vals = torch.zeros(self.params.batch_size, device=self.params.device)
    with torch.no_grad():
      if torch.sum(non_terminal_mask) > 0:
        argmax_q_idx = self._policy_net_pass(non_terminal_next_batch).argmax(1).detach()
        q_vals = self._target_net_pass(non_terminal_next_batch).detach()
        next_state_vals[torch.nonzero(non_terminal_mask, as_tuple=True)] = q_vals[range(q_vals.shape[0]), argmax_q_idx]
    
    expected_qs = (next_state_vals * self.params.gamma) + reward_batch

    q_loss = torch.mean((state_qs - expected_qs.unsqueeze(1)).pow(2))

    # Sample episode chunks
    chunks = self._exp_replay.get_sample_chunks(self.params.batch_size // self.params.chunk_size, self.params.chunk_size)
    chunks_by_seq = list(zip(*chunks))

    violation_loss = 0
    recon_loss = 0
    # safety_rnn_state = torch.zeros((self.params.batch_size // self.params.chunk_size, self.params.rnn_hidden_size), device=self.params.device)

    for step in range(self.params.chunk_size):
      chunk_batches = SafetyTransition(*zip(*chunks_by_seq[step]))
      state_batch = torch.cat(chunk_batches.state)
      pack_spec_batch = nn.utils.rnn.pack_sequence(chunk_batches.safety_spec, enforce_sorted=False)
      action_batch = torch.cat(chunk_batches.action)
      pad_prog_spec = nn.utils.rnn.pad_sequence(chunk_batches.prog_spec, batch_first=True)
      violation = torch.tensor(chunk_batches.violation, dtype=torch.float, device=self.params.device).unsqueeze(1)

      # Calculate Violation Loss
      encoded_specs = self._spec_encoder(pack_spec_batch)
      if self.multi_dim_input:
        encoded_state = self._state_encoder(state_batch)
      else:
        encoded_state = state_batch

      # p_violation, pred_prog_spec, safety_rnn_state = self._ca_shield(
      #   encoded_specs, encoded_state, action_batch, hidden_state=safety_rnn_state)
      p_violation, pred_prog_spec = self._ca_shield(encoded_specs, encoded_state, action_batch)

      class_weightings = torch.tensor(
        [self.params.violation_weighting_alpha, 1 - self.params.violation_weighting_alpha],
        device=self.params.device, dtype=torch.float)
      sample_weightings = class_weightings[violation.data.view(-1).long()].view_as(violation)

      bce = F.binary_cross_entropy(p_violation, violation, reduction="none")
      weighted_bce = bce * sample_weightings
      violation_loss += torch.mean(weighted_bce)

      # Calculate Reconstructed Spec Loss

      target_lens = [spec.shape[0] for spec in chunk_batches.prog_spec]
      max_len = max(target_lens)
      decoded_spec_logits = self._spec_decoder(pred_prog_spec, pad_prog_spec, max_len)

      padding_mask = torch.tensor(
        [[0 if i < target_lens[j] else -1 for i in range(max_len)] for j in range(self.params.batch_size // self.params.chunk_size)],
        dtype=torch.int,
        device=self.params.device
      )

      target_spec = torch.argmax(pad_prog_spec, dim=2) + padding_mask
      recon_loss += F.cross_entropy(decoded_spec_logits.permute(0, 2, 1), target_spec, reduction="mean", ignore_index=-1)
    
    # Backpropagation

    loss = ((self.params.q_loss_scale * q_loss) + 
      (self.params.vio_loss_scale * violation_loss) +
      (self.params.recon_loss_scale * recon_loss))

    self._optimizer.zero_grad()
    loss.backward()

    for param in self._policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    for param in self._spec_encoder.parameters():
      param.grad.data.clamp_(-1, 1)
    for param in self._spec_decoder.parameters():
      param.grad.data.clamp_(-1, 1)
    # TODO(PrabSG@): Check if want to clamp RNN grads

    if self.multi_dim_input:
      for param in self._state_encoder.parameters():
        param.grad.data.clamp_(-1, 1)

    self._optimizer.step()
    return self.params.q_loss_scale * q_loss.cpu().item(), self.params.vio_loss_scale * violation_loss.cpu().item(), recon_loss.cpu().item() * self.params.recon_loss_scale

  def train(self, env: SafetyConstrainedEnv, print_logging=False, writer=None):
    if print_logging:
      print("Training DDQN agent...")

    self.train_mode()

    optimize_steps = 0

    for i_episode in tqdm(range(self.params.episodes)):
      if print_logging and (i_episode+1) % 5 == 0:
        print(f"Agent exploring in episode {i_episode + 1}")
      env.reset(random_start=True)

      raw_safety_spec = env.get_safety_spec()
      safety_spec_strs = safety_spec_to_str(raw_safety_spec)
      safety_spec = get_one_hot_spec(safety_spec_strs, env.get_num_props()).to(device=self.params.device)

      total_reward = 0
      num_violations = 0
      state = env.get_observation().unsqueeze(0).to(self.params.device)
      comb_state = SafetyState(state, safety_spec.unsqueeze(1))
      # safety_rnn_state = None
      episode_transitions = []

      train_loss_ep = 0
      q_loss_ep = 0
      vio_loss_ep = 0
      recon_loss_ep = 0

      eps = self.params.eps_func(i_episode)
      for t in range(self.params.max_episode_len):
        with torch.no_grad():
          # action, _, safety_rnn_state  = self._get_safe_action(comb_state, rnn_state=safety_rnn_state, eps=eps)
          action, _  = self._get_safe_action(comb_state, eps=eps)

          next_state, reward, done, info = env.step(action.squeeze().detach().cpu())
          total_reward += reward
          next_state = next_state.unsqueeze(0).to(self.params.device)
          reward = torch.tensor([reward], device=self.params.device, dtype=torch.float)
          violation = info["violation"]
          num_violations += 1 if violation else 0
          prog_safety_spec = get_one_hot_spec(safety_spec_to_str(info["prog_formula"]), env.get_num_props()).to(device=self.params.device)

          episode_transitions.append(SafetyTransition(state, safety_spec, action, reward, next_state, prog_safety_spec, violation, done))

        loss = self._optimize_model()
        if loss is not None:
          q_loss, vio_loss, recon_loss = loss
          train_loss_ep += q_loss + vio_loss + recon_loss
          q_loss_ep += q_loss
          vio_loss_ep += vio_loss
          recon_loss_ep += recon_loss

          optimize_steps += 1
          if optimize_steps % self.params.update_steps == 0:
            self._update_target_network()
          
        if done:
          break
        state = next_state
        if not violation:
          # I.e rewind safety spec in case of violation
          safety_spec = prog_safety_spec
        comb_state = SafetyState(state, safety_spec.unsqueeze(1))

      self._exp_replay.add_episode(episode_transitions)

      self.metrics["steps"].append(t+1 + (0 if len(self.metrics["steps"]) == 0 else self.metrics["steps"][-1]))
      self.metrics["episode_rewards"].append(total_reward)
      self.metrics["cum_num_violations"].append(num_violations + (0 if len(self.metrics["cum_num_violations"]) == 0 else self.metrics["cum_num_violations"][-1]))
      self.metrics["train_losses"].append(train_loss_ep / t)
      self.metrics["q_losses"].append(q_loss_ep / t)
      self.metrics["vio_losses"].append(vio_loss_ep / t)
      self.metrics["recon_losses"].append(recon_loss_ep / t)

      if print_logging and (i_episode+1) % 5 == 0:
        print(f"Episode reward: {total_reward}")

      if writer is not None:
        writer.add_scalar("train_reward", self.metrics["episode_rewards"][-1], self.metrics["steps"][-1])
        writer.add_scalar("opt_steps/train_loss", self.metrics["train_losses"][-1], self.metrics["steps"][-1])
        writer.add_scalar("opt_steps/q_loss", self.metrics["q_losses"][-1], self.metrics["steps"][-1])
        writer.add_scalar("opt_steps/vio_loss", self.metrics["vio_losses"][-1], self.metrics["steps"][-1])
        writer.add_scalar("opt_steps/recon_loss", self.metrics["recon_losses"][-1], self.metrics["steps"][-1])
        writer.add_scalar("episodic/train_reward", self.metrics["episode_rewards"][-1], i_episode)
        writer.add_scalar("episodic/cum_num_violations", self.metrics["cum_num_violations"][-1], i_episode)
        writer.add_scalar("episodic/train_loss", self.metrics["train_losses"][-1], i_episode)
        writer.add_scalar("episodic/q_loss", self.metrics["q_losses"][-1], i_episode)
        writer.add_scalar("episodic/vio_loss", self.metrics["vio_losses"][-1], i_episode)
        writer.add_scalar("episodic/recon_loss", self.metrics["recon_losses"][-1], i_episode)
    
    env.close()
    return self.metrics

  def train_mode(self):
    self._policy_net.train()
    self._spec_encoder.train()
    self._spec_decoder.train()
    if self.multi_dim_input:
      self._state_encoder.train()

  def evaluate_mode(self):
    self._policy_net.eval()
    self._spec_encoder.eval()
    self._spec_decoder.eval()
    if self.multi_dim_input:
      self._state_encoder.eval()