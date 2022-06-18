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
  def __init__(self, num_env_props, *args, q_loss_scale=10, spec_encoding_hidden_size=64, violation_weighting_alpha=0.25, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_props = num_env_props
    self.q_loss_scale = q_loss_scale
    self.spec_hidden_size = spec_encoding_hidden_size
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


class DDQNL(Agent):
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
    self._policy_net = QModel(input_shape + (2 * params.spec_hidden_size), self.action_size, params.nn_sizes).to(params.device)
    self._target_net = QModel(input_shape + (2 * params.spec_hidden_size), self.action_size, params.nn_sizes).to(params.device)
    self._target_net.load_state_dict(self._policy_net.state_dict())
    
    if self.multi_dim_input:
      self._optimizer = torch.optim.Adam(
        itertools.chain(self._state_encoder.parameters(), self._policy_net.parameters(), self._spec_encoder.parameters()), lr=params.lr)
    else:
      self._optimizer = torch.optim.Adam(itertools.chain(self._policy_net.parameters(), self._spec_encoder.parameters()), lr=params.lr)
    
    self._exp_replay = EpisodicExperienceReplay(tuple_shape=SAFETY_TRANSITION_TUPLE_SIZE, buffer_size=params.buff_size)

    self.metrics = {
      "episode_rewards": [],
      "train_losses": [],
      "q_losses": [],
      "cum_num_violations": [],
      "steps": [],
      "metric_titles": {
        "episode_rewards": "Total Training Episode Reward",
        "train_losses": "Loss Function",
        "q_losses": "TD-Learning Loss",
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

    # Backpropagation

    loss = self.params.q_loss_scale * q_loss

    self._optimizer.zero_grad()
    loss.backward()

    for param in self._policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    for param in self._spec_encoder.parameters():
      param.grad.data.clamp_(-1, 1)

    if self.multi_dim_input:
      for param in self._state_encoder.parameters():
        param.grad.data.clamp_(-1, 1)

    self._optimizer.step()
    return self.params.q_loss_scale * q_loss.cpu().item()

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
      episode_transitions = []

      train_loss_ep = 0
      q_loss_ep = 0

      for t in range(self.params.max_episode_len):
        with torch.no_grad():
          eps = self.params.eps_func(i_episode)
          action = self._get_action(comb_state, eps=eps)

          next_state, reward, done, info = env.step(action.squeeze().detach().cpu())
          total_reward += reward
          next_state = next_state.unsqueeze(0).to(self.params.device)
          reward = torch.tensor([reward], device=self.params.device, dtype=torch.float)
          violation = info["violation"]
          num_violations += 1 if violation else 0
          prog_safety_spec = get_one_hot_spec(safety_spec_to_str(info["prog_formula"]), env.get_num_props()).to(device=self.params.device)

          episode_transitions.append(SafetyTransition(state, safety_spec, action, reward, next_state, prog_safety_spec, violation, done))

        q_loss = self._optimize_model()
        if q_loss is not None:
          train_loss_ep += q_loss
          q_loss_ep += q_loss

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

      if print_logging and (i_episode+1) % 5 == 0:
        print(f"Episode reward: {total_reward}")

      if writer is not None:
        writer.add_scalar("train_reward", self.metrics["episode_rewards"][-1], self.metrics["steps"][-1])
        writer.add_scalar("opt_steps/train_loss", self.metrics["train_losses"][-1], self.metrics["steps"][-1])
        writer.add_scalar("opt_steps/q_loss", self.metrics["q_losses"][-1], self.metrics["steps"][-1])
        writer.add_scalar("episodic/train_reward", self.metrics["episode_rewards"][-1], i_episode)
        writer.add_scalar("episodic/cum_num_violations", self.metrics["cum_num_violations"][-1], i_episode)
        writer.add_scalar("episodic/train_loss", self.metrics["train_losses"][-1], i_episode)
        writer.add_scalar("episodic/q_loss", self.metrics["q_losses"][-1], i_episode)
    
    env.close()
    return self.metrics

  def train_mode(self):
    self._policy_net.train()
    self._spec_encoder.train()
    if self.multi_dim_input:
      self._state_encoder.train()

  def evaluate_mode(self):
    self._policy_net.eval()
    self._spec_encoder.eval()
    if self.multi_dim_input:
      self._state_encoder.eval()