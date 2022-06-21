from collections import namedtuple
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from agents.agent import Agent
from envs.env import Environment
from replay import ExperienceReplay
from utils import exp_decay_epsilon

TRANSITION_TUPLE_SIZE = 5

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class Basic2dEncoder(nn.Module):
  def __init__(self, input_shape, output_shape, kernel_sizes, layer_channels):
    super().__init__()
    # Expect input shape to be (C x H x W)
    num_input_channels = 1 if len(input_shape) == 2 else input_shape[0]
    
    self.input_conv = nn.Sequential(
      nn.Conv2d(num_input_channels, layer_channels[0], kernel_size=kernel_sizes[0], padding="same"),
      nn.ReLU())
    self.hidden_convs = nn.ModuleList([
      nn.Sequential(
        nn.Conv2d(layer_channels[i], layer_channels[i+1], kernel_size=kernel_sizes[i+1], padding="same"),
        nn.ReLU())
      for  i in range(len(layer_channels) - 1)
    ])
    self.output_conv = nn.Sequential(
      nn.Conv2d(layer_channels[len(layer_channels) - 1], output_shape, kernel_size=input_shape[1:], padding=0),
      nn.ReLU())

  def forward(self, x):
    # Dims of x: (batch x C x H x W)
    x = self.input_conv(x)
    for conv in self.hidden_convs:
      x = conv(x)
    
    # Dims of output: (batch x output_shape x 1 x 1)
    output = self.output_conv(x)

    # Squeeze to match dims: (batch x output_shape)
    num_dims = len(output.shape)
    return output.squeeze(num_dims - 1).squeeze(num_dims - 2)

class QModel(nn.Module):
  def __init__(self, input_shape, output_shape, layer_sizes):
    super().__init__()

    num_hidden = len(layer_sizes)
    self.input_layer = nn.Linear(input_shape, layer_sizes[0])
    self.hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(num_hidden - 1)])
    self.output_layer = nn.Linear(layer_sizes[num_hidden - 1], output_shape)

  def forward(self, x):
    x = F.relu(self.input_layer(x))
    for layer in self.hidden_layers:
      x = F.relu(layer(x))

    return self.output_layer(x)

class DDQNParams():
  def __init__(self,
               num_episodes,
               max_episode_length,
               buffer_size,
               batch_size,
               hidden_layer_sizes,
               learning_rate,
               target_update_steps,
               gamma,
               eps_start,
               eps_end,
               eps_decay,
               encoding_size=None,
               cnn_kernels=None,
               cnn_channels=None,
               device="cpu"):
    self.episodes = num_episodes
    self.max_episode_len = max_episode_length
    self.buff_size = buffer_size
    self.batch_size = batch_size
    self.nn_sizes = hidden_layer_sizes
    self.lr = learning_rate
    self.update_steps = target_update_steps
    self.gamma = gamma
    self.eps_func = exp_decay_epsilon(eps_start, eps_end, eps_decay)
    self.encoding_size = encoding_size
    self.cnn_kernels = cnn_kernels
    self.cnn_channels = cnn_channels
    self.device = device

  def __str__(self):
    # TODO: Modify to include CNN encoder params
    return (f"Window size: {self.w_size}\n" +
            f"Number of episodes: {self.episodes}\n" +
            f"Replay buffer size: {self.buff_size}\n" +
            f"Batch size: {self.batch_size}\n" +
            f"Network layout: [{', '.join(map(str, self.nn_sizes))}]\n" +
            f"Learning rate: {self.lr}\n" +
            f"Gamma: {self.gamma}\n" +
            f"Update frequency: {str(self.update_steps)} steps\n" +
            f"Exponential epsilon: start - {self.eps_start}, end - {self.eps_end}, decay rate - {self.eps_decay}")

  def __repr__(self):
    # TODO: Modify to include CNN encoder params
    return (f"wSize-{self.w_size}_" +
            f"numEps-{self.episodes}_" +
            f"buffSize-{self.buff_size}_" +
            f"batchSize-{self.batch_size}_" +
            f"nnSizes-{'-'.join(map(str, self.nn_sizes))}_" +
            f"lr-{self.lr}_" +
            f"gamma-{self.gamma}_" +
            f"updateSteps-{str(self.update_steps)}_" +
            f"expEps-{self.eps_start}-{self.eps_end}-{self.eps_decay}")

  def __eq__(self, other):
    if isinstance(other, DDQNParams):
      return str(self) == str(other)
    else:
      return False
  
  def __hash__(self):
    return hash(self.__repr__())

class DDQNAgent(Agent):
  def __init__(self, state_size, action_size, params: DDQNParams) -> None:
    super().__init__()

    self.state_size = state_size
    self.action_size = np.prod(action_size)
    self.params = params

    if isinstance(state_size, tuple):
      assert params.encoding_size is not None
      assert params.cnn_kernels is not None
      assert params.cnn_kernels is not None
      
      self.multi_dim_input = True
      self._policy_net_encoder = Basic2dEncoder(
          state_size, params.encoding_size, params.cnn_kernels, params.cnn_channels
        ).to(self.params.device)
      self._target_net_encoder = Basic2dEncoder(
          state_size, params.encoding_size, params.cnn_kernels, params.cnn_channels
        ).to(self.params.device)
      self._target_net_encoder.load_state_dict(self._policy_net_encoder.state_dict())
      input_shape = params.encoding_size
    else:
      self.multi_dim_input = False
      input_shape = state_size


    self._policy_net = QModel(input_shape, self.action_size, params.nn_sizes).to(self.params.device)
    self._target_net = QModel(input_shape, self.action_size, params.nn_sizes).to(self.params.device)
    self._target_net.load_state_dict(self._policy_net.state_dict())
    
    if self.multi_dim_input:
      self._optimizer = torch.optim.Adam(
        itertools.chain(self._policy_net_encoder.parameters(), self._policy_net.parameters()), lr=params.lr)
    else:
      self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=params.lr)
    
    self._exp_replay = ExperienceReplay(tuple_shape=TRANSITION_TUPLE_SIZE, buffer_size=params.buff_size)

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
    return self._get_action(tensor_state, eps=eps).squeeze().detach().cpu()

  def _update_target_network(self):
    self._target_net.load_state_dict(self._policy_net.state_dict())
    if self.multi_dim_input:
      self._target_net_encoder.load_state_dict(self._policy_net_encoder.state_dict())

  def _policy_net_pass(self, state):
    if self.multi_dim_input:
      encoded_state = self._policy_net_encoder(state)
      return self._policy_net(encoded_state)
    else:
      return self._policy_net(state)

  def _target_net_pass(self, state):
    if self.multi_dim_input:
      encoded_state = self._target_net_encoder(state)
      return self._target_net(encoded_state)
    else:
      return self._target_net(state)

  def _optimize_model(self):
    if len(self._exp_replay) < self.params.batch_size:
      return None

    transitions = self._exp_replay.get_sample(self.params.batch_size)
    batches = Transition(*zip(*transitions))

    state_batch = torch.cat(batches.state)
    action_batch = torch.cat(batches.action)
    reward_batch = torch.cat(batches.reward)
    next_state_batch = torch.cat(batches.next_state)
    non_terminal_mask = torch.logical_not(torch.tensor(batches.done))

    if torch.sum(non_terminal_mask) > 0:
      non_terminal_next_states = next_state_batch[torch.nonzero(non_terminal_mask, as_tuple=True)]
      # non_terminal_next_states = torch.cat([next_state_batch[i].unsqueeze(dim=0) for i in range(self.params.batch_size) if non_terminal_mask[i]])
    else:
      non_terminal_next_states = torch.empty(0, self.state_size, device=self.params.device)

    state_qs = self._policy_net_pass(state_batch).gather(1, torch.argmax(action_batch, dim=1, keepdim=True))

    next_state_vals = torch.zeros(self.params.batch_size, device=self.params.device)
    with torch.no_grad():
      if torch.sum(non_terminal_mask) > 0:
        argmax_q_idx = self._policy_net_pass(non_terminal_next_states).argmax(1).detach()
        q_vals = self._target_net_pass(non_terminal_next_states).detach()
        next_state_vals[torch.nonzero(non_terminal_mask, as_tuple=True)] = q_vals[range(q_vals.shape[0]), argmax_q_idx]
    
    expected_qs = (next_state_vals * self.params.gamma) + reward_batch

    loss = torch.mean((state_qs - expected_qs.unsqueeze(1)).pow(2))
    self._optimizer.zero_grad()
    loss.backward()

    for param in self._policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    
    if self.multi_dim_input:
      for param in self._policy_net_encoder.parameters():
        param.grad.data.clamp_(-1, 1)
    
    self._optimizer.step()
    return loss.item()

  def train(self, env: Environment, print_logging=False, writer=None):
    if print_logging:
      print("Training DDQN agent...")

    self.train_mode()

    optimize_steps = 0
    

    for i_episode in range(self.params.episodes):
      if print_logging and (i_episode+1) % 5 == 0:
        print(f"Agent exploring in episode {i_episode + 1}")
      env.reset(random_start=True)

      total_reward = 0
      num_violations = 0
      q_loss_ep = 0
      state = env.get_observation().unsqueeze(0).to(self.params.device)

      self.metrics["train_losses"].append(0)

      for t in range(self.params.max_episode_len):
        with torch.no_grad():
          eps = self.params.eps_func(i_episode)
          action = self._get_action(state, eps=eps)

          next_state, reward, done, info = env.step(action.squeeze().detach().cpu())
          total_reward += reward
          next_state = next_state.unsqueeze(0).to(self.params.device)
          reward = torch.tensor([reward], device=self.params.device, dtype=torch.float)
          violation = info["violation"] if "violation" in info else False
          num_violations += 1 if violation else 0
          self._exp_replay.add(Transition(state, action, reward, next_state, done))

        loss = self._optimize_model()
        if loss is not None:
          q_loss_ep += loss

          optimize_steps += 1
          if optimize_steps % self.params.update_steps == 0:
            self._update_target_network()
          
        if done:
          break
        state = next_state        

      self.metrics["steps"].append(t+1 + (0 if len(self.metrics["steps"]) == 0 else self.metrics["steps"][-1]))
      self.metrics["episode_rewards"].append(total_reward)
      self.metrics["cum_num_violations"].append(num_violations + (0 if len(self.metrics["cum_num_violations"]) == 0 else self.metrics["cum_num_violations"][-1]))
      self.metrics["train_losses"].append(q_loss_ep / t)
      self.metrics["q_losses"].append(q_loss_ep / t)
      
      if print_logging and (i_episode+1) % 5 == 0:
        print(f"Episode reward: {total_reward}")

      if writer is not None:
        writer.add_scalar("opt_steps/train_reward", self.metrics["episode_rewards"][-1], self.metrics["steps"][-1])
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
    if self.multi_dim_input:
      self._policy_net_encoder.train()

  def evaluate_mode(self):
    self._policy_net.eval()
    if self.multi_dim_input:
      self._policy_net_encoder.eval()