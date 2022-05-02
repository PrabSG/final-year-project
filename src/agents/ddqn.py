from collections import namedtuple
import random

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

class QModel(nn.Module):
  def __init__(self, num_inputs, num_outputs, nn_sizes):
    super().__init__()
    num_hidden = len(nn_sizes)
    self.input_layer = nn.Linear(num_inputs, nn_sizes[0])
    self.hidden_layers = nn.ModuleList([nn.Linear(nn_sizes[i], nn_sizes[i+1]) for i in range(num_hidden - 1)])
    self.output_layer = nn.Linear(nn_sizes[num_hidden - 1], num_outputs)

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
    self.device = device

  def __str__(self):
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
    self.state_size = np.prod(state_size)
    self.action_size = np.prod(action_size)
    self.params = params
    self._policy_net = QModel(self.state_size, self.action_size, params.nn_sizes).to(self.params.device)
    self._optimizer = torch.optim.Adam(self._policy_net.parameters(), lr=params.lr)
    self._target_net = QModel(self.state_size, self.action_size, params.nn_sizes).to(self.params.device)
    self._target_net.load_state_dict(self._policy_net.state_dict())
    self._exp_replay = ExperienceReplay(tuple_shape=TRANSITION_TUPLE_SIZE, buffer_size=params.buff_size)
  
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
      return self._to_one_hot(self._policy_net(state))

  def choose_action(self, state, eps=0):
    tensor_state = torch.tensor(state, device=self.params.device, dtype=torch.float)
    return self._get_action(tensor_state, eps=eps).squeeze().detach().cpu().numpy()

  def _update_target_network(self):
    self._target_net.load_state_dict(self._policy_net.state_dict())

  def _optimize_model(self):
    if len(self._exp_replay) < self.params.batch_size:
      return

    transitions = self._exp_replay.get_sample(self.params.batch_size)
    batches = Transition(*zip(*transitions))

    state_batch = torch.cat(batches.state)
    action_batch = torch.cat(batches.action)
    reward_batch = torch.cat(batches.reward)
    next_state_batch = torch.cat(batches.next_state)
    non_terminal_mask = np.logical_not(np.array(batches.done))

    if np.sum(non_terminal_mask) > 0:
      non_terminal_next_states = torch.cat([next_state_batch[i].unsqueeze(dim=0) for i in range(self.params.batch_size) if non_terminal_mask[i]])
    else:
      non_terminal_next_states = torch.empty(0, self.state_size)

    state_qs = self._policy_net(state_batch).gather(1, torch.argmax(action_batch, dim=1, keepdim=True))

    next_state_vals = torch.zeros(self.params.batch_size, device=self.params.device)
    with torch.no_grad():
      if np.sum(non_terminal_mask) > 0:
        argmax_q_idx = self._policy_net(non_terminal_next_states).argmax(1).detach()
        q_vals = self._target_net(non_terminal_next_states).detach()
        next_state_vals[non_terminal_mask] = q_vals[range(q_vals.shape[0]), argmax_q_idx]
    
    expected_qs = (next_state_vals * self.params.gamma) + reward_batch

    loss = torch.mean((state_qs - expected_qs.unsqueeze(1)).pow(2))
    self._optimizer.zero_grad()
    loss.backward()

    for param in self._policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    
    self._optimizer.step()
    return loss

  def train(self, env: Environment):
    self._policy_net.train()

    optimize_steps = 0
    episode_rewards = []
    train_losses = []

    for i_episode in range(self.params.episodes):
      env.reset(random_start=True)


      total_reward = 0
      state = torch.tensor([env.get_observation()], device=self.params.device, dtype=torch.float)
      for t in range(self.params.max_episode_len):
        eps = self.params.eps_func(i_episode)
        action = self._get_action(state, eps=eps)

        next_state, reward, done = env.step(action.squeeze().detach().cpu().numpy())
        total_reward += reward
        next_state = torch.tensor([next_state], device=self.params.device, dtype=torch.float)
        reward = torch.tensor([reward], device=self.params.device, dtype=torch.float)

        self._exp_replay.add(Transition(state, action, reward, next_state, done))

        loss = self._optimize_model()
        train_losses.append(loss)

        optimize_steps += 1
        if optimize_steps % self.params.update_steps == 0:
          self._update_target_network()
        
        if done:
          break
        state = next_state        

      episode_rewards.append(total_reward)
    
    env.close()
    return self.params.episodes, episode_rewards, optimize_steps, train_losses

  def evaluate(self):
    self._policy_net.eval()