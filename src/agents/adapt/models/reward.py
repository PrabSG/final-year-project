import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class RewardModel(nn.Module):
  def __init__(
    self, rssm_state_size, reward_shape, hidden_size, num_layers, dist="normal",
    activation=nn.ELU
  ):
    super().__init__()
    self.rssm_state_size = rssm_state_size
    self.reward_shape = reward_shape
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dist = dist
    self.activation = activation
    self.model = self.build_model()
    
  def build_model(self):
    layers = [nn.Linear(self.rssm_state_size, self.hidden_size), self.activation()]
    for _ in range(1, self.num_layers):
      layers += [nn.Linear(self.hidden_size, self.hidden_size), self.activation()]
    layers += [nn.Linear(self.hidden_size, int(np.prod(self.reward_shape)))]
    
    return nn.Sequential(*layers)
  
  def forward(self, state):
    rewards = self.model(state)
    reshaped_rewards = torch.reshape(rewards, state.shape[:-1] + self.reward_shape)
    if self.dist == "normal":
      return td.independent.Independent(td.Normal(reshaped_rewards, 1), len(self.reward_shape))
    else:
      raise NotImplementedError
