import numpy as np
import torch.distributions as td
import torch.nn as nn

class ActionModel(nn.Module):
  def __init__(
    self, rssm_state_size, action_size, hidden_size, num_layers, action_dist,
    activation=nn.ELU, min_std=1e-4, init_std=5, mean_scale=5
    ):
    super().__init__()
    self.rssm_state_size = rssm_state_size
    self.action_size = action_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.action_dist = action_dist
    self.activation = activation
    self.min_std = min_std
    self.init_std = init_std
    self.mean_scale = mean_scale
    self.model = self.build_model()
    self.raw_init_std = np.log(np.exp(self.init_std) - 1)
    
  def build_model(self):
    layers = [nn.Linear(self.rssm_state_size, self.hidden_size), self.activation()]
    for _ in range(1, self.num_layers):
      layers += [nn.Linear(self.hidden_size, self.hidden_size), self.activation()]
    if self.action_dist == "tanh_normal":
      layers += [nn.Linear(self.hidden_size, 2 * self.action_size)]
    elif self.action_dist == "one_hot":
      layers += [nn.Linear(self.hidden_size, self.action_size)]
    else:
      raise NotImplementedError
    
    return nn.Sequential(*layers)

  def forward(self, state):
    x = self.model(state)
    dist = None
    if self.action_dist == "tanh_normal":
      raise NotImplementedError
    elif self.action_dist == "one_hot":
      dist = td.OneHotCategorical(logits=x)
    else:
      raise NotImplementedError
    return dist
