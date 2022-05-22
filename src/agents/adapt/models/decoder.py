from typing import Tuple

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class DecoderModel(nn.Module):
  def __init__(
    self,
    embedding_size: int,
    obs_shape: Tuple[int, int, int],
    activation_func=nn.ReLU
  ) -> None:
    super().__init__()
    self.obs_shape = obs_shape

    self.linear = nn.Sequential(nn.Linear(embedding_size, 128), activation_func())
    self.decoder = nn.Sequential(
      nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1),
      activation_func(),
      nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1),
      activation_func(),
      nn.ConvTranspose2d(32, obs_shape[0], kernel_size=3, stride=1)
    )

  def forward(self, x):
    batch_shape = x.shape[:-1]
    embed_size = x.shape[-1]
    batch_size = np.prod(batch_shape).item()
    x = x.reshape(batch_size, embed_size)
    
    x = self.linear(x)
    x = torch.reshape(x, (batch_shape, 128, 1, 1))
    x = self.decoder(x)
    mean = torch.reshape(*batch_shape, *self.obs_shape)
    obs_dist = td.Independent(td.Normal(mean, 1), len(self.obs_shape))
    return obs_dist

    