import random

import torch.nn as nn
import numpy as np

from agent import Agent
from utils import DimensionError

class ExperienceReplay():
  _BUFFER_SIZE = 1000

  def __init__(self, tuple_shape, buffer_size=_BUFFER_SIZE) -> None:
    self._buffer_size = buffer_size
    self._buffer = [None] * buffer_size
    self._idx = 0
    self._full = False
    self._shape = tuple_shape
  
  def add(self, exp_sample):
    if len(exp_sample) != self._shape:
      raise DimensionError(expected=self._shape, given=len(exp_sample))

    self._buffer[self._idx] = exp_sample

    self._idx += 1
    if self._idx == self._buffer_size:
      self._full = True
      self._idx = 0
  
  def __len__(self):
    if self._full:
      return self._buffer_size
    else:
      return self._idx
  
  def get_sample(self, sample_size):
    """Return uniformly random sample of experiences with replacement."""
    max_idx = self._buffer_size
    
    if not self._full:
      max_idx = self.idx
    
    return random.choices(self._buffer[:max_idx], k=sample_size)
