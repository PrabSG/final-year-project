import random
from typing import List, Tuple

import numpy as np

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
      max_idx = self._idx
    
    return random.choices(self._buffer[:max_idx], k=sample_size)


class EpisodicExperienceReplay():
  _BUFFER_SIZE = 1000

  def __init__(self, tuple_shape, buffer_size=_BUFFER_SIZE) -> None:
    self._buffer_size = buffer_size
    self._buffer = []
    self._ep_lens = np.array([], dtype=np.int64)
    self._num_eps = 0
    self._num_samples = 0
    self._shape = tuple_shape

  def __len__(self):
    return self._num_samples

  def add_episode(self, episode_samples: List[Tuple]):
    if len(episode_samples[0]) != self._shape:
      raise DimensionError(expected=self._shape, given=len(episode_samples[0]))
    
    new_ep_len = len(episode_samples)
    while new_ep_len + self._num_samples > self._buffer_size:
      self._evict_oldest_episode()
    
    self._buffer.append(episode_samples)
    self._ep_lens = np.append(self._ep_lens, new_ep_len)
    self._num_eps += 1
    self._num_samples += new_ep_len

  def get_sample(self, sample_size):
    """Return uniformly random sample of experiences with replacement."""
    chosen_episode_idxs = random.choices(list(range(self._num_eps)), weights=self._ep_lens, k=sample_size)
    return [self._buffer[idx][random.randrange(0, self._ep_lens[idx])] for idx in chosen_episode_idxs]

  def get_sample_chunks(self, num_chunks, chunk_size):
    """
    Return uniformly random sample of contiguous chunks of experiences (within the same episode)
    with replacement.
    """
    adj_weights = self._ep_lens - chunk_size + 1
    chosen_episode_idxs = random.choices(list(range(self._num_eps)), weights=adj_weights, k=num_chunks)
    chunk_start_idxs = [random.randrange(0, self._ep_lens[idx] - chunk_size + 1) for idx in chosen_episode_idxs]
    chunks = [
      self._buffer[chosen_episode_idxs[i]][chunk_start_idxs[i] : chunk_start_idxs[i] + chunk_size]
      for i in range(num_chunks)
    ]

    return chunks

  def _evict_oldest_episode(self):
    self._buffer.pop(0)
    oldest_ep_len = self._ep_lens[0]
    self._ep_lens = self._ep_lens[1:]
    self._num_eps -= 1
    self._num_samples -= oldest_ep_len
