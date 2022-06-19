from typing import Optional

import numpy as np

class RootPDistShiftScheduler:
  def __init__(self, num_categories: int, t_grow: int, sharpness_delta: int,
               category_difficulties: np.ndarray, init_weights: Optional[np.ndarray] = None,
               final_weights: Optional[np.ndarray] = None):
    self._b = num_categories
    self._t_grow = t_grow
    self._delta = sharpness_delta
    self._ds = category_difficulties
    self._w0 = init_weights if init_weights is not None else np.ones(num_categories)
    self._wT = final_weights if final_weights is not None else np.ones(num_categories)
    # Calculated terms
    self._ps = self._delta ** self._ds
    self._lambda_0s = self._lambda_0()

  def _lambda_0(self):
    w0_sum = np.sum(self._w0)
    return self._w0 / (self._wT * w0_sum)


  def get_weightings(self, t: int):
    ones = np.ones(self._b)
    if t > self._t_grow: # Short circuit
      return self._wT * ones

    surd_term = (((1 - self._lambda_0s) ** self._ps) / self._t_grow) * t
    root = surd_term ** (1 / self._ps) # Inverting power becomes a root
    return self._wT * np.min([ones, root + self._lambda_0s], axis=0)

  def get_probabilities(self, t: int):
    weightings = self.get_weightings(t)
    return weightings / np.sum(weightings)
