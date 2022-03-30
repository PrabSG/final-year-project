from abc import ABC, abstractmethod
from symbol import global_stmt
import numpy as np


class Environment(ABC):
  
  @abstractmethod
  def _get_state(self):
    pass

  @abstractmethod
  def get_observation(self):
    return self._get_state()

  @abstractmethod
  def step(self, action):
    pass

  @abstractmethod
  def is_complete(self):
    pass

  @abstractmethod
  def reset(self):
    pass

  @property
  def action_size(self):
    pass
  
  @property
  def state_size(self):
    pass


class BasicEnv(Environment):
  
  def __init__(self) -> None:
      super().__init__()
      self._action_size = (2)
      self._action_value_range = (-1, 1)
      self._state_size = (10, 10)

      self._state = np.array([5, 5], dtype=np.float)
      self._goal_state = np.array([8, 9], dtype=np.float)

  def _get_state(self):
    return self._state

  def get_observation(self):
      return super().get_observation()

  def step(self, action):
    clipped_action = np.clip(action, -1, 1)
    reward = self._get_reward(self._state, clipped_action)
    self._state = self._state + clipped_action
    return self._state, reward

  def is_complete(self):
    return np.all(np.isclose(self._state, self._goal_state))

  def reset(self):
    self._state = np.array([5, 5], dtype=np.float)

  def _get_reward(self, init_state, action):
    result_state = init_state + action
    return - np.linalg.norm(result_state - self._goal_state) ** 2

  @property
  def action_size(self):
    return self._action_size
  
  @property
  def action_value_range(self):
    return self._action_value_range

  @property
  def state_size(self):
    return self._state_size
