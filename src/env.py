from abc import ABC, abstractmethod
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
    """Take given action in environment and return new state and reward emitted."""
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
  """2-D grid environment with single goal state.

    State Space:
      10x10 grid, no obstacles
    Goal:
      To reach a pre-defined goal state
    Action Space:
      (1, 4) one-hot encoded vector for moving in the 4 directions respectively: up, right, down,
      left. If the vector is not one-hot encoded with a discrete action, the argmax will be taken
      as the action to take.
  """
  
  _UP_DIR = 0
  _RIGHT_DIR = 1
  _DOWN_DIR = 2
  _LEFT_DIR = 3

  def __init__(self) -> None:
      super().__init__()
      self._action_size = (4)
      self._action_value_range = (0, 1)
      self._state_size = (10, 10)

      self._state = np.array([5, 5], dtype=np.float)
      self._goal_state = np.array([8, 9], dtype=np.float)

  def _get_state(self):
    return self._state

  def _get_next_state(self, init_state, action):
    a_direction = np.argmax(action)
    a = np.zeros_like(action)
    a[a_direction] = 1

    new_state = np.copy(init_state)
    new_state[0] += a[self._RIGHT_DIR] - a[self._LEFT_DIR]
    new_state[1] += a[self._UP_DIR] - a[self._DOWN_DIR]
    return np.clip(new_state, 0, 9)

  def _get_reward(self, new_state):
    return - np.linalg.norm(new_state - self._goal_state) ** 2

  def get_observation(self):
      return super().get_observation()

  def step(self, action):
    next_state = self._get_next_state(self._state, action)
    reward = self._get_reward(next_state)
    self._state = next_state
    return self._state, reward

  def is_complete(self):
    return np.all(np.isclose(self._state, self._goal_state))

  def reset(self):
    self._state = np.array([5, 5], dtype=np.float)

  @property
  def action_size(self):
    return self._action_size
  
  @property
  def action_value_range(self):
    return self._action_value_range

  @property
  def state_size(self):
    return self._state_size
