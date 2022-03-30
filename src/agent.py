from abc import ABC, abstractmethod
from ssl import SSL_ERROR_INVALID_ERROR_CODE

import numpy as np

class Agent(ABC):

  @abstractmethod
  def choose_action(self):
    pass

class RandomAgent(Agent):
  def __init__(self, env_state_size, env_action_size, env_action_value_range) -> None:
    super().__init__()
    self.state_size = env_state_size
    self.action_size = env_action_size
    self.action_min = env_action_value_range[0]
    self.action_max = env_action_value_range[1]

  def choose_action(self, state):
    return np.random.uniform(self.action_min, self.action_max, size=self.action_size)