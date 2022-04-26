from abc import ABC, abstractmethod
from ssl import SSL_ERROR_INVALID_ERROR_CODE

import numpy as np

class Agent(ABC):

  @abstractmethod
  def train(self, env):
    pass

  @abstractmethod
  def choose_action(self):
    pass

  @abstractmethod
  def evaluate(self):
    pass

class RandomAgent(Agent):
  def __init__(self, env_state_size, env_action_size) -> None:
    super().__init__()
    self.state_size = env_state_size
    self.action_size = env_action_size

  def train(self, env):
    pass

  def choose_action(self, state):
    return np.random.uniform(0, 1, size=self.action_size)

  def evaluate(self):
    pass