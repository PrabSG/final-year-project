from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

class Agent(ABC):

  @abstractmethod
  def train(self, env) -> Dict:
    pass

  @abstractmethod
  def choose_action(self, observation, info):
    pass

  def run_tests(self, n_episodes, env, args, print_logging=False, visualise=False, episode=None):
    self.evaluate_mode()

    total_rewards = []
    if visualise:
      frames = []

    for i_episode in range(n_episodes):
      print(f"Test Episode {i_episode+1}")
      env.reset()

      timestep = 0

      trace = []
      total_rewards.append(0)

      observation = env.get_observation()
      info = env.get_info()
      str_grid = str(env)

      while not env.is_complete() and timestep <= args.max_episode_length:
        if visualise:
          frames.append(np.moveaxis(env._env.render("rgb_array"), 2, 0))

        action = self.choose_action(observation, info)
        new_observation, reward, done, info = env.step(action)
        total_rewards[i_episode] += reward

        if print_logging:
          new_str_grid = str(env)
          trace.append((str_grid, action, reward, new_str_grid))
          str_grid = new_str_grid

        timestep += 1
        observation = new_observation

      if print_logging:
        for s, a, r, new_s in trace:
          print("State:\n", s, "\nAction:", a, "Reward:", r, "\nNew State:\n", new_s)
    
    self.train_mode()

    return total_rewards, frames if visualise else []

  @abstractmethod
  def train_mode(self):
    pass

  @abstractmethod
  def evaluate_mode(self):
    pass

class RandomAgent(Agent):
  def __init__(self, env_state_size, env_action_size) -> None:
    super().__init__()
    self.state_size = env_state_size
    self.action_size = env_action_size

  def train(self, env):
    return {}

  def choose_action(self, state, info):
    return np.random.uniform(0, 1, size=self.action_size)

  def train_mode(self):
    pass

  def evaluate_mode(self):
    pass