from abc import ABC, abstractmethod
import random

import numpy as np
import torch
import gym

def init_env(args):
  if args.env == "basic":
    return BasicEnv()
  elif args.env == "unsafe-micro":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingMicro-v0", max_steps=args.max_episode_length)
  elif args.env == "unsafe-small":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingN1-v0", max_steps=args.max_episode_length)
  elif args.env == "unsafe-med":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingN2-v0", max_steps=args.max_episode_length)
  else:
    raise ValueError(f"Environment Type '{args.env}' not defined.")

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
  def reset(self, random_start=False):
    pass

  @abstractmethod
  def close(self):
    pass

  @abstractmethod
  def sample_random_action(self):
    pass

  @property
  def action_size(self):
    pass
  
  @property
  def state_size(self):
    pass

class MiniGridEnvWrapper(Environment):

  def __init__(self, env_key, seed=None, use_tensors=True, **kwargs) -> None:
    super().__init__()
    self.use_tensors = use_tensors
    self._env = gym.make(env_key, **kwargs)
    self._env.seed(seed)
    self._is_done = False

  def __str__(self) -> str:
    return self._env.__str__()

  def _idx_to_action(self, idx):
    if idx == 0:
      return self._env.actions.left
    elif idx == 1:
      return self._env.actions.right
    elif idx == 2:
      return self._env.actions.forward
    elif idx == 3:
      return self._env.actions.pickup
    elif idx == 4:
      return self._env.actions.drop
    elif idx == 5:
      return self._env.actions.toggle
    elif idx == 6:
      return self._env.actions.done
    else:
      raise IndexError()

  def _one_hot_to_action_enum(self, action):
    if self.use_tensors:
      a_idx = torch.argmax(action).item()
    else:
      a_idx = np.argmax(action)
    return self._idx_to_action(a_idx)

  def _get_state(self):
    return self._env.grid

  def get_observation(self, obs=None):
    """Return image observation as C x H x W.
    
    Optionally pass in raw environment observation to convert to correct format.
    """
    if obs is None:
      obs = self._env.gen_obs()["image"].transpose(2, 0, 1)
    else:
      obs = obs["image"].transpose(2, 0, 1)
    return torch.tensor(obs, dtype=torch.float) if self.use_tensors else obs
  
  def step(self, action):
    enum_action = self._one_hot_to_action_enum(action)
    obs, reward, done, info = self._env.step(enum_action)
    self._is_done = done
    return self.get_observation(obs=obs), reward, done, info
  
  def is_complete(self):
    return self._is_done
  
  def reset(self, random_start=True, seed=None):
    """Reset Environment for next episode. If random_start is set to False, the same exact
      environment and configuration will be used, otherwise a new one will be generated from the
      seed."""
    
    if not random_start and seed is not None:
      self._env.seed(seed)
    
    self._env.reset()
    return self.get_observation()

  def close(self):
    self._env.close()

  def sample_random_action(self):
    actions = torch.zeros(self.action_size, dtype=torch.float)
    chosen = random.randrange(0, self.action_size)
    actions[chosen] = 1.0
    return actions if self.use_tensors else actions.numpy()

  @property
  def action_size(self):
    return len(self._env.actions)
  
  @property
  def state_size(self):
    """Return image state space shape as C x H x W."""
    img_shape = self._env.observation_space["image"].shape
    return (img_shape[2], *img_shape[:2])



class BasicEnv(Environment):
  """2-D grid environment with single goal state.

    State Space:
      10x10 grid, no obstacles
    Goal:
      To reach a pre-defined goal state at (8, 9)
    Action Space:
      (4) one-hot encoded vector for moving in the 4 directions respectively: up, right, down,
      left. If the vector is not one-hot encoded with a discrete action, the argmax will be taken
      as the action to take.
  """
  
  _UP_DIR = 0
  _RIGHT_DIR = 1
  _DOWN_DIR = 2
  _LEFT_DIR = 3

  GRID_SIZE = 10
  DEFAULT_START = [3, 3]

  GOAL_REWARD = 100
  PENALTY = -1

  def __init__(self) -> None:
      super().__init__()
      self._action_size = (4)
      self._state_size = (2)

      self._state = np.array([5, 5], dtype=np.float)
      self._goal_state = np.array([8, 7], dtype=np.float)

  def __str__(self) -> str:
    return self._state.__str__()

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
    # Classic time-based reward function
    if np.array_equal(new_state, self._goal_state):
      return self.GOAL_REWARD
    else:
      return self.PENALTY
    # Reward Shaping
    # return - np.linalg.norm(new_state - self._goal_state) ** 2

  def get_observation(self):
    return super().get_observation()

  def step(self, action):
    next_state = self._get_next_state(self._state, action)
    reward = self._get_reward(next_state)
    self._state = next_state
    return self._state, reward, self.is_complete(), {}

  def is_complete(self):
    return np.all(np.isclose(self._state, self._goal_state))

  def reset(self, random_start=False):
    if random_start:
      self._state = np.array([random.randrange(self.GRID_SIZE), random.randrange(self.GRID_SIZE)])
      while np.array_equal(self._goal_state, self._state):
        self._state = np.array([random.randrange(self.GRID_SIZE), random.randrange(self.GRID_SIZE)])
    else:
      self._state = np.array(self.DEFAULT_START, dtype=np.float)

  def close(self):
    pass

  def sample_random_action(self):
    actions = np.zeros(self.action_size)
    chosen = random.randrange(0, self.action_size)
    actions[chosen] = 1
    return actions

  @property
  def action_size(self):
    return self._action_size

  @property
  def state_size(self):
    return self._state_size
