from abc import ABC
import random

import gym
import numpy as np
import torch

from envs.env import Environment
from gym_minigrid import minigrid
from safety.monitor import SafetyMonitor

class SafetyConstrainedEnv(Environment, ABC):
  def get_safety_spec(self):
    pass

  def get_violation_count(self) -> int:
    pass

  def get_num_props(self) -> int:
    pass


class MiniGridSafetyEnv(SafetyConstrainedEnv):
  def __init__(self, env_key, violation_penalty=-1, seed=None, use_tensors=True, **kwargs) -> None:
    super().__init__()
    self.use_tensors = use_tensors
    self._env = gym.make(env_key, **kwargs)
    self._env.seed(seed)
    self._is_done = False    
    self._safety_monitor = SafetyMonitor(
      self._env.get_safety_spec() if hasattr(self._env, "get_safety_spec")
      else "True")
    self.violation_penalty = violation_penalty

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
      obs = self._env.gen_obs()

    obs_type = "one_hot" if "one_hot" in obs else "image"
    obs = obs[obs_type].transpose(2, 0, 1)
    return torch.tensor(obs, dtype=torch.float) if self.use_tensors else obs
  
  def step(self, action):
    enum_action = self._one_hot_to_action_enum(action)
    obs, reward, done, info = self._env.step(enum_action)

    # Check safety specifications are still satisfied
    violation, prog_formula = self._safety_monitor.step(info["true_props"] if "true_props" in info else set())
    info["violation"] = violation
    info["prog_formula"] = prog_formula
    if violation:
      reward += self.violation_penalty

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

    # Reset safety monitor with potentially new safety specification
    self._safety_monitor = SafetyMonitor(
      self._env.get_safety_spec() if hasattr(self._env, "get_safety_spec")
      else "True")

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
    return (len(minigrid.IDX_TO_OBJECT), *img_shape[:2])

  @staticmethod
  def _obj_idx_to_default_encoding(type_idx):
    obj_type = minigrid.IDX_TO_OBJECT[type_idx]

    if obj_type == 'empty':
        return (0, 0, 0)
    elif obj_type == 'unseen':
      return (1, 0, 0)

    if obj_type == 'wall':
        v = minigrid.Wall()
    elif obj_type == 'floor':
        v = minigrid.Floor()
    elif obj_type == 'ball':
        v = minigrid.Ball()
    elif obj_type == 'key':
        v = minigrid.Key()
    elif obj_type == 'box':
        v = minigrid.Box(color="purple")
    elif obj_type == 'door':
        v = minigrid.Door(color="blue")
    elif obj_type == 'goal':
        v = minigrid.Goal()
    elif obj_type == 'lava':
        v = minigrid.Lava()
    elif obj_type == 'agent':
        v = minigrid.Agent()
    elif obj_type == 'water':
        v = minigrid.Water()
    elif obj_type == 'glass':
        v = minigrid.Glass()
    else:
        assert False, "unknown object type in decode '%s'" % obj_type

    return v.encode()

  def get_safety_spec(self):
    return self._safety_monitor.get_safety_spec()

  def get_violation_count(self) -> int:
    return self._safety_monitor.violation_count
  
  def get_num_props(self) -> int:
    return len(self._env._safety_props) if hasattr(self._env, "_safety_props") else 0