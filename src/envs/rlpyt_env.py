import gym
import numpy as np
from rlpyt.envs.base import Env
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper


class EnvWrapper(Env):
  def __init__(self, env: Env):
    self.env = env

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError("attempted to get missing private attribute '{}'".format(name))
    return getattr(self.env, name)

  def step(self, action):
    o, r, d, info = self.env.step(action)
    return np.transpose(o, (2, 0, 1)).astype("float32"), r, d, info

  def reset(self):
    return np.transpose(self.env.reset(), (2, 0, 1)).astype("float32")

  @property
  def action_space(self):
    return self.env.action_space

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def horizon(self):
    return self.env.horizon

  def close(self):
    self.env.close()

class OneHotAction(EnvWrapper):
  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, IntBox) or isinstance(env.action_space, GymSpaceWrapper)
    super().__init__(env)
    self._dtype = np.float32

  @property
  def action_space(self):
    shape = (self.env.action_space.n,)
    space = FloatBox(low=0, high=1, shape=shape, dtype=self._dtype)
    space.sample = self._sample_action
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    if not np.allclose(reference, action, atol=1e6):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return super().step(index)

  def reset(self):
    return super().reset()

  def _sample_action(self):
    actions = self.env.action_space.n
    index = self.np_random.randint(0, actions)
    reference = np.zeros(actions, dtype=self._dtype)
    reference[index] = 1.0
    return reference