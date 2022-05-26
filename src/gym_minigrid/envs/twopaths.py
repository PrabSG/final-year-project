from enum import IntEnum
import numpy as np
from gym_minigrid import minigrid
from gym_minigrid.minigrid import Goal, Grid, Lava, MiniGridEnv
from gym_minigrid.register import register

MAX_EPISODE_STEPS = 50

class TwoPathEnv(MiniGridEnv):
  class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
  
  def __init__(
    self,
    seed=None,
    **kwargs
  ):
    self.obstacle_type = "lava"
    grid_size=7
    super().__init__(grid_size=grid_size, seed=seed, agent_view_size=5, **kwargs)
    self.actions = TwoPathEnv.Actions

  def _gen_grid(self, width, height):
    self.grid = Grid(width, height)
    self.grid.wall_rect(0, 0, width, height)
    self.grid.vert_wall((width // 2) - 1, 1, length=5)
    self.grid.vert_wall((width // 2) + 1, 1, length=5)
    self.agent_pos = (width // 2, height // 2)
    self.agent_dir = 0 # Right

    rand_pos = int(self._rand_bool())

    self.put_obj(Goal(), 2, rand_pos * 4)
    self.put_obj(Lava(), 2, (1 - rand_pos) * 4)

    self.mission = (
      f"avoid the lava and get to the green goal square"
    )

  def _reward(self, done=True, violation=False):
    if done:
      if violation:
        return -1
      else:
        return 1
    else:
      return 0

  def gen_obs(self):
    obs = super().gen_obs()
    one_hot_image = self._to_one_hot_obs(obs["image"])
    obs["one_hot"] = one_hot_image
    return obs

  def _to_one_hot_obs(self, obs):
    obs_shape = obs.shape
    one_hot_obs = np.zeros((obs_shape[0], obs_shape[1], len(minigrid.IDX_TO_OBJECT)))

    for i in range(obs_shape[0]):
      for j in range(obs_shape[1]):
        one_hot_obs[i, j] = self._obj_encoding_to_one_hot(obs[i, j])

    return one_hot_obs

  def _obj_encoding_to_one_hot(self, encoded_obj):
    one_hot_obj = np.zeros((len(minigrid.IDX_TO_OBJECT)))
    one_hot_obj[encoded_obj[0]] = 1
    return one_hot_obj
  

class TwoPathEnvSimple(TwoPathEnv):
  def __init__(self, seed=None, **kwargs):
    super().__init__(seed, **kwargs)

register(
  id="MiniGrid-TwoPathSimple-v0",
  entry_point="gym_minigrid.envs:TwoPathEnvSimple",
  kwargs={"max_steps": MAX_EPISODE_STEPS}
)
