from gym_minigrid import minigrid
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

DEF_MAX_EPISODE_LENGTH = 50

class UnsafeCrossingEnv(MiniGridEnv):
  """
  Environment with gaps in walls to pass through to reach a goal whilst avoiding a random object.
  """

  gap_objs = {"floor": Floor(), "door": Door(color="green", is_open=True), "water": Water(), "glass": Glass()}

  # Override set of possible actions
  class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

  def __init__(
    self,
    num_crossings,
    obstacle_types,
    grid_size=None,
    width=None,
    height=None,
    seed=None,
    agent_view_size=7,
    random_crossing=True,
    no_safe_obstacle=False,
    **kwargs
  ):
    self.num_crossings = num_crossings
    self.obstacle_types = obstacle_types
    self.obstacle_objs = {}
    for t in obstacle_types:
      if t == "lava":
        self.obstacle_objs["lava"] = Lava()
      else:
        self.obstacle_objs[t] = self.gap_objs[t]
      

    self.safe_gap_types = [obj_type for obj_type in self.gap_objs.keys() if not (obj_type in self.obstacle_types)]
    self.random_crossing = random_crossing
    self.no_safe_obstacle = no_safe_obstacle
    super().__init__(grid_size=grid_size, width=width, height=height, seed=seed, agent_view_size=agent_view_size, **kwargs)
    self.actions = UnsafeCrossingEnv.Actions

  def _gen_grid(self, width, height):
    assert self.num_crossings <= math.ceil((width - 4) / 2)
    assert height >= 5
    assert width % 2 == 1 and height % 2 == 1

    # Pick an obstacle type for this grid
    self.obstacle_type = self._rand_elem(self.obstacle_types)

    self.grid = Grid(width, height)

    # Generate outside walls
    self.grid.wall_rect(0, 0, width, height)

    # Place agent at start in centre
    self.agent_pos = (1, height // 2)
    self.agent_dir = 0 # Right

    # Place goal square on opposite side
    self.put_obj(Goal(), width - 2, height // 2)

    for n in range(self.num_crossings):
      if self.random_crossing:
        gaps = self._rand_subset(range(1, height - 1), 2)
      else:
        if (height < 7):
          spacing = 0
        else:
          spacing = (math.ceil((height - 2) / 2)) // 2
        gaps = [spacing + 1, height - 1 - (spacing + 1)]
      i = 2 + (n * 2)
      for j in range(1, height - 1):
        if not (j in gaps):
          self.put_obj(Wall(), i, j)
        
      safe_gap_idx = int(self._rand_bool())
      # safe_gap_idx = 0

      if not self.no_safe_obstacle:
        self.put_obj(self.gap_objs[self._rand_elem(self.safe_gap_types)], i, gaps[abs(0 - safe_gap_idx)])
      self.put_obj(self.obstacle_objs[self.obstacle_type], i, gaps[abs(1 - safe_gap_idx)])

    self.mission = (
      f"avoid the {self.obstacle_type} and get to the green goal square"
    )
  
  def _reward(self, done=True, violation=False):
    """Override default reward function to penalise on each non-successful step."""
    if done:
      if violation:
        return 0
        # return -0.5 / self.max_steps
        # return - (0.5 * self.steps_remaining) / self.max_steps
      else:
        return 1
    else:
      return 0
      # return -0.5 / self.max_steps

  def gen_obs(self):
    obs = super().gen_obs()
    one_hot_image = self._to_one_hot_obs(obs["image"])
    obs["one_hot"] = one_hot_image
    return obs

  def _to_one_hot_obs(self, obs):
    """
    Take an observation in minigrid format of (H x W x 3) for the 3 WorldObj fields, and convert
    into a one-hot encoding for a reduced set of objects with size (H x W x N) where N is the
    number of distinct objects in the environment.
    """

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

class UnsafeCrossingSimpleEnv(UnsafeCrossingEnv):
  def __init__(self, **kwargs):
    super().__init__(num_crossings=1, obstacle_types=["lava"], grid_size=5, random_crossing=False, agent_view_size=5, no_safe_obstacle=True, **kwargs)

class UnsafeCrossingMicroEnv(UnsafeCrossingEnv):
  def __init__(self, **kwargs):
    super().__init__(num_crossings=1, obstacle_types=["lava"], grid_size=5, random_crossing=False, agent_view_size=5, **kwargs)

class UnsafeCrossingSmallEnv(UnsafeCrossingEnv):
  def __init__(self, **kwargs):
    super().__init__(num_crossings=1, obstacle_types=["lava", "glass"], width=5, height=7, **kwargs)
  
class UnsafeCrossingMedEnv(UnsafeCrossingEnv):
  def __init__(self, **kwargs):
    super().__init__(num_crossings=2, obstacle_types=["lava", "glass"], grid_size=9, **kwargs)

register(
  id="MiniGrid-UnsafeCrossingSimple-v0",
  entry_point="gym_minigrid.envs:UnsafeCrossingSimpleEnv",
  kwargs={"max_steps": DEF_MAX_EPISODE_LENGTH}
)

register(
  id="MiniGrid-UnsafeCrossingMicro-v0",
  entry_point="gym_minigrid.envs:UnsafeCrossingMicroEnv",
  kwargs={"max_steps": DEF_MAX_EPISODE_LENGTH}
)

register(
  id="MiniGrid-UnsafeCrossingN1-v0",
  entry_point="gym_minigrid.envs:UnsafeCrossingSmallEnv",
  kwargs={"max_steps": DEF_MAX_EPISODE_LENGTH}
)

register(
  id="MiniGrid-UnsafeCrossingN2-v0",
  entry_point="gym_minigrid.envs:UnsafeCrossingMedEnv",
  kwargs={"max_steps": DEF_MAX_EPISODE_LENGTH}
)
