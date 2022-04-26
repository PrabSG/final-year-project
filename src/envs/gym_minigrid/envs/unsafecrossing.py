from envs.gym_minigrid.minigrid import *
from envs.gym_minigrid.register import register


class UnsafeCrossingEnv(MiniGridEnv):
  """
  Environment with gaps in walls to pass through to reach a goal whilst avoiding a random object.
  """

  gap_objs = {"floor": Floor(), "door": Door(is_open=True), "water": Water(), "glass": Glass()}

  def __init__(
    self,
    num_crossings,
    obstacle_types,
    grid_size=None,
    width=None,
    height=None,
    seed=None,
    agent_view_size=7
  ):
    self.num_crossings = num_crossings
    self.obstacle_types = obstacle_types
    self.obstacle_objs = {}
    for t in obstacle_types:
      if t == "lava":
        self.obstacle_objs["lava"] = Lava()
      else:
        self.obstacle_objs[t] = self.gap_objs[t]
      
    self.safe_gap_types = [obj_type for obj_type in self.gap_objs.keys() if obj_type != self.obstacle_type]
    super().__init__(grid_size=grid_size, width=width, height=height, seed=seed, agent_view_size=agent_view_size)

  def _gen_grid(self, width, height):
    assert self.num_crossings <= math.ceil((width - 4) / 2)
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
      gaps = self._rand_subset(range(1, height - 1), 2)
      i = 2 + (n * 2)
      for j in range(1, height - 1):
        if not (j in gaps):
          self.put_obj(Wall(), i, j)
        
      safe_gap_idx = int(self._rand_bool())

      self.put_obj(self.gap_objs[self._rand_elem(self.safe_gap_types)], i, gaps[abs(0 - safe_gap_idx)])
      self.put_obj(self.obstacle_objs[self.obstacle_type], i, gaps[abs(1 - safe_gap_idx)])

    self.mission = (
      f"avoid the {self.obstacle_type} and get to the green goal square"
    )
  
class UnsafeCrossingSmallEnv(UnsafeCrossingEnv):
  def __init__(self):
    super().__init__(num_crossings=1, obstacle_types=["lava", "glass"], width=5, height=7)
  
class UnsafeCrossingMedEnv(UnsafeCrossingEnv):
  def __init__(self):
    super().__init__(num_crossings=2, obstacle_types=["lava", "glass"], grid_size=9)

register(
  id="MiniGrid-UnsafeCrossingN1-v0",
  entry_point="gym_minigrid.envs:UnsafeCrossingSmallEnv"
)

register(
  id="MiniGrid-UnsafeCrossingN2-v0",
  entry_point="gym_minigrid.envs:UnsafeCrossingMedEnv"
)