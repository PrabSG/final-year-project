import re
from typing import Dict
from gym_minigrid import minigrid
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from safety.condition import AvoidRequirement, SafetyRequirement, UntilRequirement

DEF_MAX_EPISODE_LENGTH = 50

# Reward Values
GOAL_REWARD = 1
NO_MOVE_REWARD = - 0.1
DEFAULT_REWARD = -0.01

SAFETY_PROPS_TO_SYMBOLS = {
  "touching_lava": "a",
  "touching_water": "b",
  "touching_glass": "c",
  "standing_still": "d",
  "reach_goal": "e"
}
STANDING_STILL_LIMIT = 3

class UnsafeCrossingEnv(MiniGridEnv):
  """
  Environment with gaps in walls to pass through to reach a goal whilst avoiding a random object.
  """

  gap_objs = {"lava": Lava(), "floor": Floor(), "water": Water(), "glass": Glass()}
  
  # Specifiable Safety requirements for UnsafeCrossingEnv
  touch_water = SafetyRequirement(SAFETY_PROPS_TO_SYMBOLS["touching_water"], req_objs=["water"])
  avoid_lava = AvoidRequirement(SAFETY_PROPS_TO_SYMBOLS["touching_lava"], avoid_objs=["lava"])
  avoid_glass = AvoidRequirement(SAFETY_PROPS_TO_SYMBOLS["touching_glass"], avoid_objs=["glass"])
  avoid_water = AvoidRequirement(SAFETY_PROPS_TO_SYMBOLS["touching_water"], avoid_objs=["water"])
  avoid_standing_still = AvoidRequirement(SAFETY_PROPS_TO_SYMBOLS["standing_still"])
  avoid_standing_glass = AvoidRequirement(("and", SAFETY_PROPS_TO_SYMBOLS["touching_glass"], SAFETY_PROPS_TO_SYMBOLS["standing_still"]), ["glass"])
  avoid_specs = [avoid_lava, avoid_glass, avoid_water, avoid_standing_still, avoid_standing_glass]

  # Example safety specs
  no_glass_until_water = UntilRequirement(avoid_glass, touch_water)

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
    safety_spec="random",
    **kwargs
  ):
    self.num_crossings = num_crossings
    self.obstacle_type = None
    self.obstacle_types = obstacle_types
    self.obstacle_objs = {}
    for t in obstacle_types:
      if t == "lava":
        self.obstacle_objs["lava"] = Lava()
      else:
        self.obstacle_objs[t] = self.gap_objs[t]

    self.safe_gap_types = set([obj_type for obj_type in self.gap_objs.keys() if not (obj_type in self.obstacle_types)])
    self.random_crossing = random_crossing
    self.no_safe_obstacle = no_safe_obstacle
    
    # Environment propositions for building safety specifications
    self._safety_props = {
      "touching_lava": False, # The agent makes contact with a lava tile
      "touching_water": False, # The agent makes contact with a water tile
      "touching_glass": False, # The agent makes contact with a glass tile
      "standing_still": False, # The agent's position does not change for 3 consecutive actions
      "reach_goal": False # The agent reaches the goal position
    }
    self._no_move_count = 0
    self.randomized_safety_spec = safety_spec == "random"

    # Have to set seed here to use random functions
    self.seed(seed=seed)

    # Safety specification for environment
    if self.randomized_safety_spec:
      self._safety_spec, self._avoid_objs, self._req_objs = self._gen_random_safety_spec()
    else:
      self._safety_spec, self._avoid_objs, self._req_objs = safety_spec, set(self.obstacle_types), set()

    super().__init__(grid_size=grid_size, width=width, height=height, seed=seed, agent_view_size=agent_view_size, **kwargs)    

    self.actions = UnsafeCrossingEnv.Actions

  def _gen_grid(self, width, height):
    assert self.num_crossings <= math.ceil((width - 4) / 2)
    assert height >= 5
    assert width % 2 == 1 and height % 2 == 1

    # Pick an obstacle type for this grid
    obstacle_types = self._avoid_objs

    # Determine gap objects
    gap_types = self._req_objs if len(self._req_objs) > 0 else self.safe_gap_types

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

      if not self.no_safe_obstacle:
        self.put_obj(self.gap_objs[self._rand_elem(gap_types)], i, gaps[abs(0 - safe_gap_idx)])
      self.put_obj(self.gap_objs[self._rand_elem(obstacle_types)], i, gaps[abs(1 - safe_gap_idx)])

    self.mission = (
      f"avoid the lava and get to the green goal square. Safety Constraint: {self._safety_spec}"
    )
  
  def _reward(self, done=True):
    """Override reward function to penalise on each non-successful step."""
    if done:
      return GOAL_REWARD
    elif self.no_change:
      return NO_MOVE_REWARD
    else:
      return DEFAULT_REWARD

  def step(self, action):
    """Override step function to set environment propositions."""
    obs, reward, done, info = super().step(action)
    props = self._set_prop_values(reward, done)
    info["true_props"] = self._extract_true_props(props)
    return obs, reward, done, info

  def reset(self):
    """Override reset function to also handle randomly setting safety specficiations."""
    if self.randomized_safety_spec:
      self._safety_spec, self._avoid_objs, self._req_objs = self._gen_random_safety_spec()
    return super().reset()

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

  def _gen_random_safety_spec(self):
    # TODO(PrabSG@): Change to use curriculum learning to cycle specifications
    formula = ""
    to_avoid = set(["lava"])
    required = set()
    spec_configs = ["one_avoid", "two_avoid", "cond_avoid"]
    config = self._rand_elem(spec_configs)
    if config == "one_avoid":
      avoid = self._rand_elem(self.avoid_specs)
      formula = ("until", avoid.get_formula(), SAFETY_PROPS_TO_SYMBOLS["reach_goal"])
      to_avoid = to_avoid.union(avoid.get_avoid_objs())
      required = required.union(avoid.get_req_objs())
    if config == "two_avoid":
      avoid1, avoid2 = self._rand_subset(self.avoid_specs, 2)
      formula = ("until", ("and", avoid1.get_formula(), avoid2.get_formula()), SAFETY_PROPS_TO_SYMBOLS["reach_goal"])
      to_avoid = to_avoid.union(avoid1.get_avoid_objs().union(avoid2.get_avoid_objs()))
      required = required.union(avoid1.get_req_objs().union(avoid2.get_req_objs()))
    elif config == "cond_avoid":
      avoid = self.no_glass_until_water
      formula = avoid.get_formula()
      to_avoid = avoid.get_avoid_objs()
      required = avoid.get_req_objs()

    return formula, to_avoid, required

  def get_safety_spec(self):
    return self._safety_spec
  
  def _set_prop_values(self, reward, done):
    # Check touching_lava
    # Entered non-goal terminating state, i.e. touched lava/obstacle
    self._safety_props["touching_lava"] = done and reward != GOAL_REWARD
    self._safety_props["reach_goal"] = done and reward == GOAL_REWARD

    curr_grid_cell = self.grid.get(*self.agent_pos)
    if curr_grid_cell != None:
      # Check touching water
      self._safety_props["touching_water"] = curr_grid_cell.type == "water"

      # Check touching glass
      self._safety_props["touching_glass"] = curr_grid_cell.type == "glass"
    else:
      self._safety_props["touching_water"] = False
      self._safety_props["touching_glass"] = False

    # Check standing still
    self._no_move_count = self._no_move_count + 1 if self.no_change else 0
    self._safety_props["standing_still"] = self._no_move_count >= STANDING_STILL_LIMIT
    
    return self.safety_props

  @property
  def safety_props(self):
    return self._safety_props

  def _extract_true_props(self, props: Dict[str, bool]):
    true_props = set()
    for prop, val in props.items():
      if val:
        true_props.add(SAFETY_PROPS_TO_SYMBOLS[prop])
    return true_props


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
