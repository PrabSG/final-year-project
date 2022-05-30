import numpy as np
from rlpyt.agents.base import AgentStep, BaseAgent, RecurrentAgentMixin
from rlpyt.utils.buffer import buffer_func, buffer_to
from rlpyt.utils.collections import namedarraytuple
import torch

from agents.adapt.models import AgentModel


EXPLORE_TYPES = ["epsilon_greedy", "gauss_noise"]

AdaptDreamerAgentInfo = namedarraytuple("AdaptDreamerAgentInfo", ["prev_state"])

class AdaptDreamerAgent(RecurrentAgentMixin, BaseAgent):
  def __init__(self, ModelCls=AgentModel, train_noise=0.4, train_noise_min=0.1, eval_noise=0, explore_decay=200000, explore_type="epsilon_greedy", model_kwargs=None, initial_model_state_dict=None):
    assert explore_type in EXPLORE_TYPES

    self.train_noise = train_noise
    self.train_noise_min = train_noise_min
    self.eval_noise = eval_noise
    self.explore_decay = explore_decay
    self.explore_type = explore_type
    
    super().__init__(ModelCls, model_kwargs, initial_model_state_dict)

    self._mode = 'train'
    self._itr = 0

  def make_env_to_model_kwargs(self, env_spaces):
    """Generate any keyword args to the model which depend on environment interfaces."""
    return dict(
      action_shape=env_spaces.action.shape,
      observation_shape=env_spaces.observation.shape
    )

  def __call__(self, observation, prev_action, init_rnn_state):
    model_inputs = buffer_to((observation, prev_action, init_rnn_state), device=self.device)
    return self.model(*model_inputs)

  @torch.no_grad()
  def step(self, observation: torch.Tensor, prev_action: torch.Tensor, prev_reward):
    """
    Return policy's action by sampling from calculated distribution. Takes an observation,
    recurrent state, action as inputs to advance recurrent state, calculate value, reward and
    violation predictions, as well as calculating policy's action.
    """
    # Expand dims if do not already have a time batch
    if len(observation.shape) == len(self.env_model_kwargs["observation_shape"]):
      observation = torch.unsqueeze(observation, dim=0)
      prev_reward = torch.unsqueeze(prev_reward, dim=0)
    if len(prev_action.shape) == len(self.env_model_kwargs["action_shape"]):
      prev_action = torch.unsqueeze(prev_action, dim=0)

    model_inputs = buffer_to((observation, prev_action), device=self.device)
    if self.prev_rnn_state is not None:
      prev_state = buffer_func(self.prev_rnn_state, torch.squeeze)
    else:
      prev_state = None
    action, _, _, _, new_state = self.model(*model_inputs, prev_state)
    action = self.exploration_noise(action).squeeze()
    # Initialise states to zeroes if None
    prev_state = self.prev_rnn_state or buffer_func(new_state, torch.zeros_like)
    
    self.advance_rnn_state(
      buffer_func(new_state, torch.unsqueeze, dim=0) if
      len(new_state.stoch.shape) == 2 else
      new_state
    )
    agent_info = AdaptDreamerAgentInfo(prev_state=buffer_func(prev_state, torch.squeeze))
    agent_step = AgentStep(action=action, agent_info=agent_info)

    # Return all tensors to cpu for interaction with sampler and environments
    return buffer_to(agent_step, device="cpu")

  def exploration_noise(self, action: torch.Tensor) -> torch.Tensor:
    if self._mode in ["train", "sample"]:
      eps = self.train_noise
      if self.explore_decay: # Linear Decay to minimum train noise
        eps += ((self.train_noise - self.train_noise_min) * self._itr / self.explore_decay)
        eps = max(eps, self.train_noise_min)
    elif self._mode == "eval":
      eps = self.eval_noise
    else:
      raise NotImplementedError

    if self.explore_type == "epsilon_greedy":
      if np.random.uniform(0, 1) < eps:
        num_actions = self.env_model_kwargs["action_shape"][0]
        index = torch.randint(0, num_actions, action.shape[:-1], device=action.device)
        action = torch.zeros_like(action)
        action[..., index] = 1
    elif self.explore_type == "gauss_noise":
      noise = torch.randn(*action.shape, device=action.device)
      action = torch.clamp(action + noise, -1, 1)
    return action
    
      
  