from collections import namedtuple
from turtle import forward
from typing import List

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.rlpyt.utils.buffer import buffer_method

RSSMState = namedtuple("RSSMState", ["mean", "std", "stoch", "det"])

def stack_states(rssm_states: List[RSSMState], dim: int):
  return RSSMState(
    torch.stack([state.mean for state in rssm_states], dim=dim),
    torch.stack([state.std for state in rssm_states], dim=dim),
    torch.stack([state.stoch for state in rssm_states], dim=dim),
    torch.stack([state.det for state in rssm_states], dim=dim)
  )

class TransitionModel(nn.Module):
  """
  Model to take previous state and action to compute prior state for current state.
  (h_{t-1}, a_{t-1}) -> (z^_t, h_t)
  """

  def __init__(
    self,
    action_size,
    stochastic_size,
    deterministic_size,
    hidden_size,
    min_std=0.1,
    activation=nn.ELU,
    distribution=td.Normal
  ):
    super().__init__()
    self._action_size = action_size
    self._stoch_size = stochastic_size
    self._det_size = deterministic_size
    self._hidden_size = hidden_size
    self._min_std = min_std
    self._activation = activation
    self._dist = distribution
    self._cell = nn.GRUCell(hidden_size, deterministic_size)
    self._rnn_input_model = self._build_rnn_input_model()
    self._stochastic_prior_model = self._build_stochastic_model()

  def _build_rnn_input_model(self):
    return nn.Sequential(
      nn.Linear(self._action_size + self._stoch_size, self._hidden_size),
      self._activation()
    )

  def _build_stochastic_model(self):
    """
    Build model to calculate stochastic prior distribution from deterministic state.
    """
    return nn.Sequential(
      nn.Linear(self._det_size, self._hidden_size),
      self._activation(),
      nn.Linear(self._hidden_size, 2 * self._stoch_size)
    )
  
  def initial_state(self, batch_size, **kwargs):
    return RSSMState(
      torch.zeros((batch_size, self._stoch_size), **kwargs),
      torch.zeros((batch_size, self._stoch_size), **kwargs),
      torch.zeros((batch_size, self._stoch_size), **kwargs),
      torch.zeros((batch_size, self._det_size), **kwargs)
    )

  def forward(self, prev_action: torch.Tensor, prev_state: RSSMState):
    rnn_input = self._rnn_input_model(torch.cat([prev_action, prev_state.stoch], dim=-1))
    det_state = self._cell(rnn_input, prev_state.det)
    mean, std = torch.chunk(self._stochastic_prior_model(det_state), 2, dim=-1)
    std = F.softplus(std) + self._min_std
    dist = self._dist(mean, std)
    stoch_state = dist.rsample()
    return RSSMState(mean, std, stoch_state, det_state)

class RepresentationModel(nn.Module):
  """
  Model to take previous state and action to compute prior state for current state, as well as also using
  current encoded observation to further compute current posterior state.
  (h_{t-1}, a_{t-1}, o_t) -> (z^_t, h_t), (z_t, h_t)
  """

  def __init__(
    self,
    transition_model,
    encoder_embed_size,
    action_size,
    stochastic_size,
    deterministic_size,
    hidden_size,
    min_std=0.1,
    activation=nn.ELU,
    distribution=td.Normal
  ):
    super().__init__()
    self._transition_model = transition_model
    self._encoder_embed_size = encoder_embed_size
    self._action_size = action_size
    self._stoch_size = stochastic_size
    self._det_size = deterministic_size
    self._hidden_size = hidden_size
    self._min_std = min_std
    self._activation = activation
    self._dist = distribution
    self._stochastic_posterior_model = self._build_stochastic_model()

  def _build_stochastic_model(self):
    """
    Build model to calculate stochastic posterior distribution from deterministic state and
    encoded observation.
    """
    return nn.Sequential(
      nn.Linear(self._det_size + self._encoder_embed_size, self._hidden_size),
      self._activation(),
      nn.Linear(self._hidden_size, 2 * self._stoch_size) # One output for mean, one for std
    )
  
  def initial_state(self, batch_size, **kwargs):
    return RSSMState(
      torch.zeros((batch_size, self._stoch_size), **kwargs),
      torch.zeros((batch_size, self._stoch_size), **kwargs),
      torch.zeros((batch_size, self._stoch_size), **kwargs),
      torch.zeros((batch_size, self._det_size), **kwargs)
    )
  
  def forward(self, encoded_obs: torch.Tensor, prev_action: torch.Tensor, prev_state: RSSMState):
    prior_state = self._transition_model(prev_action, prev_state) # (a_{t-1}, z_{t-1}, h_{t-1}) -> (z^_t, h_t)
    x = torch.cat([prior_state.deter, encoded_obs], dim=-1) # (h_t, o_t)
    mean, std = torch.chunk(self._stochastic_posterior_model(x), 2, dim=-1)
    std = F.softplus(std) + self._min_std
    dist = self._dist(mean, std)
    stoch_state = dist.rsample()
    posterior_state = RSSMState(mean, std, stoch_state, prior_state.det)
    return prior_state, posterior_state


class RSSMRollout(nn.Module):
  def __init__(
    self,
    representation_model: RepresentationModel,
    transition_model: TransitionModel
  ) -> None:
    super().__init__()
    self.representation_model = representation_model
    self.transition_model = transition_model

  def forward(self, steps, encoded, actions, prev_state):
    return self.rollout_representation(steps, encoded, actions, prev_state)
    

  def rollout_representation(self, steps: int, encoded: torch.Tensor, actions: torch.Tensor, state: RSSMState):
    """
    Rollout out model with given actions and observations.
    :param steps: number of steps to roll out, must be <= batch_t
    :param encoded: size(batch_t, batch_b, encoder_embed_size)
    :param actions: size(batch_t, batch_b, action_size)
    :param state: initial RSSMState size(batch_b, state_size)
    :return prior, posterior states: size(steps, batch_b, state_size)
    """
    assert steps <= actions.shape[0]
    priors = []
    posteriors = []

    for t in range(steps):
      prior_state, posterior_state = self.representation_model(encoded[t], actions[t], state)
      priors.append(prior_state)
      posteriors.append(posterior_state)
      state = posterior_state
    
    priors = stack_states(priors, dim=0)
    posteriors = stack_states(posteriors, dim=0)
    return priors, posteriors
  
  def rollout_transition(self, steps, actions: torch.Tensor, state: RSSMState):
    """
    Rollout model with given actions, using imagined transitions.
    :param steps: number of steps to roll out, must be <= batch_t
    :param actions: size(batch_t, batch_b, action_size)
    :param state: initial RSSMState size(batch_b, state_size)
    :return prior states: size(steps, batch_b, state_size)
    """
    assert steps <= actions.shape[0]
    priors = []

    for t in range(steps):
      state = self.transition_model(actions[t], state)
      priors.append(state)

    return stack_states(priors, dim=0)
  
  def rollout_policy(self, steps: int, policy: nn.Module, state: RSSMState):
    """
    Rollout model using given policy to choose actions.
    :param steps: number of steps to roll out
    :param policy: RSSMState size(batch_b, state_size) -> size(batch_b, action_size)
    :param state: initial RSSMState size(batch_b, state_size)
    :return next states: size(steps, batch_b, state_size), actions size(steps, batch_b, action_size)
    """
    next_states = []
    actions = []
    state = buffer_method(state, "detach")

    for t in range(steps):
      action, _ = policy(buffer_method(state, "detach"))
      state = self.transition_model(action, state)
      next_states.append(state)
      actions.append(action)

    next_states = stack_states(next_states, dim=0)
    actions = torch.stack(actions, dim=0)
    return next_states, actions