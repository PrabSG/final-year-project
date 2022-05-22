from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from agents.adapt.models import EncoderModel
from agents.adapt.models.decoder import DecoderModel
from agents.adapt.models.rssm import RSSMState

class AgentModel(nn.Module):
  def __init__(
    self,
    action_shape,
    stochastic_size=30,
    deterministic_size=200,
    transition_hidden=200,
    input_shape=(3, 5, 5),
    action_hidden=200,
    action_layers=3,
    action_dist="one_hot",
    reward_shape=(1,),
    reward_hidden=300,
    reward_layers=3,
    value_shape=(1,),
    value_hidden=200,
    value_layers=3,
    dtype=torch.float,
    use_pcont=False,
    pcont_layers=3,
    pcont_hidden=200,
    **kwargs
  ) -> None:
    super().__init__()
    rssm_state_size = stochastic_size + deterministic_size
    self.action_dist = action_dist
    self.action_size = np.prod(action_shape)
    self.action_shape = action_shape

    self.encoder = EncoderModel(input_shape)
    encoder_embed_size = self.encoder.embed_size
    self.decoder = DecoderModel(encoder_embed_size, input_shape)
    self.transition_model = TransitionModel(self.action_size, stochastic_size, deterministic_size, transition_hidden)
    self.representation_model = RepresentationModel(self.transition_model, encoder_embed_size, self.action_size, stochastic_size, deterministic_size, transition_hidden)
    self.action_model = ActionModel(rssm_state_size, self.action_size, action_hidden, action_layers, action_dist)
    self.reward_model = RewardModel(rssm_state_size, reward_shape, reward_hidden, reward_layers)
    self.value_model = ValueModel(rssm_state_size, value_shape, value_hidden, value_layers)

    self.dtype = dtype
    self.stochastic_size = stochastic_size
    self.deterministic_size = deterministic_size

  def forward(self, observation: torch.Tensor, prev_action: Optional[torch.Tensor] = None, prev_state: Optional[RSSMState] = None):
    state = self.get_state_representation(observation, prev_action, prev_state)
    action, action_dist = self.policy(state)
    value = self.value_model(torch.cat((state.stoch, state.det), dim=-1))
    reward = self.reward_model(torch.cat((state.stoch, state.det), dim=-1))
    return action, action_dist, value, reward, state

  def get_state_representation(
    self,
    observation: torch.Tensor,
    prev_action: Optional[torch.Tensor] = None,
    prev_state: Optional[RSSMState] = None
  ) -> RSSMState:
    """
    Takes an observation o_t, and the previous RSSM state composed of the determinstic state
    h_{t-1} and either the prior or posterior stochastic state z_{t-1}, and returns the currest
    RSSM state of determinstic state h_t and the posterior stochastic state z_t.
    """
    encoded_obs = self.encoder(observation)
    
    if prev_action is None:
      prev_action = torch.zeros((observation.shape[0], self.action_size), device=observation.device, dtype=observation.dtype)
    
    if prev_state is None:
      prev_state = self.representation_model.initial_state(observation.shape[0], device=observation.device, dtype=observation.dtype)

    _, state = self.representation_model(encoded_obs, prev_action, prev_state)

    return state

  def get_state_transition(
    self,
    prev_action: torch.Tensor,
    prev_state: torch.Tensor
  ) -> RSSMState:
    """Advances RSSM state using previous state and given action using world model to predict
    transition dynamics. Generates next determinstic state and prior stochastic state."""
    state = self.transition_model(prev_action, prev_state)
    return state


  def policy(self, state: RSSMState):
    features = torch.cat((state.stoch, state.det), dim=-1)
    action_dist = self.action_model(features)

    if self.action_dist == "one_hot":
      action = action_dist.sample() # Sampling carries no gradient
      action = action + action_dist.prob - action_dist.probs.detach() # Enables straight-through grads with auto-grad
    if self.action_dist == "tanh_normal":
      if self.training:
        action = action_dist.rsample()
      else:
        action = action_dist.rsample()
    return action, action_dist







