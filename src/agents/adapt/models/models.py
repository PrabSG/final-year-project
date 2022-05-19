import numpy as np
import torch
import torch.nn as nn

from agents.adapt.models import EncoderModel

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
    self.action_size = np.prod(action_shape)
    self.action_shape = action_shape

    self.encoder = EncoderModel(input_shape)
    encoder_embed_size = self.encoder.embed_size
    self.decoder = DecoderModel(rssm_state_size, input_shape)
    self.transition_model = TransitionModel(self.action_size, stochastic_size, deterministic_size, transition_hidden)
    self.representation_model = RepresentationModel(self.transition_model, encoder_embed_size, self.action_size, stochastic_size, deterministic_size, transition_hidden)
    self.action_model = ActionModel(rssm_state_size, self.action_size, action_hidden, action_layers, action_dist)
    self.reward_model = RewardModel(rssm_state_size, reward_shape, reward_hidden, reward_layers)
    self.value_model = ValueModel(rssm_state_size, value_shape, value_hidden, value_layers)

    self.dtype = dtype
    self.stochastic_size = stochastic_size
    self.deterministic_size = deterministic_size

