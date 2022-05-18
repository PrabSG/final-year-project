import torch.nn as nn

from agents.adapt.models import Encoder

class AgentModel(nn.Module):
  def __init__(
    self,
    action_shape,
    stochastic_size=30,
    deterministic_size=200,
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
    self.encoder = Encoder()
    self.encoder_embed_size = self.encoder.embed_size