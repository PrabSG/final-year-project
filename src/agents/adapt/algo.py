from curses import has_colors
import itertools

from numpy import dtype
from agents.adapt.replay import initialize_replay_buffer, samples_to_buffer

from rlpyt.rlpyt.algos.base import RlAlgorithm
from rlpyt.rlpyt.replays.sequence.n_step import SamplesFromReplay
from rlpyt.rlpyt.utils.collections import namedarraytuple
from rlpyt.rlpyt.utils.quick_args import save__init__args
from rlpyt.rlpyt.utils.tensor import infer_leading_dims
import torch

loss_info_fields = ["model_loss", "actor_loss", "value_loss", "prior_entropy", "post_entropy", "divergence",
                    "reward_loss", "image_loss"]

LossInfo = namedarraytuple("LossInfo", loss_info_fields)
OptInfo = namedarraytuple("OptInfo",
                          ["loss", "grad_clip_model", "grad_clip_actor", "grad_clip_value"] + loss_info_fields)

class Dreamer(RlAlgorithm):
  def __init__(
    self,
    batch_size=50,
    batch_length=50,
    train_every=1000,
    train_steps=100,
    pretrain=100,
    model_lr=6e-4,
    value_lr=8e-5,
    actor_lr=8e-5,
    grad_clip=100.0,
    dataset_balance=False,
    discount=0.99,
    discount_lambda=0.95,
    horizon=15,
    action_dist="one_hot",
    action_init_std=5.0,
    expl="epsilon_greedy",
    expl_amount=0.4,
    expl_decay=200000,
    expl_min=0.1,
    OptimCls=torch.optim.Adam,
    optim_kwargs=None,
    initial_optim_state_dict=None,
    replay_sizr=int(5e6),
    replay_ratio=8,
    n_step_return=1,
    updates_per_sync=1,
    free_nats=3,
    kl_scale=0.1,
    type=torch.float,
    prefill=5000,
    log_video=True,
    video_every=10,
    video_summary_t=25,
    video_summary_b=4
  ) -> None:
    super().__init__()

    if optim_kwargs is None:
      optim_kwargs = {}
    self._batch_size = batch_size
    del batch_size
    save__init__args(locals())
    self.update_counter = 0
    self.optimizer = None
    self.type = type

  def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0):
    self.agent = agent
    self.n_itr = n_itr
    self.batch_spec = batch_spec
    self.mid_batch_reset = mid_batch_reset
    self.replay_buffer = initialize_replay_buffer(self, examples, batch_spec)
    self.optim_initialize(rank)
  
  def optim_initialize(self, rank=0):
    self.rank = rank
    model = self.agent.model
    self.model_modules = [
      model.encoder,
      model.decoder,
      model.reward_model,
      model.representation_model,
      model.transition_model
    ]

    self.actor_modules = [model.action_model]
    self.value_modules = [model.value_model]
    self.model_optimizer = torch.optim.Adam(itertools.chain(*self.model_modules), lr=self.model_lr,
                                            **self.optim_kwargs)
    self.actor_optimizer = torch.optim.Adam(itertools.chain(*self.actor_modules), lr=self.actor_lr,
                                            **self.optim_kwargs)
    self.value_optimizer = torch.optim.Adam(itertools.chain(*self.value_modules), lr=self.value_lr,
                                            **self.optim_kwargs)

    if self.initial_optim_state_dict is not None:
      self.load_optim_state_dict(self.initial_optim_state_dict)

    self.opt_info_fields = OptInfo._fields
                                            
  def optim_state_dict(self):
    return dict(
      model_optimizer_dict=self.model_optimizer.state_dict(),
      actor_optimizer_dict=self.actor_optimizer.state_dict(),
      value_optimizer_dict=self.value_optimizer.state_dict()
    )
  
  def load_optim_state_dict(self, state_dict):
    self.model_optimizer.load_state_dict(state_dict["model_optimizer_dict"])
    self.actor_optimizer.load_state_dict(state_dict["actor_optimizer_dict"])
    self.value_optimizer.load_state_dict(state_dict["value_optimizer_dict"])

  def optimize_agent(self, itr, samples=None, sampler_itr=None):
    itr = itr if sampler_itr is None else sampler_itr
    if samples is not None:
      self.replay_buffer.append(samples(samples_to_buffer(samples)))
    
    opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
    if itr < self.prefill:
      return opt_info
    if itr % self.train_every != 0:
      return opt_info
    
    for i in tqdm(range(self.train_steps), desc="Optimisation via Imagination"):
      replay_samples = self.replay_buffer.sample_batch(self.batch_size, self.batch_length)
      buffed_samples = buffer_to(replay_samples, self.agent.device)
      model_loss, actor_loss, value_loss, loss_info = self.loss(buffed_samples, itr, i)

      # Backpropagation of gradients
      self.model_optimizer.zero_grad()
      self.actor_optimizer.zero_grad()
      self.value_optimizer.zero_grad()

      model_loss.backward()
      actor_loss.backward()
      value_loss.backward()

      grad_clip_model = torch.nn.utils.clip_grad_norm_(itertools.chain(*self.model_modules), self.grad_clip)
      grad_clip_actor = torch.nn.utils.clip_grad_norm_(itertools.chain(*self.actor_modules), self.grad_clip)
      grad_clip_value = torch.nn.utils.clip_grad_norm_(itertools.chain(*self.value_modules), self.grad_clip)

      self.model_optimizer.step()
      self.actor_optimizer.step()
      self.value_optimizer.step()

      # Add info for logging
      with torch.no_grad:
        loss = model_loss + actor_loss + value_loss
      opt_info.loss.append(loss.item())

      if torch.is_tensor(grad_clip_model):
        opt_info.grad_clip_model.append(grad_clip_model.item())
        opt_info.grad_clip_actor.append(grad_clip_actor.item())
        opt_info.grad_clip_value.append(grad_clip_value.item())
      else:
        opt_info.grad_clip_model.append(grad_clip_model)
        opt_info.grad_clip_actor.append(grad_clip_actor)
        opt_info.grad_clip_value.append(grad_clip_value)
      for field in loss_info_fields:
        if hasattr(opt_info, field):
          getattr(opt_info, field).append(getattr(loss_info, field).item())
      
    return opt_info
  
  def loss(self, samples: SamplesFromReplay, sample_itr: int, opt_itr: int):
    model = self.agent.model

    observations = samples.all_observation[:-1] # [t, t+batch_length+1] -> [t, t+batch_length]
    actions = samples.all_action[1:] # [t-1, t+batch_length] -> [t, t+batch_length]
    rewards = samples.all_reward[1:] # [t-1, t+batch_length] -> [t, t+batch_length]
    rewards = rewards.unsqueeze(2) # Add additional dimension after time and batch
    done = samples.done
    done = done.unsqueeze(2) # Add additional dimension after time and batch

    # Samples drawn from replay are in size (batch_length, batch_size, *img_shape)
    lead_dim, batch_t, batch_b, img_shape = infer_leading_dims(observations, 3)
    batch_size = batch_t * batch_b

    encoded = model.encoder(observations)
    prev_state = model.representation_model.initial_state(batch_b, device=actions.device, dtype=actions.dtype)


      