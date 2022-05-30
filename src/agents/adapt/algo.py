import itertools
from typing import Iterable

from rlpyt.agents.base import BaseAgent
from rlpyt.algos.base import RlAlgorithm
from rlpyt.replays.sequence.n_step import SamplesFromReplay
from rlpyt.utils.buffer import buffer_method, buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import infer_leading_dims
import torch
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm

from agents.adapt.replay import initialize_replay_buffer, samples_to_buffer
from agents.adapt.utils import FreezeParameters, get_params

loss_info_fields = ["model_loss", "actor_loss", "value_loss", "prior_entropy", "posterior_entropy", "divergence",
                    "reward_loss", "obs_loss"]

LossInfo = namedarraytuple("LossInfo", loss_info_fields)
OptInfo = namedarraytuple("OptInfo",
                          ["loss", "grad_clip_model", "grad_clip_actor", "grad_clip_value"] + loss_info_fields)

class Dreamer(RlAlgorithm):
  def __init__(
    self,
    batch_size=50,
    batch_length=50,
    train_every=10,
    train_steps=10,
    pretrain=10,
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
    replay_size=int(5e6),
    replay_ratio=8,
    n_step_return=1,
    updates_per_sync=1,
    free_nats=3,
    kl_balancing=0.8,
    kl_scale=0.1,
    type=torch.float,
    prefill=500,
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

  def initialize(self, agent: BaseAgent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0):
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
      # model.transition_model # TODO(PrabSG@): Pytorch warns of duplicate parameters being passed as representation model uses transition model
    ]

    self.actor_modules = [model.action_model]
    self.value_modules = [model.value_model]
    self.model_optimizer = torch.optim.Adam(itertools.chain(*get_params(self.model_modules)), lr=self.model_lr,
                                            **self.optim_kwargs)
    self.actor_optimizer = torch.optim.Adam(itertools.chain(*get_params(self.actor_modules)), lr=self.actor_lr,
                                            **self.optim_kwargs)
    self.value_optimizer = torch.optim.Adam(itertools.chain(*get_params(self.value_modules)), lr=self.value_lr,
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
      self.replay_buffer.append_samples(samples_to_buffer(samples))
    
    opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
    if itr < self.prefill:
      return opt_info
    print("Buffer cursor:", self.replay_buffer.t)
    if itr % self.train_every != 0:
      print("train every")
      return opt_info
    print("optimising")
    
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

      grad_clip_model = torch.nn.utils.clip_grad_norm_(itertools.chain(*get_params(self.model_modules)), self.grad_clip)
      grad_clip_actor = torch.nn.utils.clip_grad_norm_(itertools.chain(*get_params(self.actor_modules)), self.grad_clip)
      grad_clip_value = torch.nn.utils.clip_grad_norm_(itertools.chain(*get_params(self.value_modules)), self.grad_clip)

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
    init_states = model.representation_model.initial_state(batch_b, device=actions.device, dtype=actions.dtype)
    # Rollout model to compare against seen experience
    priors, posteriors = model.rollout.rollout_representation(batch_t, encoded, actions, init_states)

    # Compute Losses
    # Model Loss
    state = torch.cat((posteriors.stoch, posteriors.det), dim=-1)
    pred_obs = model.decoder(state)
    pred_reward = model.reward_model(state)
    reward_loss = -torch.mean(pred_reward.log_prob(rewards))
    obs_loss = -torch.mean(pred_obs.log_prob(observations))
    # Calculate KL Divergence using KL balancing to train priors quicker
    kl_div = self.kl_balancing * td.kl_divergence(
      td.Normal(posteriors.mean.detach(), posteriors.std.detach()),
      td.Normal(priors.mean, priors.std)
    )
    kl_div += (1 - self.kl_balancing) * td.kl_divergence(
      td.Normal(posteriors.mean, posteriors.std),
      td.Normal(priors.mean.detach(), priors.std.detach())
    )
    kl_loss = torch.mean(kl_div, dim=0)
    if self.free_nats is not None:
      kl_loss = torch.max(kl_loss - self.free_nats, torch.zeros_like(kl_loss, device=kl_loss.device))
    model_loss = self.kl_scale * kl_loss + reward_loss + obs_loss

    # Actor Loss
    with torch.no_grad():
      flattened_posterior = buffer_method(posteriors, "reshape", batch_size, -1)
    # Rollout policy for horizon, H, steps
    with FreezeParameters(self.model_modules):
      imag_rssm_state, _ = model.rollout.rollout_policy(self.horizon, model.policy, flattened_posterior)
    imag_state = torch.cat((imag_rssm_state.stoch, imag_rssm_state.det), dim=-1)
    with FreezeParameters(self.model_modules + self.value_modules):
      imag_reward = model.reward_model(imag_state).mean
      value = model.value_model(imag_state).mean
    returns = self.compute_return(imag_reward[:-1], value[:-1], self.discount, bootstrap=value[-1], lambda_=self.discount_lambda)
    # TODO(PrabSG@): Look at predicting discount value due to early terminations in environment which modifies loss function
    actor_loss = -torch.mean(returns)

    # Value Loss
    with torch.no_grad():
      value_state = imag_state[:-1].detach()
      target_return = returns.detach()
    pred_value = model.value_model(value_state)
    value_loss = -torch.mean(pred_value.log_prob(target_return))

    # Loss info
    with torch.no_grad():
      prior_entropy = torch.mean(td.Normal(priors.mean, priors.std).entropy())
      posterior_entropy = torch.mean(td.Normal(posteriors.mean, posteriors.std).entropy())
      loss_info = LossInfo(model_loss, actor_loss, value_loss, prior_entropy, posterior_entropy, kl_div, reward_loss, obs_loss)
      
      # TODO(PrabSG@): Optionally add visualisation code here
    print(f"Model Loss: {model_loss}, Actor Loss: {actor_loss}, Value Loss: {value_loss}, KL Div: {kl_div}")

    return model_loss, actor_loss, value_loss, loss_info

  def compute_return(self, imag_reward, pred_value, discount, bootstrap, lambda_):
    """Compute the discounted reward given a batch of imagined rewards and values."""
    next_values = torch.cat([pred_value[1:], bootstrap.unsqueeze(0)], dim=0)
    discount = discount * torch.ones_like(imag_reward)
    target = imag_reward + discount * next_values * (1 - lambda_)
    outputs = []
    acc_reward = bootstrap
    timesteps = range(imag_reward.shape[0] - 1, -1, -1)
    for t in timesteps:
      acc_reward = target[t] + discount[t] * lambda_ * acc_reward
      outputs.append(acc_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns

    