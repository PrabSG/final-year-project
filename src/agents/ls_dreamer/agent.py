"""Latent Shielded Dreamer agent using Approximate Bounded Prescience for Latent Trajectories.

Adapted heavily but core optimisation loop taken from repository for 'Do Androids Dream of Electric Fences? Safe Reinforcement Learning with
Imagination-Based Agents' by Peter He."""

import itertools
import os

from array2gif import write_gif
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from agents.agent import Agent
from agents.ls_dreamer.bps import BoundedPrescienceShield, ShieldBatcher
from agents.ls_dreamer.env import EnvBatcher
from agents.ls_dreamer.memory import ExperienceReplay
from agents.ls_dreamer.models import ActorModel, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ViolationModel, bottle
from agents.ls_dreamer.planner import MPCPlanner
from agents.ls_dreamer.utils import FreezeParameters, imagine_ahead, lambda_return, lineplot, write_video
from envs.env import MiniGridEnvWrapper
from utils import visualise_agent

class LSDreamerParams():
  def __init__(self,
               args,
               results_dir,
               algo="dreamer",
               seed=1,
               symbolic_env=False,
               max_episode_length=1000,
               experience_size=1000000,
               cnn_activation_function="relu",
               dense_activation_function="elu",
               embedding_size=1024,
               hidden_size=64,
               belief_size=200,
               state_size=30,
               action_repeat=1,
               eps_max=0.4,
               eps_min=0.1,
               eps_decay=200000,
               episodes=1000,
               seed_episodes=5,
               collect_interval=100,
               batch_size=50,
               chunk_size=50,
               worldmodel_LogProbLoss=False,
               overshooting_distance=50,
               overshooting_kl_beta=0,
               overshooting_reward_scale=0,
               global_kl_beta=0,
               kl_balancing_alpha=0.8,
               kl_scaling_beta=0.1,
               free_nats=3,
               bit_depth=3,
               model_learning_rate=6e-4,
               actor_learning_rate=8e-5,
               value_learning_rate=8e-5,
               learning_rate_schedule=0,
               adam_epsilon=1e-7,
               grad_clip_norm=100.0,
               planning_horizon=15,
               discount=0.99,
               disclaim=0.95,
               optimisation_iters=10,
               candidates=1000,
               top_candidates=100,
               test=False,
               test_interval=25,
               test_episodes=10,
               checkpoint_interval=50,
               checkpoint_experience=False,
               models="",
               experience_replay="",
               render=False,
               paths_to_sample=40,
               violation_threshold=10,
               vis_freq=None,
               device="cpu"):
    self.args = args
    self.results_dir = results_dir
    self.algo = algo
    self.seed = seed
    self.symbolic_env = symbolic_env
    self.max_episode_length = max_episode_length
    self.experience_size = experience_size
    self.cnn_activation_function = cnn_activation_function
    self.dense_activation_function = dense_activation_function
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.belief_size = belief_size
    self.state_size = state_size
    self.action_repeat = action_repeat
    self.eps_max = eps_max
    self.eps_min = eps_min
    self.eps_decay = eps_decay
    self.episodes = episodes
    self.seed_episodes = seed_episodes
    self.collect_interval = collect_interval
    self.batch_size = batch_size
    self.chunk_size = chunk_size
    self.worldmodel_LogProbLoss = worldmodel_LogProbLoss
    self.overshooting_distance = overshooting_distance
    self.overshooting_kl_beta = overshooting_kl_beta
    self.overshooting_reward_scale = overshooting_reward_scale
    self.global_kl_beta = global_kl_beta
    self.kl_balancing_alpha = kl_balancing_alpha
    self.kl_scaling_beta = kl_scaling_beta
    self.free_nats = free_nats
    self.bit_depth = bit_depth
    self.model_learning_rate = model_learning_rate
    self.actor_learning_rate = actor_learning_rate
    self.value_learning_rate = value_learning_rate
    self.learning_rate_schedule = learning_rate_schedule
    self.adam_epsilon = adam_epsilon
    self.grad_clip_norm = grad_clip_norm
    self.planning_horizon = planning_horizon
    self.discount = discount
    self.disclaim = disclaim
    self.optimisation_iters = optimisation_iters
    self.candidates = candidates
    self.top_candidates = top_candidates
    self.test = test
    self.test_interval = test_interval
    self.test_episodes = test_episodes
    self.checkpoint_interval = checkpoint_interval
    self.checkpoint_experience = checkpoint_experience
    self.models = models
    self.experience_replay = experience_replay
    self.render = render
    self.paths_to_sample = paths_to_sample
    self.violation_threshold = violation_threshold
    self.vis_freq = vis_freq
    self.device = device
    

class LatentShieldedDreamer(Agent):
  VIOLATION_REWARD_SCALING = 0.5

  def __init__(self, params: LSDreamerParams, env):
    super().__init__()

    self.params = params
    self.metrics = {
      'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [],
      'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': [],
      'violation_loss': [], 'violation_count': []
    }

    # Initialise epsilon for linear decay
    self._expl_eps = self.params.eps_max
    self._optimize_steps = 0

    self.D = ExperienceReplay(params.experience_size, params.symbolic_env, env.state_size, env.action_size, params.bit_depth, params.device)

    # Initialise Models
    self.transition_model = TransitionModel(params.belief_size, params.state_size, env.action_size, params.hidden_size, params.embedding_size, params.dense_activation_function).to(device=params.device)
    self.observation_model = ObservationModel(params.symbolic_env, env.state_size, params.belief_size, params.state_size, params.embedding_size, params.cnn_activation_function).to(device=params.device)
    self.reward_model = RewardModel(params.belief_size, params.state_size, params.hidden_size, params.dense_activation_function).to(device=params.device)
    self.violation_model = ViolationModel(params.belief_size, params.state_size, params.hidden_size, params.dense_activation_function).to(device=params.device)
    self.encoder = Encoder(params.symbolic_env, env.state_size, params.embedding_size, params.cnn_activation_function).to(device=params.device)
    self.actor_model = ActorModel(params.belief_size, params.state_size, params.hidden_size, env.action_size, params.dense_activation_function).to(device=params.device)
    self.value_model = ValueModel(params.belief_size, params.state_size, params.hidden_size, params.dense_activation_function).to(device=params.device)
    self.param_list = itertools.chain(self.transition_model.parameters(), self.observation_model.parameters(), self.reward_model.parameters(), self.violation_model.parameters(), self.encoder.parameters())
    # value_actor_param_list = list(self.value_model.parameters()) + list(self.actor_model.parameters()) # TODO(@PrabSG): Check redundant
    # params_list = self.param_list + value_actor_param_list # TODO(@PrabSG): Check redundant
    self.model_optimizer = optim.Adam(self.param_list, lr=0 if params.learning_rate_schedule != 0 else params.model_learning_rate, eps=params.adam_epsilon)
    self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=0 if params.learning_rate_schedule != 0 else params.actor_learning_rate, eps=params.adam_epsilon)
    self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=0 if params.learning_rate_schedule != 0 else params.value_learning_rate, eps=params.adam_epsilon)

    # Load pre-trained models if path given
    if params.models != '' and os.path.exists(params.models):
      model_dicts = torch.load(params.models)
      self.transition_model.load_state_dict(model_dicts['transition_model'])
      self.observation_model.load_state_dict(model_dicts['observation_model'])
      self.reward_model.load_state_dict(model_dicts['reward_model'])
      self.violation_model.load_state_dict(model_dicts['violation_model'])
      self.encoder.load_state_dict(model_dicts['encoder'])
      self.actor_model.load_state_dict(model_dicts['actor_model'])
      self.value_model.load_state_dict(model_dicts['value_model'])
      self._optimize_steps = model_dicts['optimize_steps']
    
    # Choose planning algorithm
    if params.algo == "dreamer":
      self.planner = self.actor_model
    else:
      self.planner = MPCPlanner(env.action_size, params.planning_horizon, params.optimisation_iters, params.candidates, params.top_candidates, self.transition_model, self.reward_model)

    # Initialise Shield
    self.shield = BoundedPrescienceShield(self.transition_model, self.violation_model, violation_threshold=params.violation_threshold, paths_to_sample=params.paths_to_sample)

    self.global_prior = Normal(torch.zeros(params.batch_size, params.state_size, device=params.device), torch.ones(params.batch_size, params.state_size, device=params.device))  # Global prior N(0, I)
    
    if params.free_nats != 0:
      self.free_nats = torch.full((1, ), params.free_nats, device=params.device)  # Allowed deviation in KL divergence

  def _class_weighted_bce_loss(self, pred, target, positive_weight, negative_weight):
    # Calculate class-weighted BCE loss
    return negative_weight * (target - 1) * torch.clamp(torch.log(1 - pred), -100, 0) - positive_weight * target * torch.clamp(torch.log(pred), -100, 0)
    
  def _update_belief_and_act(self, env, belief, posterior_state, action, observation, violation, shield, episode, explore=False):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action.size()) torch.Size([1, 6])
    belief, _, _, _, posterior_state, _, _ = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief, self.encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
    imagd_violation = torch.argmax(bottle(self.violation_model, (belief, posterior_state)).squeeze())

    # if not torch.is_tensor(violation):
    #   if violation == 1 and imagd_violation > 0.8:
    #     print('correctly pred violation')
    #   elif violation == 1 and imagd_violation < 0.8:
    #     print('missed violation')
    #   elif violation == 0 and imagd_violation > 0.8:
    #     print('incorrectly pred violation')

    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
    if self.params.algo=="dreamer":
      action = self.planner.get_action(belief, posterior_state, det=not(explore)).to(self.params.device)
    else:
      action = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
      # Exploration is epsilon-greedy for discrete control envs
      curr_eps = self._expl_eps
      self._expl_eps = self.params.eps_max - (((self.params.eps_max - self.params.eps_min) / self.params.eps_decay) * self._optimize_steps)
      self._expl_eps = max(self._expl_eps, self.params.eps_min)

      if np.random.uniform(0, 1) <= curr_eps:
        action = env.sample_random_action().to(self.params.device).unsqueeze(0)
      # action = torch.clamp(Normal(action.float(), self.params.eps_max).rsample(), -1, 1).to(self.params.device) # Add gaussian exploration noise on top of the sampled action

    # shield_interfered = False
    # if episode > 60 or (episode > 40 and episode % 2 == 0):
    #   shield_action, shield_interfered = shield.step(belief, posterior_state, action, self.observation_model, self.planner, observation, self.encoder)
    #   action = shield_action.to(device=self.params.device)
    #   if shield_interfered:
    #     print('interfered')
    next_observation, reward, done, info = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()) # action[0].cpu())  # Perform environment step (action repeats handled internally)
    violation = info['violation']
    # if shield_interfered or (torch.any(violation) if torch.is_tensor(violation) else violation):
    # if (torch.any(violation) if torch.is_tensor(violation) else violation):
    #   reward -= self.VIOLATION_REWARD_SCALING * violation
      
    return belief, posterior_state, action, next_observation, reward, violation, done

  def _optimize_models(self, losses, model_modules):
    print("Latent-Shielded Dreamer Training loop")
    for s in tqdm(range(self.params.collect_interval)):
      # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
      observations, actions, rewards, violations, nonterminals = self.D.sample(self.params.batch_size, self.params.chunk_size) # Transitions start at time t = 0
      # Create initial belief and state for time t = 0
      init_belief, init_state = torch.zeros(self.params.batch_size, self.params.belief_size, device=self.params.device), torch.zeros(self.params.batch_size, self.params.state_size, device=self.params.device)
      # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
      embedded_observations = bottle(self.encoder, (observations[1:], ))
      beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.transition_model(init_state, actions[:-1], init_belief, embedded_observations, nonterminals[:-1])
      # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
      if self.params.worldmodel_LogProbLoss:
        # observation_dist = Normal(bottle(self.observation_model, (beliefs, posterior_states)), 1)
        # observation_loss = -observation_dist.log_prob(observations[1:]).sum(dim=2 if self.params.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
        observation_loss = F.cross_entropy(bottle(self.observation_model, (beliefs, posterior_states)).view(-1, 13, 5, 5), torch.argmax(observations[1:], dim=2).view(-1, 5, 5), reduction="none").sum(dim=2 if self.params.symbolic_env else (1, 2)).mean(dim=(0))

      else: 
        # observation_loss = F.mse_loss(bottle(self.observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=2 if self.params.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))

        observation_loss = F.cross_entropy(bottle(self.observation_model, (beliefs, posterior_states)).view(-1, 13, 5, 5), torch.argmax(observations[1:], dim=2).view(-1, 5, 5), reduction="none").sum(dim=2 if self.params.symbolic_env else (1, 2)).mean(dim=(0))

      if self.params.worldmodel_LogProbLoss:
        reward_dist = Normal(bottle(self.reward_model, (beliefs, posterior_states)),1)
        reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
        # TODO: implement violation loss here
      else:
        reward_loss = F.mse_loss(bottle(self.reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0,1))
        # if episode > 50:
      violation_loss = F.cross_entropy(
        bottle(self.violation_model, (beliefs, posterior_states, )).reshape(len(violations[:-1]) * len(violations), 2), 
        violations[:-1].reshape(len(violations[:-1]) * len(violations)),
        weight=torch.tensor([1.,3.]).to(self.params.device),
        reduction='none'
        ).mean()
        # else:
          # violation_loss = torch.zeros([1]).to(self.params.device) # TODO(@PrabSG): Check why violation loss 0 before 50 episodes
      # transition loss
      kl_loss = self.params.kl_balancing_alpha * kl_divergence(Normal(posterior_means.detach(), posterior_std_devs.detach()), Normal(prior_means, prior_std_devs)).sum(dim=2)
      kl_loss += (1 - self.params.kl_balancing_alpha) * kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means.detach(), prior_std_devs.detach())).sum(dim=2)
      kl_loss = kl_loss.mean(dim=(0,1))
      if self.params.free_nats != 0:
        kl_loss = torch.max(kl_loss - self.free_nats, torch.zeros_like(kl_loss))
      # kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
      if self.params.global_kl_beta != 0:
        kl_loss += self.params.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), self.global_prior).sum(dim=2).mean(dim=(0, 1))
      # Calculate latent overshooting objective for t > 0
      if self.params.overshooting_kl_beta != 0:
        overshooting_vars = []  # Collect variables for overshooting to process in batch
        for t in range(1, self.params.chunk_size - 1):
          d = min(t + self.params.overshooting_distance, self.params.chunk_size - 1)  # Overshooting distance
          t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
          seq_pad = (0, 0, 0, 0, 0, t - d + self.params.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
          # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
          overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad), F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, self.params.batch_size, self.params.state_size, device=self.params.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
        overshooting_vars = tuple(zip(*overshooting_vars))
        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs = self.transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
        seq_mask = torch.cat(overshooting_vars[7], dim=1)
        # Calculate overshooting KL loss with sequence mask
        kl_loss += (1 / self.params.overshooting_distance) * self.params.overshooting_kl_beta * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), self.free_nats).mean(dim=(0, 1)) * (self.params.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 
        # Calculate overshooting reward prediction loss with sequence mask
        if self.params.overshooting_reward_scale != 0: 
          reward_loss += (1 / self.params.overshooting_distance) * self.params.overshooting_reward_scale * F.mse_loss(bottle(self.reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (self.params.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence) 
      # Apply linearly ramping learning rate schedule
      if self.params.learning_rate_schedule != 0:
        for group in self.model_optimizer.param_groups:
          group['lr'] = min(group['lr'] + self.params.model_learning_rate / self.params.model_learning_rate_schedule, self.params.model_learning_rate)
      model_loss = observation_loss + reward_loss + (self.params.kl_scaling_beta * kl_loss) + violation_loss 
      # Update model parameters
      self.model_optimizer.zero_grad()
      model_loss.backward()
      nn.utils.clip_grad_norm_(self.param_list, self.params.grad_clip_norm, norm_type=2)
      self.model_optimizer.step()

      #Dreamer implementation: actor loss calculation and optimization    
      with torch.no_grad():
        vis_observation = observations.detach().cpu().numpy()[1:][0, 0]
        actor_states = posterior_states.detach()
        actor_beliefs = beliefs.detach()
      with FreezeParameters(model_modules):
        imagination_traj, a0s = imagine_ahead(actor_states, actor_beliefs, self.actor_model, self.transition_model, self.params.planning_horizon)
        init_vis_belief = actor_beliefs[0, 0]
        init_vis_state = actor_states[0, 0]
        vis_beliefs = imagination_traj[0][:, 0]
        vis_states = imagination_traj[1][:, 0]
        init_imag_obs = self.observation_model(init_vis_belief.unsqueeze(0), init_vis_state.unsqueeze(0)).detach().cpu().squeeze(0)
        imag_obs = self.observation_model(vis_beliefs, vis_states).detach().cpu()
        pred_obs = [vis_observation, init_imag_obs.numpy()] + [obs.numpy() for obs in imag_obs]
      imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
      with FreezeParameters(model_modules + self.value_model.modules):
        imged_reward = bottle(self.reward_model, (imged_beliefs, imged_prior_states))
        value_pred = bottle(self.value_model, (imged_beliefs, imged_prior_states))
      returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=self.params.discount, lambda_=self.params.disclaim)
      actor_loss = -torch.mean(returns)
      # Update model parameters
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      nn.utils.clip_grad_norm_(self.actor_model.parameters(), self.params.grad_clip_norm, norm_type=2)
      self.actor_optimizer.step()
  
      #Dreamer implementation: value loss calculation and optimization
      with torch.no_grad():
        value_beliefs = imged_beliefs.detach()
        value_prior_states = imged_prior_states.detach()
        target_return = returns.detach()
      value_dist = Normal(bottle(self.value_model, (value_beliefs, value_prior_states)),1) # detach the input tensor from the transition network.
      value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1)) 
      # Update model parameters
      self.value_optimizer.zero_grad()
      value_loss.backward()
      nn.utils.clip_grad_norm_(self.value_model.parameters(), self.params.grad_clip_norm, norm_type=2)
      self.value_optimizer.step() # TODO(@PrabSG): Check if optimizers should step
      
      # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss (5) violation loss
      losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item(), violation_loss.item()])

      return pred_obs, a0s

  def _update_plot_losses(self, losses):
    losses = tuple(zip(*losses))
    self.metrics['observation_loss'].append(np.mean(losses[0]))
    self.metrics['reward_loss'].append(np.mean(losses[1]))
    self.metrics['kl_loss'].append(np.mean(losses[2]))
    self.metrics['actor_loss'].append(np.mean(losses[3]))
    self.metrics['value_loss'].append(np.mean(losses[4]))
    self.metrics['violation_loss'].append(np.mean(losses[5]))
    lineplot(self.metrics['episodes'][-len(self.metrics['observation_loss']):], self.metrics['observation_loss'], 'observation_loss', self.params.results_dir)
    lineplot(self.metrics['episodes'][-len(self.metrics['reward_loss']):], self.metrics['reward_loss'], 'reward_loss', self.params.results_dir)
    lineplot(self.metrics['episodes'][-len(self.metrics['kl_loss']):], self.metrics['kl_loss'], 'kl_loss', self.params.results_dir)
    lineplot(self.metrics['episodes'][-len(self.metrics['actor_loss']):], self.metrics['actor_loss'], 'actor_loss', self.params.results_dir)
    lineplot(self.metrics['episodes'][-len(self.metrics['value_loss']):], self.metrics['value_loss'], 'value_loss', self.params.results_dir)
    lineplot(self.metrics['episodes'][-len(self.metrics['violation_loss']):], self.metrics['violation_loss'], 'violation_loss', self.params.results_dir)

  def _update_plot_rewards(self, t, episode, total_reward, violations):
    self.metrics['steps'].append(t + self.metrics['steps'][-1])
    self.metrics['episodes'].append(episode)
    self.metrics['train_rewards'].append(total_reward)
    self.metrics['violation_count'].append((episode, violations))
    lineplot(self.metrics['episodes'][-len(self.metrics['train_rewards']):], self.metrics['train_rewards'], 'train_rewards', self.params.results_dir)
    lineplot([x for x, y in self.metrics['violation_count']], [y for x, y in self.metrics['violation_count']], 'violation_count', self.params.results_dir)

  def _test_agent(self, env, episode):

    total_rewards, video_frames = self.run_tests(self.params.test_episodes, env, self.params.args, episode=episode)

    # Update and plot reward metrics (and write video if applicable) and save metrics
    self.metrics['test_episodes'].append(episode)
    self.metrics['test_rewards'].append(total_rewards.tolist())
    lineplot(self.metrics['test_episodes'], self.metrics['test_rewards'], 'test_rewards', self.params.results_dir)
    lineplot(np.asarray(self.metrics['steps'])[np.asarray(self.metrics['test_episodes']) - 1], self.metrics['test_rewards'], 'test_rewards_steps', self.params.results_dir, xaxis='step')
    if not self.params.symbolic_env:
      episode_str = str(episode).zfill(len(str(self.params.episodes)))
      # TODO(@PrabSG): Check video output usefulness and performance impact
      # write_video(video_frames, 'test_episode_%s' % episode_str, self.params.results_dir)  # Lossy compression
      # TODO(@PrabSG): Reformat one-hot observations to work with below image saves
      # save_image(torch.as_tensor(video_frames[-1]), os.path.join(self.params.results_dir, 'test_episode_%s.png' % episode_str))
    torch.save(self.metrics, os.path.join(self.params.results_dir, 'metrics.pth'))

  def train(self, env, writer=None):
    # Initialise Experience Replay D with S random seed episodes
    for s in range(1, self.params.seed_episodes + 1):
      violation_count = 0
      observation, done, t = env.reset(), False, 0
      while not done:
        action = env.sample_random_action()
        next_observation, reward, done, info = env.step(action)
        violation = info['violation']
        if violation:
          # reward -= self.VIOLATION_REWARD_SCALING
          violation_count += 1
        self.D.append(observation, action, reward, violation, done)
        observation = next_observation
        t += 1
      self.metrics['violation_count'].append((s, violation_count))
      self.metrics['steps'].append(t * self.params.action_repeat + (0 if len(self.metrics['steps']) == 0 else self.metrics['steps'][-1]))
      self.metrics['episodes'].append(s)
    
    self.train_mode()

    # Training on episodes
    for episode in tqdm(range(self.metrics['episodes'][-1] + 1, self.params.episodes + 1), total=self.params.episodes, initial=self.metrics['episodes'][-1] + 1):
      # Model fitting
      losses = []
      model_modules = (
        self.transition_model.modules +
        self.encoder.modules +
        self.observation_model.modules +
        self.reward_model.modules +
        self.violation_model.modules)

      pred_obs, a0s = self._optimize_models(losses, model_modules)
      # TODO: REMOVE THIS VISUALISATION CODE^^

      self._optimize_steps += 1

      self._update_plot_losses(losses)
      
      # Data collection
      print("Data collection")
      violations = 0
      with torch.no_grad():
        observation, total_reward = env.reset(), 0
        belief, posterior_state, action, violation = torch.zeros(1, self.params.belief_size, device=self.params.device), torch.zeros(1, self.params.state_size, device=self.params.device), torch.zeros(1, env.action_size, device=self.params.device), torch.zeros(1, 1, device=self.params.device)
        pbar = tqdm(range(self.params.max_episode_length // self.params.action_repeat))
        for t in pbar:
          # print("step",t)
          belief, posterior_state, action, next_observation, reward, violation, done = self._update_belief_and_act(env, belief, posterior_state, action, observation.to(device=self.params.device), violation, self.shield, episode, explore=True)
          self.D.append(observation, action.cpu(), reward, violation, done)
          total_reward += reward
          observation = next_observation
          if violation:
            violations += 1
          if self.params.render:
            env.render()
          if done:
            pbar.close()
            break
        
        self._update_plot_rewards(t, episode, total_reward, violations)
      
      # Test model periodically
      if self.params.test and episode % self.params.test_interval == 0:
        self._test_agent(env, episode)
      
      # Logging
      if writer is not None:
        writer.add_scalar("train_reward", self.metrics['train_rewards'][-1], self.metrics['steps'][-1])
        writer.add_scalar("train/episode_reward", self.metrics['train_rewards'][-1], self.metrics['steps'][-1] * self.params.action_repeat)
        writer.add_scalar("observation_loss", self.metrics['observation_loss'][-1], self.metrics['steps'][-1])
        writer.add_scalar("reward_loss", self.metrics['reward_loss'][-1], self.metrics['steps'][-1])
        writer.add_scalar("kl_loss", self.metrics['kl_loss'][-1], self.metrics['steps'][-1])
        writer.add_scalar("actor_loss", self.metrics['actor_loss'][-1], self.metrics['steps'][-1])
        writer.add_scalar("value_loss", self.metrics['value_loss'][-1], self.metrics['steps'][-1])  
      print("episodes: {}, total_steps: {}, train_reward: {}, violations: {} ".format(self.metrics['episodes'][-1], self.metrics['steps'][-1], self.metrics['train_rewards'][-1], self.metrics['violation_count'][-1][1]))

      # Checkpoint models
      if episode % self.params.checkpoint_interval == 0:
        torch.save({'transition_model': self.transition_model.state_dict(),
                    'observation_model': self.observation_model.state_dict(),
                    'reward_model': self.reward_model.state_dict(),
                    'violation_model': self.violation_model.state_dict(),
                    'encoder': self.encoder.state_dict(),
                    'actor_model': self.actor_model.state_dict(),
                    'value_model': self.value_model.state_dict(),
                    'model_optimizer': self.model_optimizer.state_dict(),
                    'actor_optimizer': self.actor_optimizer.state_dict(),
                    'value_optimizer': self.value_optimizer.state_dict(),
                    'optimize_steps': self._optimize_steps
                    }, os.path.join(self.params.results_dir, 'models_%d.pth' % episode))
        if self.params.checkpoint_experience:
          torch.save(self.D, os.path.join(self.params.results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes
      
      if self.params.vis_freq is not None and episode % self.params.vis_freq == 0:
        # TODO: REMOVE THIS IMAGINE VISUALISATIONS
        print(f"Episode {episode} - imagined actions")
        action_str = ""
        for a in a0s:
          # print("Action taken:", env._one_hot_to_action_enum(a.cpu()))
          action_str += str(torch.argmax(a.cpu()).item())
        pred_frames = []
        for obs in pred_obs:
          obs = self._one_hot_to_encoded_observations(obs)
          obs_t = obs.transpose(1, 2, 0)
          pred_frames.append(np.moveaxis(env._env.get_obs_render(obs_t), 2, 0))
        imag_dir = self.params.args.results_dir + f"/imagine_preds"
        os.makedirs(imag_dir, exist_ok=True)
        pred_fname = f"/imagine_ep{episode}_pred_action_{action_str}.gif"
        write_gif(pred_frames, imag_dir + pred_fname, fps=1/0.25)

        obs, preds = self._visualise_observation_prediction(env, episode=episode)

        save_dir = self.params.args.results_dir + f"/ep{episode}_frame_preds"
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(obs)):
          ob_fname = f"/frame_ep{episode}_f{i}_seen.gif"
          pred_fname = f"/frame_ep{episode}_f{i}_pred.gif"
          write_gif(obs[i], save_dir + ob_fname)
          write_gif(preds[i], save_dir + pred_fname)

        visualise_agent(env, self, self.params.args, episode=episode)

    # Close training environment
    env.close()

  def choose_action(self):
    # TODO(@PrabSG): Deprecate this function and find better way to yield states for visualisations
    return super().choose_action()

  def _one_hot_to_encoded_observations(self, obs):
    obs_obj_idxs = np.argmax(obs, axis=0)
    encoded_obs = np.zeros((3, *obs.shape[1:]))
    for i in range(obs.shape[1]):
      for j in range(obs.shape[2]):
        encoding = MiniGridEnvWrapper._obj_idx_to_default_encoding(obs_obj_idxs[i, j])
        for k in range(len(encoding)):
          encoded_obs[k, i, j] = encoding[k]
    return encoded_obs

  def _visualise_observation_prediction(self, env, episode=None):
    if episode is None:
      epsiode = self.params.episodes
    
    obs_frames = []
    pred_frames = []

    self.evaluate_mode()

    with torch.no_grad():
      observation, total_reward = env.reset(), 0
      belief, posterior_state, action = torch.zeros(1, self.params.belief_size, device=self.params.device), torch.zeros(1, self.params.state_size, device=self.params.device), torch.zeros(1, env.action_size, device=self.params.device)
      violation = torch.zeros(1, 1, device=self.params.device)
      done = False
      pbar = tqdm(range(self.params.max_episode_length // self.params.action_repeat))

      for t in pbar:
        obs_frames.append(np.moveaxis(env._env.get_obs_render(self._one_hot_to_encoded_observations(observation.cpu().numpy()).transpose(1, 2, 0)), 2, 0))

        belief, posterior_state, action, next_observation, reward, violation, done = self._update_belief_and_act(env, belief, posterior_state, action, observation.to(device=self.params.device), violation, None, episode)
        total_reward += reward
        pred_obs = self.observation_model(belief, posterior_state).cpu()
        pred_obs = self._one_hot_to_encoded_observations(pred_obs.squeeze().numpy())
        pred_frames.append(np.moveaxis(env._env.get_obs_render(pred_obs.transpose(1, 2, 0)), 2, 0))
        observation = next_observation
        if done:
          pbar.close()
          break
      
    self.train_mode()

    return obs_frames, pred_frames


  def run_tests(self, n_episodes, env, args, print_logging=False, visualise=False, episode=None):
    # If running tests at no particular episode, run 'after' all training episodes
    if episode is None:
      episode = self.params.episodes

    self.evaluate_mode()

    if visualise:
      frames = {}
      for i in range(n_episodes):
        frames[i] = []

    # Initialise parallelised test environments
    test_envs = EnvBatcher(self.params.args, n_episodes)
    shield = ShieldBatcher(BoundedPrescienceShield, test_envs.envs, self.transition_model, self.violation_model, violation_threshold=self.params.violation_threshold, paths_to_sample=self.params.paths_to_sample)
    with torch.no_grad():
      observation, total_rewards, video_frames = test_envs.reset(), torch.zeros((n_episodes)), []
      belief, posterior_state, action = torch.zeros(n_episodes, self.params.belief_size, device=self.params.device), torch.zeros(n_episodes, self.params.state_size, device=self.params.device), torch.zeros(n_episodes, env.action_size, device=self.params.device)
      violation = torch.zeros(1,1, device=self.params.device)
      done = torch.zeros(n_episodes, dtype=torch.bool, device=self.params.device)
      pbar = tqdm(range(self.params.max_episode_length // self.params.action_repeat))
      num_steps = torch.zeros(n_episodes, dtype=torch.long)
      for t in pbar:        
        if visualise:
          for i in range(n_episodes):
            if not done[i]:
              frames[i].append(np.moveaxis(test_envs.envs[i]._env.render("rgb_array"), 2, 0))

        belief, posterior_state, action, next_observation, reward, violation, done = self._update_belief_and_act(test_envs, belief, posterior_state, action, observation.to(device=self.params.device), violation, shield, episode)
        total_rewards = total_rewards + reward
        num_steps[torch.logical_not(done)] += 1
        if not self.params.symbolic_env:  # Collect real vs. predicted frames for video
          video_frames.append(make_grid(torch.cat([observation, self.observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
        observation = next_observation
        if torch.all(done) == n_episodes:
          pbar.close()
          break

    self.train_mode()

    # Close test environments
    test_envs.close()

    if print_logging:
      for episode in range(n_episodes):
        print(f"Episode {episode+1} total reward: {total_rewards[episode]} - {num_steps[episode]} steps")

    if visualise:
      gif_frames = np.concatenate([frames[i] for i in range(n_episodes)])

    return total_rewards, video_frames if not visualise else gif_frames

  def evaluate_mode(self):
    # Set models to eval mode
    self.transition_model.eval()
    self.observation_model.eval()
    self.reward_model.eval() 
    self.violation_model.eval()
    self.encoder.eval()
    self.actor_model.eval()
    self.value_model.eval()
  
  def train_mode(self):
    # Set models to train mode
    self.transition_model.train()
    self.observation_model.train()
    self.reward_model.train()
    self.violation_model.train()
    self.encoder.train()
    self.actor_model.train()
    self.value_model.train()
