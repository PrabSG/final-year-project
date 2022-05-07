"""Latent Shielded Dreamer agent using Approximate Bounded Prescience for Latent Trajectories.

Taken from repository for 'Do Androids Dream of Electric Fences? Safe Reinforcement Learning with
Imagination-Based Agents' by Peter He."""

import os

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from agents.agent import Agent
from agents.ls_dreamer.planner import MPCPlanner
from agents.ls_dreamer.memory import ExperienceReplay
from agents.ls_dreamer.models import ActorModel, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ViolationModel, bottle

class LatentShieldedDreamer(Agent):
  def __init__(self, params, env):
    super().__init__()

    self.metrics = {
      'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [],
      'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': [],
      'violation_loss': []
    }

    # TODO(@PrabSG): Check env.observation_size will work with env.state_size
    self.D = ExperienceReplay(params.experience_size, params.symbolic_env, env.observation_size, env.action_size, params.bit_depth, params.device)

    # Initialise Models
    self.transition_model = TransitionModel(params.belief_size, params.state_size, env.action_size, params.hidden_size, params.embedding_size, params.dense_activation_function).to(device=params.device)
    self.observation_model = ObservationModel(params.symbolic_env, env.observation_size, params.belief_size, params.state_size, params.embedding_size, params.cnn_activation_function).to(device=params.device)
    self.reward_model = RewardModel(params.belief_size, params.state_size, params.hidden_size, params.dense_activation_function).to(device=params.device)
    self.violation_model = ViolationModel(params.belief_size, params.state_size, params.hidden_size, params.dense_activation_function).to(device=params.device)
    self.encoder = Encoder(params.symbolic_env, env.observation_size, params.embedding_size, params.cnn_activation_function).to(device=params.device)
    self.actor_model = ActorModel(params.belief_size, params.state_size, params.hidden_size, env.action_size, params.dense_activation_function).to(device=params.device)
    self.value_model = ValueModel(params.belief_size, params.state_size, params.hidden_size, params.dense_activation_function).to(device=params.device)
    param_list = list(self.transition_model.parameters()) + list(self.observation_model.parameters()) + list(self.reward_model.parameters()) + list(self.violation_model.parameters()) + list(self.encoder.parameters())
    value_actor_param_list = list(self.value_model.parameters()) + list(self.actor_model.parameters())
    params_list = param_list + value_actor_param_list # TODO(@PrabSG): Check redundant
    self.model_optimizer = optim.Adam(param_list, lr=0 if params.learning_rate_schedule != 0 else params.model_learning_rate, eps=params.adam_epsilon)
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
    
    # Choose planning algorithm
    if params.algo == "dreamer":
      self.planner = self.actor_model
    else:
      self.planner = MPCPlanner(env.action_size, params.planning_horizon, params.optimisation_iters, params.candidates, params.top_candidates, self.transition_model, self.reward_model)

    self.global_prior = Normal(torch.zeros(params.batch_size, params.state_size, device=params.device), torch.ones(params.batch_size, params.state_size, device=params.device))  # Global prior N(0, I)
    self.free_nats = torch.full((1, ), params.free_nats, device=params.device)  # Allowed deviation in KL divergence

  def _class_weighted_bce_loss(self, pred, target, positive_weight, negative_weight):
    # Calculate class-weighted BCE loss
    return negative_weight * (target - 1) * torch.clamp(torch.log(1 - pred), -100, 0) - positive_weight * target * torch.clamp(torch.log(pred), -100, 0)
    
  def _update_belief_and_act(self, args, env, planner, transition_model, violation_model, encoder, belief, posterior_state, action, observation, violation, explore=False):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action.size()) torch.Size([1, 6])
    belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
    imagd_violation = torch.argmax(bottle(violation_model, (belief, posterior_state)).squeeze())

    if not isinstance(violation, torch.Tensor): 
      if violation == 1 and imagd_violation > 0.8:
        print('correctly pred violation')
      elif violation == 1 and imagd_violation < 0.8:
        print('missed violation')
      elif violation == 0 and imagd_violation > 0.8:
        print('incorrectly pred violation')

    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
    if args.algo=="dreamer":
      action = planner.get_action(belief, posterior_state, det=not(explore)).to(args.device)
    else:
      action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
      action = torch.clamp(Normal(action.float(), args.action_noise).rsample(), -1, 1).to(args.device) # Add gaussian exploration noise on top of the sampled action
      # action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    next_observation, reward, violation, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())# action[0].cpu())  # Perform environment step (action repeats handled internally)
    reward -= 20 * violation
      
    return belief, posterior_state, action, next_observation, reward, violation, done

  def train(self, env):
    # Initialise Experience Replay D with S random seed episodes
    for s in range(1, params.seed_episodes + 1):
      observation, done, t = env.reset(), False, 0
      while not done:
        # TODO(@PrabSG): Implement env sample random action function in some format
        action = env.sample_random_action()
        next_observation, reward, violation, done = env.step(action)
        if violation:
          reward -= 20
        self.D.append(observation, action, reward, violation, done)
        observation = next_observation
        t += 1
      self.metrics['steps'].append(t * self.params.action_repeat + (0 if len(self.metrics['steps']) == 0 else self.metrics['steps'][-1]))
      self.metrics['episodes'].append(s)


    return super().train(env)
  
  def choose_action(self):
    return super().choose_action()

  def evaluate(self):
    return super().evaluate()