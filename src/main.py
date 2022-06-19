import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from envs.safe_env import SafetyConstrainedEnv

import gym_minigrid # Required import for env registration
from agents.agent import RandomAgent
from agents.ddqn import DDQNAgent, DDQNParams
from agents.ls_dreamer import LSDreamerParams, LatentShieldedDreamer
from agents.safety_ddqn import SafetyDDQNAgent, SafetyDDQNParams
from agents.ddqn_L import DDQNLAgent, DDQNLParams
from safety.utils import get_encoding_size
from utils import plot_agent_variants, visualise_agent, init_env

AGENT_CHOICES = ["random", "ddqn", "safety-ddqn", "ddqn-l", "ls-dreamer"]
ENV_CHOICES = ["basic", "unsafe-simple", "unsafe-micro", "unsafe-small", "unsafe-med", "twopath", "safety-simple", "safety-micro"]
MAX_EPISODE_LENGTH = 50
SEED_EPISODES = 5
CURRICULUM_EQ_EPS = 100
NUM_TRAINING_EPISODES = 100
NUM_TESTING_EPISODES = 10
EPS_DECAY = 200000
VISUALISATION_EPISODES = 5
VISUALISATION_FREQUENCY = 25
DEFAULT_RESULT_DIR = "../results/"

ddqn_params = (10000, 256, [64, 64, 128], 0.001, 100, 0.9, 0.9, 0.05, 25)


def init_agent(agent_type, env, args):
  if agent_type == "random":
    return RandomAgent(env.state_size, env.action_size)
  elif agent_type == "ddqn":
    params = DDQNParams(args.train_episodes, args.max_episode_length, *ddqn_params, encoding_size=64, cnn_channels=[16, 32, 64], cnn_kernels=[3, 3, 5], device=device)
    return DDQNAgent(env.state_size, env.action_size, params)
  elif agent_type == "safety-ddqn":
    assert isinstance(env, SafetyConstrainedEnv)
    params = SafetyDDQNParams(env.get_num_props(), args.train_episodes, args.max_episode_length, *ddqn_params, spec_encoding_hidden_size=64, encoding_size=64, cnn_channels=[16, 32, 64], cnn_kernels=[3, 3, 5], device=device)
    num_props = env.get_num_props() if isinstance(env, SafetyConstrainedEnv) else 0
    return SafetyDDQNAgent(env.state_size, env.action_size, get_encoding_size(num_props), params)
  elif agent_type == "ddqn-l":
    assert isinstance(env, SafetyConstrainedEnv)
    params = DDQNLParams(env.get_num_props(), args.train_episodes, args.max_episode_length, *ddqn_params, spec_encoding_hidden_size=64, encoding_size=64, cnn_channels=[16, 32, 64], cnn_kernels=[3, 3, 5], device=device)
    num_props = env.get_num_props() if isinstance(env, SafetyConstrainedEnv) else 0
    return DDQNLAgent(env.state_size, env.action_size, get_encoding_size(num_props), params)
  elif agent_type == "ls-dreamer":
    params = LSDreamerParams(
      args, args.results_dir, episodes=args.train_episodes, test=True, test_interval=25,
      test_episodes=5, max_episode_length=args.max_episode_length, embedding_size=256,
      vis_freq=args.vis_freq, seed_episodes=args.seed_episodes, planning_horizon=5, belief_size=200,
      state_size=30, eps_decay=args.eps_decay, worldmodel_LogProbLoss=True, algo="MPC"
      device=device)
    return LatentShieldedDreamer(params, env)
  else:
    raise ValueError(f"Agent Type '{agent_type}' not defined.")

def initialise(args):
  env = init_env(args)
  agent = init_agent(args.agent, env, args)
  return env, agent

def train_agent(env, agent, args):
  return agent.train(env, writer=args.writer)
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Agent and Environment options")
  # Environment arguments
  parser.add_argument("--env", default="basic", choices=ENV_CHOICES, help="Environment Type")
  parser.add_argument("--max-episode-length", type=int, default=MAX_EPISODE_LENGTH, help="Maximum number of steps per episode")
  parser.add_argument("--seed", type=int, default=1337, help="Seed used to initialise randomness in environment")
  parser.add_argument("--curriculum-eq-eps", type=int, default=CURRICULUM_EQ_EPS, help="Number of episodes before safety specifications are equally weighted for curriculum learning")
  # Agent arguments
  parser.add_argument("--agent", default="random", choices=AGENT_CHOICES, help="Agent Type")
  parser.add_argument("--train-episodes", type=int, default=NUM_TRAINING_EPISODES, help="Number of episodes allocated for training the agent")
  parser.add_argument("--seed-episodes", type=int, default=SEED_EPISODES)
  parser.add_argument("--eps-decay", type=int, default=EPS_DECAY)
  parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
  # Script options
  parser.add_argument("--id", type=str, required=True, help="ID for results of run")
  parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULT_DIR, help="Path of folder to place logs and results from runs")
  parser.add_argument("--test-episodes", type=int, default=NUM_TESTING_EPISODES, help="Number of episodes to test the agent")
  parser.add_argument("--vis-eps", type=int, default=VISUALISATION_EPISODES, help="Number of episodes to visualise at each visualisation checkpoint")
  parser.add_argument("--vis-freq", type=int, default=VISUALISATION_FREQUENCY, help="Number of episodes between visualisation checkpoints")
  # Plotting options
  parser.add_argument("--plot", action="store_true", help="Plot results into a pgf")
  parser.add_argument("--num-agents", type=int, default=1, help="Number of agents to train for plotting")
  parser.add_argument("--save-metrics", action="store_true", help="Save metrics to a pickled file")

  args = parser.parse_args()

  # Set additional arguments
  global device
  device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu") # Configuring Pytorch
  print("Using device:", device)

  results_dir = os.path.join(args.results_dir, '{}_{}'.format(args.env, args.id))
  os.makedirs(results_dir, exist_ok=True)
  args.results_dir = results_dir

  if args.plot:
    agent_metrics = []
    for n in range(args.num_agents):
      summary_name = results_dir + "/{}_{}_{}_log"
      args.writer = SummaryWriter(summary_name.format(args.env, args.id, n))
      env, agent = initialise(args)
      
      print(f"Training Agent {n+1}...")
      metrics = train_agent(env, agent, args)
      agent_metrics.append(metrics)

      # Progress to new seed or environments will be the same order of configs each time
      random.seed(args.seed)
      args.seed = random.randrange(sys.maxsize)

    if args.save_metrics:
      with open(args.results_dir + f"/metrics_{args.num_agents}_agents.pickle", "wb") as f:
        pickle.dump(agent_metrics, f)
    
    ylabels = agent_metrics[0]["metric_titles"]
    fields = list(agent_metrics[0].keys() - {"steps", "episodes", "metric_titles", "test_episodes"})
    plot_agent_variants([agent_metrics], [None], fields, ylabels, args.results_dir)

  else:
    summary_name = results_dir + "/{}_{}_log"
    writer = SummaryWriter(summary_name.format(args.env, args.id))
    args.writer = writer

    # Start running agent
    env, agent = initialise(args)
    print("Training Agent...")
    train_agent(env, agent, args)
    # n_episodes, episode_rs, n_steps, train_losses = train_agent(env, agent, args)
    # plot_training(n_episodes, episode_rs, n_steps, train_losses, results_dir + "/train_plots")
    # agent.run_tests(args.test_episodes, env, args)
    visualise_agent(env, agent, args)
