import argparse
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import gym_minigrid # Required import for env registration
from agents.agent import RandomAgent
from agents.ddqn import DDQNAgent, DDQNParams
from agents.ls_dreamer import LSDreamerParams, LatentShieldedDreamer
from agents.safety_ddqn import SafetyDDQNAgent
from envs.env import init_env
from utils import plot_training, visualise_agent

AGENT_CHOICES = ["random", "ddqn", "safety-ddqn" "ls-dreamer"]
ENV_CHOICES = ["basic", "unsafe-simple", "unsafe-micro", "unsafe-small", "unsafe-med", "twopath", "safety-simple"]
MAX_EPISODE_LENGTH = 50
SEED_EPISODES = 5
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
    params = DDQNParams(args.train_episodes, args.max_episode_length, *ddqn_params, encoding_size=64, cnn_channels=[16, 32, 64], cnn_kernels=[3, 3, 5], device=device)
    return SafetyDDQNAgent(env.state_size, env.action_size, params)
  elif agent_type == "ls-dreamer":
    params = LSDreamerParams(
      args, args.results_dir, episodes=args.train_episodes, test=True, test_interval=25,
      test_episodes=5, max_episode_length=args.max_episode_length, embedding_size=256,
      vis_freq=args.vis_freq, seed_episodes=args.seed_episodes, planning_horizon=5, belief_size=200,
      state_size=30, eps_decay=args.eps_decay, worldmodel_LogProbLoss=True,
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
  parser.add_argument("--vis-freq", type=int, default=VISUALISATION_FREQUENCY, help="Number of episodes between ")

  args = parser.parse_args()

  # Set additional arguments
  global device
  device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu") # Configuring Pytorch
  print("Using device:", device)

  results_dir = os.path.join(args.results_dir, '{}_{}'.format(args.env, args.id))
  os.makedirs(results_dir, exist_ok=True)
  args.results_dir = results_dir

  summary_name = results_dir + "/{}_{}_log"
  writer = SummaryWriter(summary_name.format(args.env, args.id))
  args.writer = writer

  # Start running agent
  env, agent = initialise(args)
  train_agent(env, agent, args)
  # n_episodes, episode_rs, n_steps, train_losses = train_agent(env, agent, args)
  # plot_training(n_episodes, episode_rs, n_steps, train_losses, results_dir + "/train_plots")
  agent.run_tests(args.test_episodes, env, args, print_logging=True)
  visualise_agent(env, agent, args)
