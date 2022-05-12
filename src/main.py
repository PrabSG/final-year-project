import argparse
import os

from array2gif import write_gif
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import gym_minigrid # Required import for env registration
from agents.agent import RandomAgent
from agents.ddqn import DDQNAgent, DDQNParams
from agents.ls_dreamer import LSDreamerParams, LatentShieldedDreamer
from envs.env import init_env, BasicEnv, MiniGridEnvWrapper
from utils import plot_training

AGENT_CHOICES = ["random", "ddqn", "ls-dreamer"]
ENV_CHOICES = ["basic", "unsafe-micro", "unsafe-small", "unsafe-med"]
MAX_EPISODE_LENGTH = 50
NUM_TRAINING_EPISODES = 100
NUM_TESTING_EPISODES = 10
VISUALISATION_EPISODES = 5

ddqn_params = (1000, 256, [32, 32, 64], 0.001, 100, 0.9, 0.9, 0.05, 25)


def init_agent(agent_type, env, args):
  if agent_type == "random":
    return RandomAgent(env.state_size, env.action_size)
  elif agent_type == "ddqn":
    params = DDQNParams(args.train_episodes, args.max_episode_length, *ddqn_params, encoding_size=32, cnn_channels=[8, 16, 16], cnn_kernels=[3, 3, 5], device=device)
    return DDQNAgent(env.state_size, env.action_size, params)
  elif agent_type == "ls-dreamer":
    params = LSDreamerParams(args, args.results_dir, episodes=args.train_episodes, test=True, test_interval=25, test_episodes=5, max_episode_length=args.max_episode_length, device=device)
    return LatentShieldedDreamer(params, env)
  else:
    raise ValueError(f"Agent Type '{agent_type}' not defined.")

def initialise(args):
  env = init_env(args)
  agent = init_agent(args.agent, env, args)
  return env, agent

def train_agent(env, agent, args):
  return agent.train(env, writer=args.writer)
    
def visualise_agent(env, agent, args):
  """Run agent in environment and visualise agent's path."""
  _, frames = agent.run_tests(args.vis_eps, env, args, visualise=True)

  print("Saving gif... ", end="")
  write_gif(np.array(frames), args.gif+".gif", fps=1/0.1)
  print("Done.")
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Agent and Environment options")
  # Environment arguments
  parser.add_argument("--env", default="basic", choices=ENV_CHOICES, help="Environment Type")
  parser.add_argument("--max-episode-length", type=int, default=MAX_EPISODE_LENGTH, help="Maximum number of steps per episode")
  # Agent arguments
  parser.add_argument("--agent", default="random", choices=AGENT_CHOICES, help="Agent Type")
  parser.add_argument("--train-episodes", type=int, default=NUM_TRAINING_EPISODES, help="Number of episodes allocated for training the agent")
  parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
  # Script options
  parser.add_argument("--id", type=str, help="ID for results of run")
  parser.add_argument("--test-episodes", type=int, default=NUM_TESTING_EPISODES, help="Number of episodes to test the agent")
  parser.add_argument("--plot", type=str, help="Filename for training losses plot")
  parser.add_argument("--gif", type=str, help="Filename for visualisation episodes gif")
  parser.add_argument("--vis-eps", type=int, default=VISUALISATION_EPISODES, help="Number of episodes to visualise")

  args = parser.parse_args()

  # Set additional arguments
  global device
  device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu") # Configuring Pytorch
  print("Using device:", device)

  results_dir = os.path.join('../results/', '{}_{}'.format(args.env, args.id))
  os.makedirs(results_dir, exist_ok=True)
  args.results_dir = results_dir

  summary_name = results_dir + "/{}_{}_log"
  writer = SummaryWriter(summary_name.format(args.env, args.id))
  args.writer = writer

  # Start running agent
  env, agent = initialise(args)
  train_agent(env, agent, args)
  # n_episodes, episode_rs, n_steps, train_losses = train_agent(env, agent, args)
  # plot_training(n_episodes, episode_rs, n_steps, train_losses, args.plot)
  agent.run_tests(args.test_episodes, env, args, print_logging=True)
  if args.gif != "":
    visualise_agent(env, agent, args)
