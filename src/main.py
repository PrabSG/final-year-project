import argparse

import torch

from agents.agent import RandomAgent
from agents.ddqn import DDQNAgent, DDQNParams
from env import BasicEnv
from utils import plot_training

AGENT_CHOICES = ["random", "ddqn"]
ENV_CHOICES = ["basic"]
MAX_EPISODE_LENGTH = 50

ddqn_params = (100, 100, 10000, 256, [32, 32, 64], 0.001, 100, 0.99, 0.9, 0.05, 10)

def init_env(env_type):
  if env_type == "basic":
    return BasicEnv()
  else:
    raise ValueError(f"Environment Type '{env_type}' not defined.")

def init_agent(agent_type, env):
  if agent_type == "random":
    return RandomAgent(env.state_size, env.action_size, env.action_value_range)
  elif agent_type == "ddqn":
    params = DDQNParams(*ddqn_params, device=device)
    return DDQNAgent(env.state_size, env.action_size, params)
  else:
    raise ValueError(f"Agent Type '{agent_type}' not defined.")

def initialise(cl_args):
  env = init_env(cl_args.env)
  agent = init_agent(cl_args.agent, env)
  return env, agent

def train_agent(env, agent, cl_args):
  return agent.train(env)

def run_agent(env, agent, cl_args):
  env.reset()
  agent.evaluate()

  timestep = 0

  trace = []

  observation = env.get_observation()

  while not env.is_complete() and timestep <= cl_args.max_episode_length:
    action = agent.choose_action(observation)
    new_observation, reward, _ = env.step(action)

    trace.append((observation, action, reward, new_observation))
    print("State:", observation, "Action:", action, "Reward:", reward, "New State:", new_observation)

    timestep += 1
    observation = new_observation

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Agent and Environment options")
  parser.add_argument("--env", default="basic", choices=ENV_CHOICES, help="Environment Type")
  parser.add_argument("--agent", default="random", choices=AGENT_CHOICES, help="Agent Type")
  parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
  parser.add_argument("--max-episode-length", type=int, default=MAX_EPISODE_LENGTH, help="Maximum number of steps per episode")

  cl_args = parser.parse_args()

  global device
  device = torch.device("cuda" if torch.cuda.is_available() and not cl_args.disable_cuda else "cpu") # Configuring Pytorch

  env, agent = initialise(cl_args)
  n_episodes, episode_rs, n_steps, train_losses = train_agent(env, agent, cl_args)
  plot_training(n_episodes, episode_rs, n_steps, train_losses)
  run_agent(env, agent, cl_args)