import argparse

import numpy as np
import torch

from agents.agent import RandomAgent
from agents.ddqn import DDQNAgent, DDQNParams
from envs.env import BasicEnv, MiniGridEnvWrapper
from utils import plot_training

AGENT_CHOICES = ["random", "ddqn"]
ENV_CHOICES = ["basic", "unsafe-small", "unsafe-med"]
MAX_EPISODE_LENGTH = 50
VISUALISATION_EPISODES = 5

ddqn_params = (100, 100, 10000, 256, [32, 32, 64], 0.001, 100, 0.99, 0.9, 0.05, 10)

def init_env(args):
  if args.env == "basic":
    return BasicEnv()
  elif args.env == "unsafe-small":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingN1-v0", max_steps=args.max_episode_length)
  elif args.env == "unsafe-med":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingN2-v0", max_steps=args.max_episode_length)
  else:
    raise ValueError(f"Environment Type '{args.env}' not defined.")

def init_agent(agent_type, env):
  if agent_type == "random":
    return RandomAgent(env.state_size, env.action_size)
  elif agent_type == "ddqn":
    params = DDQNParams(*ddqn_params, device=device)
    return DDQNAgent(env.state_size, env.action_size, params)
  else:
    raise ValueError(f"Agent Type '{agent_type}' not defined.")

def initialise(args):
  env = init_env(args)
  agent = init_agent(args.agent, env)
  return env, agent

def train_agent(env, agent, args):
  return agent.train(env)

def run_agent(env, agent, args):
  env.reset()
  agent.evaluate()

  timestep = 0

  trace = []

  observation = env.get_observation()
  str_grid = str(env)

  while not env.is_complete() and timestep <= args.max_episode_length:
    action = agent.choose_action(observation)
    new_observation, reward, done, _ = env.step(action)

    new_str_grid = str(env)
    trace.append((str_grid, action, reward, new_str_grid))

    timestep += 1
    observation = new_observation
    str_grid = new_str_grid

  for s, a, r, new_s in trace:
    print("State:\n", s, "\nAction:", a, "Reward:", r, "\nNew State:\n", new_s)

def visualise_agent(env, agent, args):
  """Run agent in environment and visualise agent's path."""
  if args.gif:
    from array2gif import write_gif
    frames = []

  # TODO: Add render function to environment interface to work with non-minigrid envs.
  env._env.render('human')

  for _ in range(args.vis_eps):
    obs = env.reset()

    while True:
      env._env.render('human')
      if args.gif:
        frames.append(np.moveaxis(env._env.render("rgb_array"), 2, 0))

      action = agent.choose_action(obs)
      obs, _, done, _ = env.step(action)
      if done or env._env.window.closed:
        break

    if env._env.window.closed:
        break

  if args.gif:
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
  parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
  parser.add_argument("--max-episode-length", type=int, default=MAX_EPISODE_LENGTH, help="Maximum number of steps per episode")
  # Script options
  parser.add_argument("--gif", type=str, help="Filename for visualisation episodes gif")
  parser.add_argument("--vis-eps", type=int, default=VISUALISATION_EPISODES, help="Number of episodes to visualise")

  args = parser.parse_args()

  global device
  device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu") # Configuring Pytorch

  env, agent = initialise(args)
  n_episodes, episode_rs, n_steps, train_losses = train_agent(env, agent, args)
  plot_training(n_episodes, episode_rs, n_steps, train_losses)
  run_agent(env, agent, args)
  visualise_agent(env, agent, args)
