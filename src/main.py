import argparse

import numpy as np

from agents.agent import RandomAgent
from envs.env import BasicEnv, MiniGridEnvWrapper
from envs.gym_minigrid.register import register

AGENT_CHOICES = ["random"]
ENV_CHOICES = ["basic", "unsafe-small", "unsafe-med"]
MAX_EPISODE_LENGTH = 100
MAX_TIME_STEPS = 100
VISUALISATION_EPISODES = 5

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
  else:
    raise ValueError(f"Agent Type '{agent_type}' not defined.")

def initialise(args):
  env = init_env(args)
  agent = init_agent(args.agent, env)
  return env, agent

def train_agent(env, agent, args):
  agent.train(env)

def run_agent(env, agent, args):
  env.reset()

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
  # Script options
  parser.add_argument("--disable-cuda", action="store_false", help="Disable CUDA")
  parser.add_argument("--gif", type=str, help="Filename for visualisation episodes gif")
  parser.add_argument("--vis-eps", type=int, default=VISUALISATION_EPISODES, help="Number of episodes to visualise")

  args = parser.parse_args()

  # TODO: Figure out how to correctly register environments
  register(
    id="MiniGrid-UnsafeCrossingN1-v0",
    entry_point="envs.gym_minigrid.envs:UnsafeCrossingSmallEnv",
    kwargs={"max_steps": args.max_episode_length}
  )

  register(
    id="MiniGrid-UnsafeCrossingN2-v0",
    entry_point="envs.gym_minigrid.envs:UnsafeCrossingMedEnv",
    kwargs={"max_steps": args.max_episode_length}
  )
  
  env, agent = initialise(args)
  train_agent(env, agent, args)
  run_agent(env, agent, args)
  visualise_agent(env, agent, args)
