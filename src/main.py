import argparse

from agents.agent import RandomAgent
from envs.env import BasicEnv, MiniGridEnvWrapper
from envs.gym_minigrid.register import register

AGENT_CHOICES = ["random"]
ENV_CHOICES = ["basic", "unsafe-small", "unsafe-med"]
MAX_EPISODE_LENGTH = 100
MAX_TIME_STEPS = 100

def init_env(args):
  if args.env == "basic":
    return BasicEnv()
  elif args.env == "unsafe-small":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingN1-v0")
  elif args.env == "unsafe-med":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingN2-v0")
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

  while not env.is_complete() and timestep <= args.max_episode_length:
    action = agent.choose_action(observation)
    new_observation, reward, done, _ = env.step(action)

    trace.append((observation, action, reward, new_observation))

    timestep += 1
    observation = new_observation

  for s, a, r, new_s in trace:
    print("State:", s, "Action:", a, "Reward:", r, "New State:", new_s)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Agent and Environment options")
  parser.add_argument("--env", default="basic", choices=ENV_CHOICES, help="Environment Type")
  parser.add_argument("--agent", default="random", choices=AGENT_CHOICES, help="Agent Type")
  parser.add_argument("--disable-cuda", action="store_false", help="Disable CUDA")
  parser.add_argument("--max-episode-length", type=int, default=MAX_EPISODE_LENGTH, help="Maximum number of steps per episode")

  # TODO: Figure out how to correctly register environments
  register(
    id="MiniGrid-UnsafeCrossingN1-v0",
    entry_point="envs.gym_minigrid.envs:UnsafeCrossingSmallEnv"
  )

  register(
    id="MiniGrid-UnsafeCrossingN2-v0",
    entry_point="envs.gym_minigrid.envs:UnsafeCrossingMedEnv"
  )

  args = parser.parse_args()
  env, agent = initialise(args)
  train_agent(env, agent, args)
  run_agent(env, agent, args)