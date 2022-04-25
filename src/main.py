import argparse

from agents.agent import RandomAgent
from envs.env import BasicEnv

AGENT_CHOICES = ["random"]
ENV_CHOICES = ["basic"]
MAX_EPISODE_LENGTH = 100
MAX_TIME_STEPS = 100

def init_env(env_type):
  if env_type == "basic":
    return BasicEnv()
  else:
    raise ValueError(f"Environment Type '{env_type}' not defined.")

def init_agent(agent_type, env):
  if agent_type == "random":
    return RandomAgent(env.state_size, env.action_size)
  else:
    raise ValueError(f"Agent Type '{agent_type}' not defined.")

def initialise(cl_args):
  env = init_env(cl_args.env)
  agent = init_agent(cl_args.agent, env)
  return env, agent

def train_agent(env, agent, cl_args):
  agent.train(env)

def run_agent(env, agent, cl_args):
  env.reset()

  timestep = 0

  trace = []

  observation = env.get_observation()

  while not env.is_complete() and timestep <= cl_args.max_episode_length:
    action = agent.choose_action(observation)
    new_observation, reward = env.step(action)

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

  cl_args = parser.parse_args()
  env, agent = initialise(cl_args)
  train_agent(env, agent, cl_args)
  run_agent(env, agent, cl_args)