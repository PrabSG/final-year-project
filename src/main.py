from agent import RandomAgent
from env import BasicEnv


MAX_TIME_STEPS = 100

def run_agent():
  env = BasicEnv()
  agent = RandomAgent(env.state_size, env.action_size, env.action_value_range)

  env.reset()

  timestep = 0

  trace = []

  observation = env.get_observation()

  while not env.is_complete() and timestep <= MAX_TIME_STEPS:
    action = agent.choose_action(observation)
    new_observation, reward = env.step(action)

    trace.append((observation, action, reward, new_observation))

    timestep += 1
    observation = new_observation

  for s, a, r, new_s in trace:
    print("State:", s, "Action:", a, "Reward:", r, "New State:", new_s)



if __name__ == "__main__":
  run_agent()