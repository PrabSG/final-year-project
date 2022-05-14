import math

from array2gif import write_gif
import matplotlib.pyplot as plt
import numpy as np

class DimensionError(IndexError):
  def __init__(self, expected, given, *args: object) -> None:
    self.message = f"Invalid dimensions. Expected {expected}, given {given}"
    super().__init__(self.message, *args)

def exp_decay_epsilon(eps_start, eps_end, eps_decay):
  def exp_eps_func(episode_num):
    return eps_end + (eps_start - eps_end) * \
        math.exp(-1. * episode_num / eps_decay)

  return exp_eps_func

def plot_training(n_episodes, episode_rs, n_steps, train_losses, filename):
  fig, axs = plt.subplots(2, figsize=(8, 6))
  axs[0].plot(range(n_episodes), episode_rs, label="Episode Rewards")
  axs[0].set_title("Episode Rewards")
  axs[1].plot(range(n_steps), train_losses, label="Training Losses")
  axs[1].set_title("Training Losses")

  plt.show()
  plt.savefig(filename)

def visualise_agent(env, agent, args, episode=None):
  """Run agent in environment and visualise agent's path."""
  _, frames = agent.run_tests(args.vis_eps, env, args, visualise=True, episode=episode)

  if episode is None:
    filename = "vis_final.gif"
  else:
    filename = f"vis_ep_{episode}.gif"

  print("Saving gif... ", end="")
  write_gif(np.array(frames), args.results_dir + filename, fps=1/0.1)
  print("Done.")