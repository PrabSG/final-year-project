import math
from re import A
import matplotlib.pyplot as plt

class DimensionError(IndexError):
  def __init__(self, expected, given, *args: object) -> None:
    self.message = f"Invalid dimensions. Expected {expected}, given {given}"
    super().__init__(self.message, *args)

def exp_decay_epsilon(eps_start, eps_end, eps_decay):
  def exp_eps_func(episode_num):
    return eps_end + (eps_start - eps_end) * \
        math.exp(-1. * episode_num / eps_decay)

  return exp_eps_func

def plot_training(n_episodes, episode_rs, n_steps, train_losses):
  fig, axs = plt.subplots(2, figsize=(8, 6))
  axs[0].plot(range(n_episodes), episode_rs, label="Episode Rewards")
  axs[0].set_title("Episode Rewards")
  axs[1].plot(range(n_steps), train_losses, label="Training Losses")
  axs[1].set_title("Training Losses")

  plt.show()
  plt.savefig('./training_plots_deterministic')
