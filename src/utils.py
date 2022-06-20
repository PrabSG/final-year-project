import argparse
import math
import pickle
from typing import Dict, List

from array2gif import write_gif
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from envs.env import BasicEnv, MiniGridEnvWrapper
from envs.safe_env import MiniGridSafetyEnv

cols = [('lightskyblue', 'deepskyblue'),  ('navajowhite', 'darkorange'), ('palegreen', 'seagreen'), ('mediumpurple', 'indigo'), ('lightcoral', 'firebrick'), ('lightgoldenrodyellow', 'gold'), ('whitesmoke', 'darkgrey'), ('pink', 'magenta')]

class DimensionError(IndexError):
  def __init__(self, expected, given, *args: object) -> None:
    self.message = f"Invalid dimensions. Expected {expected}, given {given}"
    super().__init__(self.message, *args)

def exp_decay_epsilon(eps_start, eps_end, eps_decay):
  def exp_eps_func(episode_num):
    return eps_end + (eps_start - eps_end) * \
        math.exp(-1. * episode_num / eps_decay)

  return exp_eps_func

def init_env(args):
  if args.env == "basic":
    return BasicEnv()
  elif args.env == "unsafe-simple":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingSimple-v0", seed=args.seed, max_steps=args.max_episode_length)
  elif args.env == "unsafe-micro":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingMicro-v0", seed=args.seed, max_steps=args.max_episode_length)
  elif args.env == "unsafe-small":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingN1-v0", seed=args.seed, max_steps=args.max_episode_length)
  elif args.env == "unsafe-med":
    return MiniGridEnvWrapper("MiniGrid-UnsafeCrossingN2-v0", seed=args.seed, max_steps=args.max_episode_length)
  elif args.env == "twopath":
    return MiniGridEnvWrapper("MiniGrid-TwoPathSimple-v0", seed=args.seed, max_steps=args.max_episode_length)
  elif args.env == "safety-simple":
    return MiniGridSafetyEnv("MiniGrid-UnsafeCrossingSimple-v0", seed=args.seed, max_steps=args.max_episode_length)
  elif args.env == "safety-micro":
    return MiniGridSafetyEnv("MiniGrid-UnsafeCrossingMicro-v0", seed=args.seed, max_steps=args.max_episode_length, curriculum_equality_episodes=args.curriculum_eq_eps)
  else:
    raise ValueError(f"Environment Type '{args.env}' not defined.")

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
    filename = "/vis_final.gif"
  else:
    filename = f"/vis_ep_{episode}.gif"

  print("Saving gif... ", end="")
  write_gif(np.array(frames), args.results_dir + filename, fps=1/0.1)
  print("Done.")

def plot_agent_variants(all_metrics: List[List[Dict]], variant_labels: List[str], fields: List[str], ylabels: Dict[str, str], save_dir: str):
  for field in fields:
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 16})

    for variant in range(len(variant_labels)):
      num_agents = len(all_metrics[variant])
      datapoints = []
      for i_agent in range(num_agents):
        data = all_metrics[variant][i_agent][field]
        datapoints.append(data)
      np_data = np.array(datapoints)
      data_mean = np.mean(np_data, axis=0)
      data_std = np.std(np_data, axis=0)

      # Clamp max_reward as 1
      upper_vals = np.min([np.ones(data_mean.shape), data_mean + data_std], axis=0) if "rewards" in field else data_mean + data_std

      plt.fill_between(range(len(data_mean)), upper_vals, data_mean - data_std, color=cols[variant][0], alpha=0.5)
      plt.plot(data_mean, color=cols[variant][1], label=variant_labels[variant])

    if "rewards" in field:
      plt.ylim(top=2.0)
    if "loss" in field:
      plt.ylim(bottom=0.0)
    if "kl" in field or "observation" in field:
      plt.ylim(top=100.0)

    plt.xlabel("Episodes")
    plt.ylabel(ylabels[field])
    plt.grid()
    if variant_labels[0] is not None:
      plt.legend()
    plt.savefig(save_dir + f"/{field}_plot.pgf", format="pgf")
    plt.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Agent and Environment options")
  # Plotting options
  parser.add_argument("--plot", action="store_true", help="Plot results into a pgf")
  parser.add_argument("--load-metrics", type=str, help="Pickle file location for metrics")
  parser.add_argument("--variant-metrics", type=str, help="Pickle file location of comparison metrics")
  parser.add_argument("--main-method-name", type=str)
  parser.add_argument("--variant-name", type=str)
  parser.add_argument("--results-dir", type=str, help="Location to save plots")

  args = parser.parse_args()

  if args.plot and args.load_metrics is not None and args.results_dir is not None:
    with open(args.load_metrics, "rb") as f:
      print("Loading metrics file...")
      agent_metrics = pickle.load(f)

    if args.variant_metrics is not None and args.main_method_name is not None and args.variant_name is not None:
      with open(args.variant_metrics, "rb") as vf:
        print("Loading variant metrics file...")
        variant_metrics = pickle.load(vf)

      ylabels = agent_metrics[0]["metric_titles"]
      main_fields = (agent_metrics[0].keys() - {"steps", "episodes", "metric_titles", "test_episodes", "test_rewards"})
      common_fields = main_fields.intersection(variant_metrics[0].keys())
      print("Plotting...")
      plot_agent_variants([agent_metrics, variant_metrics], [args.main_method_name, args.variant_name], common_fields, ylabels, args.results_dir)
    else:
      ylabels = agent_metrics[0]["metric_titles"]
      fields = list(agent_metrics[0].keys() - {"steps", "episodes", "metric_titles", "test_episodes", "test_rewards"})
      print("Plotting...")
      plot_agent_variants([agent_metrics], [None], fields, ylabels, args.results_dir)
