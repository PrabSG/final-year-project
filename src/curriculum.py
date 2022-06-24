from typing import Optional

import numpy as np
import matplotlib as plt

class RootPDistShiftScheduler:
  def __init__(self, num_categories: int, t_grow: int, sharpness_delta: int,
               category_difficulties: np.ndarray, init_weights: Optional[np.ndarray] = None,
               final_weights: Optional[np.ndarray] = None):
    self._b = num_categories
    self._t_grow = t_grow
    self._delta = sharpness_delta
    self._ds = category_difficulties
    self._w0 = init_weights if init_weights is not None else np.ones(num_categories)
    self._wT = final_weights if final_weights is not None else np.ones(num_categories)
    # Calculated terms
    self._ps = self._delta ** self._ds
    self._lambda_0s = self._lambda_0()

  def _lambda_0(self):
    w0_sum = np.sum(self._w0)
    return self._w0 / (self._wT * w0_sum)


  def get_weightings(self, t: int):
    ones = np.ones(self._b)
    if t > self._t_grow: # Short circuit
      return self._wT * ones

    surd_term = (((1 - self._lambda_0s) ** self._ps) / self._t_grow) * t
    root = surd_term ** (1 / self._ps) # Inverting power becomes a root
    return self._wT * np.min([ones, root + self._lambda_0s], axis=0)

  def get_probabilities(self, t: int):
    weightings = self.get_weightings(t)
    return weightings / np.sum(weightings)

def plot_schedule(scheduler: RootPDistShiftScheduler, num_eq_eps, num_categories, labels, fname=None):
  plt.figure(figsize=(8, 5))
  plt.rcParams.update({'font.size': 14})
  wts = []
  for t in range(num_eq_eps + 10):
    probs = scheduler.get_probabilities(t)
    wts.append(probs)
  np_wts = np.array(wts)
  np_wts = np_wts.transpose()

  tot_wts = np.zeros(num_eq_eps + 10)
  for c in range(num_categories):
    plt.plot(np_wts[c] + tot_wts, color=cols[c][1], label=labels[c])
    plt.fill_between(range(num_eq_eps + 10), tot_wts, np_wts[c] + tot_wts, color=cols[c][0], alpha=0.5)
    tot_wts += np_wts[c]

  plt.xlabel("Episodes, t")
  plt.ylabel("Probability Distribution Function")
  plt.grid()
  plt.legend(loc="upper right")

  if fname is None:
    plt.savefig(f"cum_pdf_{scheduler._delta}_{'-'.join(map(str, scheduler._ds.tolist()))}_plot.pgf", format="pgf")
  else:
    plt.savefig(f"{fname}.pgf", format="pgf")
  plt.close()

def plot_dist(scheduler, num_eq_eps, num_categories, labels, fname=None):
  plt.figure(figsize=(8, 5))
  plt.rcParams.update({'font.size': 14})
  wts = []
  for t in range(num_eq_eps + 10):
    probs = scheduler.get_probabilities(t)
    wts.append(probs)
  np_wts = np.array(wts)
  np_wts = np_wts.transpose()

  for c in range(num_categories):
    plt.plot(np_wts[c], color=cols[c][1], label=labels[c])

  plt.xlabel("Episodes, t")
  plt.ylabel("Probability Density Function")
  plt.grid()
  plt.legend()

  if fname is None:
    plt.savefig(f"pdf_{scheduler._delta}_{'-'.join(map(str, scheduler._ds.tolist()))}_plot.pgf", format="pgf")
  else:
    plt.savefig(f"{fname}.pgf", format="pgf")
  plt.close()


if __name__ == "__main__":
  plt.rc('pgf', texsystem='pdflatex')

  curriculum_delta = 2
  curriculum_eq_eps = 100
  num_categories = 5

  curriculum_scheduler = RootPDistShiftScheduler(
    num_categories, curriculum_eq_eps, curriculum_delta,
    category_difficulties=np.array([1.0, 0.5, 0.0, -0.5, -1.0]),
    init_weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    final_weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
  )

  plot_schedule(curriculum_scheduler, curriculum_eq_eps, num_categories, labels=["d = 1", "d = 0.5", "d = 0", "d = -0.5", "d = -1"])
  plot_dist(curriculum_scheduler, curriculum_eq_eps, num_categories, labels=["d = 1", "d = 0.5", "d = 0", "d = -0.5", "d = -1"])

  curriculum_scheduler = RootPDistShiftScheduler(
    3, curriculum_eq_eps, 1.0,
    category_difficulties=np.array([1.0, 0.0, -1.0]),
    init_weights=np.array([3, 2, 1]),
    final_weights=np.array([1, 2, 3]),
  )
  plot_schedule(curriculum_scheduler, curriculum_eq_eps, 3, labels=["d = 1", "d = 0", "d = -1"], fname="lin_change")

  curriculum_scheduler = RootPDistShiftScheduler(
    3, curriculum_eq_eps, 2.0,
    category_difficulties=np.array([1.0, 0.0, -1.0]),
    init_weights=np.array([3, 2, 1]),
    final_weights=np.array([1, 2, 3]),
  )
  plot_schedule(curriculum_scheduler, curriculum_eq_eps, 3, labels=["d = 1", "d = 0", "d = -1"], fname="quad_change")

  curriculum_scheduler = RootPDistShiftScheduler(
    3, curriculum_eq_eps, 4.0,
    category_difficulties=np.array([1.0, 0.0, -1.0]),
    init_weights=np.array([3, 2, 1]),
    final_weights=np.array([1, 2, 3]),
  )
  plot_schedule(curriculum_scheduler, curriculum_eq_eps, 3, labels=["d = 1", "d = 0", "d = -1"], fname="high_ord_change")