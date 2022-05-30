import argparse
import datetime
import os
from typing import Dict, Sequence

import gym
import gym_minigrid # Required for minigrid env registrations
from agents.adapt.agent import AdaptDreamerAgent
from agents.adapt.algo import Dreamer
from rlpyt.envs.gym import GymEnvWrapper
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.collections import TrajInfo
from rlpyt.utils.logging.context import logger_context
import torch

from envs.rlpyt_env import OneHotAction

MAX_EPISODE_LENGTH = 100


def build_and_train(log_dir, run_id, args, eval=False, cuda_idx=None, save_model="last", load_model_path=None, n_parallel=2):
  params = torch.load(load_model_path) if load_model_path is not None else {}
  agent_state_dict = params.get("agent_state_dict")
  optimizer_state_dict = params.get("optimizer_state_dict")

  env_kwargs = dict(
    env=gym.make(
      "MiniGrid-UnsafeCrossingSimple-v0",
      max_steps=args.max_episode_length
    )
  )

  factory_method = make_wrapper(
    GymEnvWrapper,
    [OneHotAction],
    [dict()]
  )

  # sampler = GpuSampler(
  #   EnvCls=factory_method,
  #   TrajInfoCls=TrajInfo,
  #   env_kwargs=env_kwargs,
  #   eval_env_kwargs=env_kwargs,
  #   batch_T=50,
  #   batch_B=50,
  #   max_decorrelation_steps=1500,
  #   eval_n_envs=10,
  #   eval_max_steps=1000,
  #   eval_max_trajectories=5
  # )

  sampler = SerialSampler(
    EnvCls=factory_method,
    TrajInfoCls=TrajInfo,
    env_kwargs=env_kwargs,
    eval_env_kwargs=env_kwargs,
    batch_T=50,
    batch_B=50,
    max_decorrelation_steps=500,
    eval_n_envs=10,
    eval_max_steps=1000,
    eval_max_trajectories=5
  )

  algo = Dreamer(initial_optim_state_dict=optimizer_state_dict)
  agent = AdaptDreamerAgent(explore_decay=200000)
  runner_cls = MinibatchRlEval if eval else MinibatchRl
  runner = runner_cls(
    algo=algo,
    agent=agent,
    sampler=sampler,
    n_steps=200000,
    affinity=dict(cuda_idx=cuda_idx)
  )
  name = "dreamer"
  # with logger_context(log_dir, run_id, name, None, snapshot_mode=save_model, override_prefix=True, use_summary_writer=True):
  runner.train()


def make_wrapper(base_class, wrapper_classes: Sequence = None, wrapper_kwargs: Sequence[Dict] = None):
  """
  Creates the correct factory method with wrapper support.
  This would get passed as the EnvCls argument in the sampler.
  Examples:
  The following code would make a factory method for atari with action repeat 2
  ``factory_method = make(AtariEnv, (ActionRepeat, ), (dict(amount=2),))``
  :param base_class: the base environment class (eg. AtariEnv)
  :param wrapper_classes: list of wrapper classes in order inner-first, outer-last
  :param wrapper_kwargs: list of kwargs dictionaries passed to the wrapper classes
  :return: factory method
  """
  if wrapper_classes is None:
    def make_env(**env_kwargs):
      """:return only the base environment instance"""
      return base_class(**env_kwargs)

    return make_env
  else:
    assert len(wrapper_classes) == len(wrapper_kwargs)

    def make_env(**env_kwargs):
      """:return the wrapped environment instance"""
      env = base_class(**env_kwargs)
      for i, wrapper_cls in enumerate(wrapper_classes):
        w_kwargs = wrapper_kwargs[i]
        if w_kwargs is None:
          w_kwargs = dict()
        env = wrapper_cls(env, **w_kwargs)
      return env

    return make_env

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--run-id', help='run identifier (logging)', type=int, default=0)
  parser.add_argument('--cuda-idx', help='gpu to use ', type=int, default=None)
  parser.add_argument('--eval', action='store_true')
  parser.add_argument('--save-model', help='save model', type=str, default='last',
                      choices=['all', 'none', 'gap', 'last'])
  parser.add_argument('--load-model-path', help='load model from path', type=str)
  default_log_dir = os.path.join(os.path.join(__file__), "data", "local", datetime.datetime.now().strftime("%Y%m%d"))
  parser.add_argument("--log-dir", type=str, default=default_log_dir)
  parser.add_argument("--max-episode-length", type=int, default=MAX_EPISODE_LENGTH)

  args = parser.parse_args()
  log_dir = os.path.abspath(args.log_dir)
  i = args.run_id
  while os.path.exists(os.path.join(log_dir, "run_" + str(i))):
    print(f"run {i} already exists.")
    i += 1
  print(f"Run id: {i}")
  args.run_id = i
  build_and_train(log_dir, args.run_id, args, eval=args.eval, cuda_idx=args.cuda_idx, save_model=args.save_model, load_model_path=args.load_model_path)