"""Environment functions for building Latent Shielded Dreamer Agent.

Taken from repository for 'Do Androids Dream of Electric Fences? Safe Reinforcement Learning with
Imagination-Based Agents' by Peter He."""

import numpy as np
import torch

from envs.env import init_env

# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 **
                                                         bit_depth).sub_(0.5)
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


# Wrapper for batching environments together
class EnvBatcher():
    def __init__(self, env_args, n):
        self.n = n
        self.envs = [init_env(env_args) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset().unsqueeze(0) for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

     # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions):
        # Done mask to blank out observations and zero rewards for previously terminated environments
        done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]
        observations, rewards, dones, infos = zip(
            *[env.step(action) for env, action in zip(self.envs, actions)])
        # Env should remain terminated if previously terminated
        dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]
        self.dones = dones
        violations = np.array(list(map(lambda x: x['violation'], infos)))
        observations, rewards, violations, dones = torch.cat([observation.unsqueeze(0) for observation in observations]), torch.tensor(
            rewards, dtype=torch.float32), torch.tensor(violations, dtype=torch.float32),  torch.tensor(dones, dtype=torch.bool)
        observations[done_mask] = 0
        rewards[done_mask] = 0
        violations[done_mask] = 0
        return observations, rewards, dones, {'violation': violations}

    def close(self):
        [env.close() for env in self.envs]
