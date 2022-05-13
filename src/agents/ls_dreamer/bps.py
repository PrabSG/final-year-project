"""Bounded Prescience Shield for building Latent Shielded Dreamer Agent.

Taken from repository for 'Do Androids Dream of Electric Fences? Safe Reinforcement Learning with
Imagination-Based Agents' by Peter He."""

import torch
from torch.distributions import Normal
import numpy as np

from agents.ls_dreamer.models import TransitionModel, ViolationModel
from agents.ls_dreamer.shield import Shield


class BoundedPrescienceShield(Shield):
    def __init__(
        self,
        transition_model: TransitionModel,
        violation_model: ViolationModel,
        depth=1,
        violation_threshold=3,
        paths_to_sample=40,
    ):
        self.transition_model = transition_model
        self.violation_model = violation_model
        self.depth = depth
        self.violation_threshold = violation_threshold
        self.paths_to_sample = paths_to_sample

    def step(self, belief, state, action, decoder, policy, observation, encoder):
        violation_occurred = False
        fallback_action_idxs = np.random.permutation(
            [i for i in range(0, len(action.squeeze()))]
        )
        for i in range(0, len(action) + 1):
            futures, violations = self.imagine_futures(belief, state, action, policy)
            if sum(violations) > self.violation_threshold:
                # Try all the actions
                action = torch.tensor(
                    [
                        [
                            1.0 if j == fallback_action_idxs[i] else 0.0
                            for j in range(len(action.squeeze()))
                        ]
                    ]
                )
                action = action.cuda() if torch.cuda.is_available() else action
                # for traj in futures:
                #     print("NEW")
                #     for belief, state in traj:
                #         pred_ob = decoder(belief.squeeze().unsqueeze(0), state.squeeze().unsqueeze(0)).squeeze().permute(1,2,0).cpu()
                #         import matplotlib.pyplot as plt
                #         plt.imshow(pred_ob)
                #         plt.show()
                violation_occurred = True
            else:
                break

        return action, violation_occurred

    def imagine_futures(
        self, belief, state, action, policy
    ):
        futures = []
        violations = []
        B, H, Z = belief.size(0), belief.size(1), state.size(1)
        for i in range(self.paths_to_sample):
            next_belief, next_state, _, _ = self.transition_model(
                state, action.unsqueeze(0), belief
            )
            violation = torch.argmax(
                self.violation_model(
                    next_belief.view(-1, H), next_state.view(-1, Z)
                ).squeeze()
            ).item()
            traj = [(next_belief, next_state)]
            for step in range(self.depth - 1):
                action = policy.get_action(
                    next_belief.view(-1, H), next_state.view(-1, Z)
                )
                action = torch.clamp(Normal(action.float(), 0.3).rsample(), -1, 1)
                next_belief, next_state, _, _ = self.transition_model(
                    next_state.view(-1, Z), action.unsqueeze(0), next_belief.view(-1, H)
                )
                if violation == 0:
                    violation = torch.argmax(
                        self.violation_model(
                            next_belief.view(-1, H), next_state.view(-1, Z)
                        ).squeeze()
                    ).item()
                traj.append((next_belief, next_state))
            violations.append(violation)
            futures.append(traj)
        return futures, violations


class ShieldBatcher(Shield):
    def __init__(
        self,
        shield_class,
        envs,
        transition_model,
        violation_model,
        depth=3,
        violation_threshold=3,
        paths_to_sample=40,
    ):
        self._shields = [
            shield_class(
                transition_model,
                violation_model,
                depth,
                violation_threshold=violation_threshold,
                paths_to_sample=paths_to_sample,
            )
            for i in range(len(envs))
        ]

    def step(self, beliefs, states, actions, decoder, policy, observations, encoder):
        safe_actions = []
        had_to_interfere = [0 for i in range(len(actions))]
        for i, action in enumerate(actions.split(1)):
            safe_action, interfered = self._shields[i].step(
                beliefs[i].unsqueeze(0),
                states[i].unsqueeze(0),
                action,
                decoder,
                policy,
                observations[i],
                encoder,
            )
            safe_actions.append(safe_action.cpu())
            had_to_interfere[i] = 1 if interfered else 0
        return torch.cat(safe_actions, 0), had_to_interfere
