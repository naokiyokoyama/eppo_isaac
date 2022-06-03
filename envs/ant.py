from typing import Tuple

import torch
from isaacgymenvs.tasks.ant import Ant


class AntV2(Ant):
    """Variant in which reward terms are stored in infos ('extras')"""

    orig_name = "Ant"

    def compute_reward(self, *args, **kwargs):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.extras["reward_terms"],
        ) = compute_ant_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
        )


@torch.jit.script
def compute_ant_reward(
    obs_buf: torch.Tensor,
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    actions: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    joints_at_limit_cost_scale: float,
    termination_height: float,
    death_cost: float,
    max_episode_length: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # reward from direction headed
        heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
        heading_reward = torch.where(
            obs_buf[:, 11] > 0.8,
            heading_weight_tensor,
            heading_weight * obs_buf[:, 11] / 0.8,
        )

        # aligning up axis of ant and environment
        up_reward = torch.zeros_like(heading_reward)
        up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

        # energy penalty for movement
        actions_cost = torch.sum(actions**2, dim=-1)
        electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
        dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

        # reward for duration of staying alive
        alive_reward = torch.ones_like(potentials) * 0.5
        progress_reward = potentials - prev_potentials

        """ Changes for reward terms start here """
        reward_terms = torch.stack(
            [
                progress_reward,
                alive_reward,
                up_reward,
                heading_reward,
                -actions_cost_scale * actions_cost,
                -energy_cost_scale * electricity_cost,
                -dof_at_limit_cost * joints_at_limit_cost_scale,
                torch.zeros_like(progress_reward),  # death cost
            ],
            dim=1,
        )

        # adjust reward for fallen agents
        num_terms = reward_terms.shape[1]
        terminated_agents = (
            (obs_buf[:, 0] < termination_height).unsqueeze(-1).repeat(1, num_terms)
        )
        death_terms = torch.cat(
            [
                torch.zeros_like(reward_terms[:, :-1]),
                torch.ones_like(reward_terms[:, -1:]) * death_cost,
            ],
            dim=1,
        )
        reward_terms = torch.where(terminated_agents, death_terms, reward_terms)

        total_reward = (
            progress_reward
            + alive_reward
            + up_reward
            + heading_reward
            - actions_cost_scale * actions_cost
            - energy_cost_scale * electricity_cost
            - dof_at_limit_cost * joints_at_limit_cost_scale
        )

        # adjust reward for fallen agents
        total_reward = torch.where(
            obs_buf[:, 0] < termination_height,
            torch.ones_like(total_reward) * death_cost,
            total_reward,
        )
        assert torch.allclose(total_reward, reward_terms.sum(1), 1e-6, 1e-6)
        """ End of changes for reward terms start here """

        # reset agents
        reset = torch.where(
            obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf
        )
        reset = torch.where(
            progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset
        )

    return total_reward, reset, reward_terms
