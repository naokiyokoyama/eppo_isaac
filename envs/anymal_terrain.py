from isaacgym.torch_utils import *
from isaacgymenvs.tasks.anymal_terrain import AnymalTerrain


class AnymalTerrainV2(AnymalTerrain):
    """Variant in which reward terms are stored in infos ('extras')"""

    orig_name = "AnymalTerrain"

    def compute_reward(self):
        # velocity tracking reward
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = (
            torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        )
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        # other base velocity penalties
        rew_lin_vel_z = (
            torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        )
        rew_ang_vel_xy = (
            torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
            * self.rew_scales["ang_vel_xy"]
        )

        # orientation penalty
        rew_orient = (
            torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
            * self.rew_scales["orient"]
        )

        # base height penalty
        rew_base_height = (
            torch.square(self.root_states[:, 2] - 0.52) * self.rew_scales["base_height"]
        )  # TODO add target base height to cfg

        # torque penalty
        rew_torque = (
            torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]
        )

        # joint acc penalty
        rew_joint_acc = (
            torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1)
            * self.rew_scales["joint_acc"]
        )

        # collision penalty
        knee_contact = (
            torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.0
        )
        rew_collision = (
            torch.sum(knee_contact, dim=1) * self.rew_scales["collision"]
        )  # sum vs any ?

        # stumbling penalty
        stumble = (
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.0
        ) * (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < 1.0)
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"]

        # action rate penalty
        rew_action_rate = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=1)
            * self.rew_scales["action_rate"]
        )

        # air time reward
        # contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        first_contact = (self.feet_air_time > 0.0) * contact
        self.feet_air_time += self.dt
        rew_airTime = (
            torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
            * self.rew_scales["air_time"]
        )  # reward only on first contact with the ground
        rew_airTime *= (
            torch.norm(self.commands[:, :2], dim=1) > 0.1
        )  # no reward for zero command
        self.feet_air_time *= ~contact

        # cosmetic penalty for hip motion
        rew_hip = (
            torch.sum(
                torch.abs(
                    self.dof_pos[:, [0, 3, 6, 9]]
                    - self.default_dof_pos[:, [0, 3, 6, 9]]
                ),
                dim=1,
            )
            * self.rew_scales["hip"]
        )

        # total reward
        self.rew_buf = (
            rew_lin_vel_xy
            + rew_ang_vel_z
            + rew_lin_vel_z
            + rew_ang_vel_xy
            + rew_orient
            + rew_base_height
            + rew_torque
            + rew_joint_acc
            + rew_collision
            + rew_action_rate
            + rew_airTime
            + rew_hip
            + rew_stumble
        )
        self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None)

        # add termination reward
        self.rew_buf += (
            self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf
        )

        """ Changes for reward terms start here """
        reward_terms = torch.stack(
            [
                rew_lin_vel_xy,
                rew_ang_vel_z,
                rew_lin_vel_z,
                rew_ang_vel_xy,
                rew_orient,
                rew_base_height,
                rew_torque,
                rew_joint_acc,
                rew_collision,
                rew_action_rate,
                rew_airTime,
                rew_hip,
                rew_stumble,
                self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf,
            ],
            dim=1,
        )
        self.extras["reward_terms"] = reward_terms
        """ End of changes for reward terms start here """

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["collision"] += rew_collision
        self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["air_time"] += rew_airTime
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["hip"] += rew_hip
