# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab_tasks_experimental.direct.locomotion.locomotion_env_warp import LocomotionWarpEnv

from isaaclab_assets import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@wp.kernel
def compute_joint_pos_targets(
    input_actions: wp.array2d(dtype=wp.float32),
    default_joint_pos: wp.array2d(dtype=wp.float32),
    joint_pos_targets: wp.array2d(dtype=wp.float32),
    action_scale: wp.float32,
):
    """Maps normalized actions to joint position targets.

    Follows the manager-based velocity environments' control style:
    q_des = q_default + scale * clamp(a, -1, 1)
    """
    env_index, joint_index = wp.tid()
    joint_pos_targets[env_index, joint_index] = default_joint_pos[env_index, joint_index] + action_scale * wp.clamp(
        input_actions[env_index, joint_index], -1.0, 1.0
    )


@configclass
class UnitreeGo2WarpEnvCfg(DirectRLEnvCfg):
    """Direct locomotion task for Unitree Go2 (Warp backend).

    This environment reuses the generic locomotion logic from :class:`LocomotionWarpEnv`
    (same integration style as `direct/humanoid`).
    """

    # env
    episode_length_s = 20.0
    # Match the non-warp (manager-based) Go2 velocity configs: dt=1/200, decimation=4.
    decimation = 4
    # Match manager-based Go2: JointPositionActionCfg(..., scale=0.25, use_default_offset=True)
    action_scale = 0.25
    action_space = 12
    observation_space = 48  # 12 + 3 * num_dof (num_dof=12 for Go2)
    state_space = 0

    solver_cfg = MJWarpSolverCfg(
        # Keep consistent with `isaaclab_tasks/.../config/go2/flat_env_cfg.py`
        njmax=65,
        nconmax=35,
        ls_iterations=20,
        cone="pyramidal",
        impratio=1,
        ls_parallel=True,
        integrator="implicit",
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation, newton_cfg=newton_cfg)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Not used for position-control mapping, but required by base class.
    joint_gears: list[float] = [1.0] * 12

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.02
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 1.0
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 0.18

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.1


class UnitreeGo2WarpEnv(LocomotionWarpEnv):
    cfg: UnitreeGo2WarpEnvCfg

    def __init__(self, cfg: UnitreeGo2WarpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Position-control targets (q_des) in joint space.
        self.joint_pos_targets = wp.zeros(
            (self.num_envs, self.robot.num_joints), dtype=wp.float32, device=self.sim.device
        )

    def _pre_physics_step(self, actions: wp.array) -> None:
        """Map actions to joint position targets (manager-based style)."""
        # keep last actions (for obs / reward)
        self.actions.assign(actions)
        wp.launch(
            compute_joint_pos_targets,
            dim=(self.num_envs, self.robot.num_joints),
            inputs=[actions, self.robot.data.default_joint_pos, self.joint_pos_targets, wp.float32(self.cfg.action_scale)],
        )

    def _apply_action(self) -> None:
        """Apply joint position targets (PD/DC-motor in the articulation)."""
        self.robot.set_joint_position_target(self.joint_pos_targets, joint_mask=self._joint_dof_mask)

