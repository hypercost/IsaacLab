# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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


@configclass
class UnitreeGo2WarpEnvCfg(DirectRLEnvCfg):
    """Direct locomotion task for Unitree Go2 (Warp backend).

    This environment reuses the generic locomotion logic from :class:`LocomotionWarpEnv`
    (same integration style as `direct/humanoid`).
    """

    # env
    episode_length_s = 20.0
    decimation = 2
    action_scale = 0.5
    action_space = 12
    observation_space = 48  # 12 + 3 * num_dof (num_dof=12 for Go2)
    state_space = 0

    solver_cfg = MJWarpSolverCfg(
        njmax=64,
        nconmax=32,
        ls_iterations=10,
        ls_parallel=True,
        cone="pyramidal",
        impratio=1,
        update_data_interval=1,
    )
    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, newton_cfg=newton_cfg)
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
    # Torque scaling per joint (3 DOF per leg). Kept simple/uniform by default.
    joint_gears: list[float] = [23.5] * 12

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.02
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 1.0
    dof_vel_scale: float = 0.2

    death_cost: float = -2.0
    termination_height: float = 0.20

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.1


class UnitreeGo2WarpEnv(LocomotionWarpEnv):
    cfg: UnitreeGo2WarpEnvCfg

    def __init__(self, cfg: UnitreeGo2WarpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

