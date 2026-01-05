# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
from isaaclab_experimental.envs import DirectRLEnvWarp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@wp.func
def spatial_rotate_inv(quat: wp.quatf, vec: wp.spatial_vectorf) -> wp.spatial_vectorf:
    """Rotate spatial vector from world into body frame.

    Convention used in existing warp tasks:
    - linear velocity is stored in indices [0:3]
    - angular velocity is stored in indices [3:6]
    """
    return wp.spatial_vector(
        wp.quat_rotate_inv(quat, wp.spatial_top(vec)),
        wp.quat_rotate_inv(quat, wp.spatial_bottom(vec)),
    )


@wp.func
def unscale(x: wp.float32, lower: wp.float32, upper: wp.float32) -> wp.float32:
    return (2.0 * x - upper - lower) / (upper - lower)


@wp.kernel
def initialize_state(
    state: wp.array(dtype=wp.uint32),
    seed: wp.int32,
):
    env_index = wp.tid()
    state[env_index] = wp.rand_init(seed, env_index)


@wp.kernel
def sample_velocity_commands(
    commands: wp.array(dtype=wp.vec3f),
    cmd_lin_x_range: wp.vec2f,
    cmd_lin_y_range: wp.vec2f,
    cmd_ang_z_range: wp.vec2f,
    env_mask: wp.array(dtype=wp.bool),
    state: wp.array(dtype=wp.uint32),
):
    env_index = wp.tid()
    if env_mask[env_index]:
        cmd_x = wp.randf(state[env_index], cmd_lin_x_range[0], cmd_lin_x_range[1])
        cmd_y = wp.randf(state[env_index], cmd_lin_y_range[0], cmd_lin_y_range[1])
        cmd_wz = wp.randf(state[env_index], cmd_ang_z_range[0], cmd_ang_z_range[1])
        commands[env_index] = wp.vec3f(cmd_x, cmd_y, cmd_wz)
        state[env_index] += wp.uint32(1)


@wp.kernel
def update_actions(
    input_actions: wp.array2d(dtype=wp.float32),
    actions: wp.array2d(dtype=wp.float32),
    joint_gears: wp.array(dtype=wp.float32),
    action_scale: wp.float32,
):
    env_index, joint_index = wp.tid()
    actions[env_index, joint_index] = (
        action_scale * joint_gears[joint_index] * wp.clamp(input_actions[env_index, joint_index], -1.0, 1.0)
    )


@wp.kernel
def compute_body_frame_velocity(
    root_pose_w: wp.array(dtype=wp.transformf),
    root_vel_w: wp.array(dtype=wp.spatial_vectorf),
    root_vel_b: wp.array(dtype=wp.spatial_vectorf),
):
    env_index = wp.tid()
    quat = wp.transform_get_rotation(root_pose_w[env_index])
    root_vel_b[env_index] = spatial_rotate_inv(quat, root_vel_w[env_index])


@wp.kernel
def compute_projected_gravity(
    root_pose_w: wp.array(dtype=wp.transformf),
    projected_gravity_b: wp.array(dtype=wp.vec3f),
):
    env_index = wp.tid()
    quat = wp.transform_get_rotation(root_pose_w[env_index])
    # world gravity direction in world frame
    g_w = wp.vec3f(0.0, 0.0, -1.0)
    # express in body frame
    projected_gravity_b[env_index] = wp.quat_rotate_inv(quat, g_w)


@wp.kernel
def scale_dof_pos(
    dof_pos: wp.array2d(dtype=wp.float32),
    dof_limits: wp.array2d(dtype=wp.vec2f),
    dof_pos_scaled: wp.array2d(dtype=wp.float32),
):
    env_index, joint_index = wp.tid()
    dof_pos_scaled[env_index, joint_index] = unscale(
        dof_pos[env_index, joint_index], dof_limits[env_index, joint_index][0], dof_limits[env_index, joint_index][1]
    )


@wp.func
def actions_cost(actions: wp.array(dtype=wp.float32)) -> wp.float32:
    sum_ = wp.float32(0.0)
    for i in range(len(actions)):
        sum_ += actions[i] * actions[i]
    return sum_


@wp.func
def dof_at_limit_cost(dof_pos_scaled: wp.array(dtype=wp.float32)) -> wp.float32:
    sum_ = wp.float32(0.0)
    for i in range(len(dof_pos_scaled)):
        if dof_pos_scaled[i] > 0.98:
            sum_ += 1.0
    return sum_


@wp.kernel
def get_dones(
    episode_length_buf: wp.array(dtype=wp.int32),
    root_pose_w: wp.array(dtype=wp.transformf),
    max_episode_length: wp.int32,
    termination_height: wp.float32,
    out_of_bounds: wp.array(dtype=wp.bool),
    time_out: wp.array(dtype=wp.bool),
    reset: wp.array(dtype=wp.bool),
):
    env_index = wp.tid()
    height = wp.transform_get_translation(root_pose_w[env_index])[2]
    out_of_bounds[env_index] = height < termination_height
    time_out[env_index] = episode_length_buf[env_index] >= (max_episode_length - 1)
    reset[env_index] = out_of_bounds[env_index] or time_out[env_index]


@wp.kernel
def compute_rewards(
    commands: wp.array(dtype=wp.vec3f),
    root_vel_b: wp.array(dtype=wp.spatial_vectorf),
    actions: wp.array2d(dtype=wp.float32),
    dof_pos_scaled: wp.array2d(dtype=wp.float32),
    reset_terminated: wp.array(dtype=wp.bool),
    lin_vel_std: wp.float32,
    ang_vel_std: wp.float32,
    lin_vel_reward_scale: wp.float32,
    ang_vel_reward_scale: wp.float32,
    action_cost_scale: wp.float32,
    dof_limit_cost_scale: wp.float32,
    alive_reward: wp.float32,
    death_cost: wp.float32,
    reward: wp.array(dtype=wp.float32),
):
    env_index = wp.tid()
    if reset_terminated[env_index]:
        reward[env_index] = death_cost
        return

    # tracking errors in body frame
    vx = root_vel_b[env_index][0]
    vy = root_vel_b[env_index][1]
    wz = root_vel_b[env_index][5]

    cmd = commands[env_index]
    err_lin = (vx - cmd[0]) * (vx - cmd[0]) + (vy - cmd[1]) * (vy - cmd[1])
    err_ang = (wz - cmd[2]) * (wz - cmd[2])

    rew_lin = wp.exp(-err_lin / (lin_vel_std * lin_vel_std))
    rew_ang = wp.exp(-err_ang / (ang_vel_std * ang_vel_std))

    reward[env_index] = (
        alive_reward
        + lin_vel_reward_scale * rew_lin
        + ang_vel_reward_scale * rew_ang
        - action_cost_scale * actions_cost(actions[env_index])
        - dof_limit_cost_scale * dof_at_limit_cost(dof_pos_scaled[env_index])
    )


@wp.kernel
def get_observations(
    commands: wp.array(dtype=wp.vec3f),
    root_vel_b: wp.array(dtype=wp.spatial_vectorf),
    projected_gravity_b: wp.array(dtype=wp.vec3f),
    dof_pos_scaled: wp.array2d(dtype=wp.float32),
    dof_vel: wp.array2d(dtype=wp.float32),
    actions_in: wp.array2d(dtype=wp.float32),
    observations: wp.array2d(dtype=wp.float32),
    dof_vel_scale: wp.float32,
    num_dof: wp.int32,
):
    env_index = wp.tid()

    # commands
    observations[env_index, 0] = commands[env_index][0]
    observations[env_index, 1] = commands[env_index][1]
    observations[env_index, 2] = commands[env_index][2]

    # base velocity in body frame (linear xyz, angular xyz)
    observations[env_index, 3] = root_vel_b[env_index][0]
    observations[env_index, 4] = root_vel_b[env_index][1]
    observations[env_index, 5] = root_vel_b[env_index][2]
    observations[env_index, 6] = root_vel_b[env_index][3]
    observations[env_index, 7] = root_vel_b[env_index][4]
    observations[env_index, 8] = root_vel_b[env_index][5]

    # projected gravity in body frame
    observations[env_index, 9] = projected_gravity_b[env_index][0]
    observations[env_index, 10] = projected_gravity_b[env_index][1]
    observations[env_index, 11] = projected_gravity_b[env_index][2]

    offset_1 = 12 + num_dof
    offset_2 = offset_1 + num_dof

    for i in range(num_dof):
        observations[env_index, 12 + i] = dof_pos_scaled[env_index, i]
    for i in range(num_dof):
        observations[env_index, offset_1 + i] = dof_vel[env_index, i] * dof_vel_scale
    for i in range(num_dof):
        observations[env_index, offset_2 + i] = actions_in[env_index, i]


@wp.func
def translate_transform(transform: wp.transformf, translation: wp.vec3f) -> wp.transformf:
    return wp.transform(
        wp.transform_get_translation(transform) + translation,
        wp.transform_get_rotation(transform),
    )


@wp.kernel
def reset_root(
    default_root_pose: wp.array(dtype=wp.transformf),
    default_root_vel: wp.array(dtype=wp.spatial_vectorf),
    env_origins: wp.array(dtype=wp.vec3f),
    root_pose: wp.array(dtype=wp.transformf),
    root_vel: wp.array(dtype=wp.spatial_vectorf),
    env_mask: wp.array(dtype=wp.bool),
):
    env_index = wp.tid()
    if env_mask[env_index]:
        root_pose[env_index] = default_root_pose[env_index]
        root_pose[env_index] = translate_transform(root_pose[env_index], env_origins[env_index])
        root_vel[env_index] = default_root_vel[env_index]


@wp.kernel
def reset_joints(
    default_joint_pos: wp.array2d(dtype=wp.float32),
    default_joint_vel: wp.array2d(dtype=wp.float32),
    joint_pos: wp.array2d(dtype=wp.float32),
    joint_vel: wp.array2d(dtype=wp.float32),
    env_mask: wp.array(dtype=wp.bool),
):
    env_index, joint_index = wp.tid()
    if env_mask[env_index]:
        joint_pos[env_index, joint_index] = default_joint_pos[env_index, joint_index]
        joint_vel[env_index, joint_index] = default_joint_vel[env_index, joint_index]


@configclass
class UnitreeGo2VelocityWarpEnvCfg(DirectRLEnvCfg):
    """Warp-based velocity tracking task for Unitree Go2 (Newton + Warp kernels)."""

    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 23.5  # matches motor effort limit
    action_space = 12
    # obs: 12 + dof_pos(12) + dof_vel(12) + last_actions(12) = 48
    observation_space = 48
    state_space = 0

    # newton solver (warp-based)
    solver_cfg = MJWarpSolverCfg(
        njmax=65,
        nconmax=35,
        ls_iterations=20,
        cone="pyramidal",
        impratio=1,
        ls_parallel=True,
        integrator="implicit",
        update_data_interval=1,
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
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list[float] = [1.0] * 12

    # commands (body frame)
    cmd_lin_x_range = (-1.0, 1.0)
    cmd_lin_y_range = (-0.5, 0.5)
    cmd_ang_z_range = (-1.0, 1.0)

    # rewards
    lin_vel_std: float = 0.25
    ang_vel_std: float = 0.25
    lin_vel_reward_scale: float = 1.0
    ang_vel_reward_scale: float = 0.5
    alive_reward: float = 0.2
    death_cost: float = -2.0

    action_cost_scale: float = 0.002
    dof_limit_cost_scale: float = 0.01
    dof_vel_scale: float = 0.05

    # terminations
    termination_height: float = 0.18


class UnitreeGo2VelocityWarpEnv(DirectRLEnvWarp):
    cfg: UnitreeGo2VelocityWarpEnvCfg

    def __init__(self, cfg: UnitreeGo2VelocityWarpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.joint_gears = wp.array(self.cfg.joint_gears, dtype=wp.float32, device=self.sim.device)
        self._joint_dof_mask, _, self._joint_dof_idx = self.robot.find_joints(".*")

        # bindings (direct views into Newton buffers)
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.root_pose_w = self.robot.data.root_pose_w
        self.root_vel_w = self.robot.data.root_vel_w
        self.soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits

        # buffers
        self.observations = wp.zeros((self.num_envs, self.cfg.observation_space), dtype=wp.float32, device=self.sim.device)
        self.rewards = wp.zeros((self.num_envs), dtype=wp.float32, device=self.sim.device)
        self.actions = wp.zeros((self.num_envs, self.robot.num_joints), dtype=wp.float32, device=self.sim.device)
        self.actions_mapped = wp.zeros((self.num_envs, self.robot.num_joints), dtype=wp.float32, device=self.sim.device)
        self.dof_pos_scaled = wp.zeros((self.num_envs, self.robot.num_joints), dtype=wp.float32, device=self.sim.device)

        self.root_vel_b = wp.zeros((self.num_envs), dtype=wp.spatial_vectorf, device=self.sim.device)
        self.projected_gravity_b = wp.zeros((self.num_envs), dtype=wp.vec3f, device=self.sim.device)
        self.commands = wp.zeros((self.num_envs), dtype=wp.vec3f, device=self.sim.device)
        self.states = wp.zeros((self.num_envs), dtype=wp.uint32, device=self.sim.device)
        self.env_origins = wp.from_torch(self.scene.env_origins, dtype=wp.vec3f)

        # init RNG
        if self.cfg.seed is None:
            self.cfg.seed = -1
        wp.launch(initialize_state, dim=self.num_envs, inputs=[self.states, self.cfg.seed])

        # sample initial commands for all envs
        wp.launch(
            sample_velocity_commands,
            dim=self.num_envs,
            inputs=[
                self.commands,
                wp.vec2f(self.cfg.cmd_lin_x_range[0], self.cfg.cmd_lin_x_range[1]),
                wp.vec2f(self.cfg.cmd_lin_y_range[0], self.cfg.cmd_lin_y_range[1]),
                wp.vec2f(self.cfg.cmd_ang_z_range[0], self.cfg.cmd_ang_z_range[1]),
                self._ALL_ENV_MASK,
                self.states,
            ],
        )

        # torch views
        self.torch_obs_buf = wp.to_torch(self.observations)
        self.torch_reward_buf = wp.to_torch(self.rewards)
        self.torch_reset_terminated = wp.to_torch(self.reset_terminated)
        self.torch_reset_time_outs = wp.to_torch(self.reset_time_outs)
        self.torch_episode_length_buf = wp.to_torch(self.episode_length_buf)

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: wp.array) -> None:
        # keep last actions (for obs)
        self.actions.assign(actions)
        wp.launch(
            update_actions,
            dim=(self.num_envs, self.robot.num_joints),
            inputs=[actions, self.actions_mapped, self.joint_gears, self.action_scale],
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_effort_target(self.actions_mapped, joint_mask=self._joint_dof_mask)

    def _compute_intermediate(self) -> None:
        wp.launch(compute_body_frame_velocity, dim=self.num_envs, inputs=[self.root_pose_w, self.root_vel_w, self.root_vel_b])
        wp.launch(compute_projected_gravity, dim=self.num_envs, inputs=[self.root_pose_w, self.projected_gravity_b])
        wp.launch(
            scale_dof_pos,
            dim=(self.num_envs, self.robot.num_joints),
            inputs=[self.joint_pos, self.soft_joint_pos_limits, self.dof_pos_scaled],
        )

    def _get_observations(self) -> None:
        self._compute_intermediate()
        wp.launch(
            get_observations,
            dim=self.num_envs,
            inputs=[
                self.commands,
                self.root_vel_b,
                self.projected_gravity_b,
                self.dof_pos_scaled,
                self.joint_vel,
                self.actions,
                self.observations,
                self.cfg.dof_vel_scale,
                self.robot.num_joints,
            ],
        )

    def _get_rewards(self) -> None:
        # assumes intermediate already computed by _get_dones()
        wp.launch(
            compute_rewards,
            dim=self.num_envs,
            inputs=[
                self.commands,
                self.root_vel_b,
                self.actions,
                self.dof_pos_scaled,
                self.reset_terminated,
                self.cfg.lin_vel_std,
                self.cfg.ang_vel_std,
                self.cfg.lin_vel_reward_scale,
                self.cfg.ang_vel_reward_scale,
                self.cfg.action_cost_scale,
                self.cfg.dof_limit_cost_scale,
                self.cfg.alive_reward,
                self.cfg.death_cost,
                self.rewards,
            ],
        )

    def _get_dones(self) -> None:
        self._compute_intermediate()
        wp.launch(
            get_dones,
            dim=self.num_envs,
            inputs=[
                self.episode_length_buf,
                self.root_pose_w,
                self.max_episode_length,
                self.cfg.termination_height,
                self.reset_terminated,
                self.reset_time_outs,
                self.reset_buf,
            ],
        )

    def _reset_idx(self, mask: wp.array | None = None):
        if mask is None:
            mask = self.robot._ALL_ENV_MASK

        super()._reset_idx(mask)

        # reset root and joints
        wp.launch(
            reset_root,
            dim=self.num_envs,
            inputs=[
                self.robot.data.default_root_pose,
                self.robot.data.default_root_vel,
                self.env_origins,
                self.root_pose_w,
                self.root_vel_w,
                mask,
            ],
        )
        wp.launch(
            reset_joints,
            dim=(self.num_envs, self.robot.num_joints),
            inputs=[
                self.robot.data.default_joint_pos,
                self.robot.data.default_joint_vel,
                self.joint_pos,
                self.joint_vel,
                mask,
            ],
        )

        # resample commands for reset envs
        wp.launch(
            sample_velocity_commands,
            dim=self.num_envs,
            inputs=[
                self.commands,
                wp.vec2f(self.cfg.cmd_lin_x_range[0], self.cfg.cmd_lin_x_range[1]),
                wp.vec2f(self.cfg.cmd_lin_y_range[0], self.cfg.cmd_lin_y_range[1]),
                wp.vec2f(self.cfg.cmd_ang_z_range[0], self.cfg.cmd_ang_z_range[1]),
                mask,
                self.states,
            ],
        )

        self._compute_intermediate()

