# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unitree Go2 velocity environments (warp-based).

Includes:
- Velocity tracking environment
- Velocity flat/direct environment
"""

import gymnasium as gym
from gymnasium.envs.registration import registry

from . import agents

##
# Register Gym environments.
##

# Environment IDs for velocity tracking
_VELOCITY_TRACKING_PRIMARY_ID = "Isaac-Velocity-Flat-Unitree-Go2-v0"
_VELOCITY_TRACKING_FALLBACK_ID = "Isaac-Velocity-Flat-Unitree-Go2-Velocity-Tracking-v0"

# Environment ID for velocity flat/direct
_VELOCITY_FLAT_ID = "Isaac-Velocity-Flat-Unitree-Go2-Direct-Warp-v0"


def _register_env(env_id: str, entry_point: str, env_cfg_entry_point: str, rsl_rl_cfg_entry_point: str = None) -> None:
    """Register a gym environment if not already registered."""
    if env_id in registry:
        return
    kwargs = {"env_cfg_entry_point": env_cfg_entry_point}
    if rsl_rl_cfg_entry_point:
        kwargs["rsl_rl_cfg_entry_point"] = rsl_rl_cfg_entry_point
    gym.register(
        id=env_id,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs=kwargs,
    )


# Register velocity tracking environment
# Prefer the user-requested id, but avoid collision with isaaclab_tasks' existing registration.
if _VELOCITY_TRACKING_PRIMARY_ID in registry:
    _register_env(
        _VELOCITY_TRACKING_FALLBACK_ID,
        f"{__name__}.go2_velocity_warp_env:UnitreeGo2VelocityWarpEnv",
        f"{__name__}.go2_velocity_warp_env:UnitreeGo2VelocityWarpEnvCfg",
    )
else:
    _register_env(
        _VELOCITY_TRACKING_PRIMARY_ID,
        f"{__name__}.go2_velocity_warp_env:UnitreeGo2VelocityWarpEnv",
        f"{__name__}.go2_velocity_warp_env:UnitreeGo2VelocityWarpEnvCfg",
    )

# Register velocity flat/direct environment
_register_env(
    _VELOCITY_FLAT_ID,
    f"{__name__}.go2_warp_env:UnitreeGo2WarpEnv",
    f"{__name__}.go2_warp_env:UnitreeGo2WarpEnvCfg",
    f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2DirectWarpPPORunnerCfg",
)
