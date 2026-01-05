# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unitree Go2 velocity tracking (warp-based) environment.
"""

import gymnasium as gym
from gymnasium.envs.registration import registry

##
# Register Gym environments.
##

_PRIMARY_ID = "Isaac-Velocity-Flat-Unitree-Go2-v0"
_FALLBACK_ID = "Isaac-Velocity-Flat-Unitree-Go2-Direct-Warp-v0"


def _register_env(env_id: str) -> None:
    if env_id in registry:
        return
    gym.register(
        id=env_id,
        entry_point=f"{__name__}.go2_velocity_warp_env:UnitreeGo2VelocityWarpEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.go2_velocity_warp_env:UnitreeGo2VelocityWarpEnvCfg",
        },
    )


# Prefer the user-requested id, but avoid collision with isaaclab_tasks' existing registration.
if _PRIMARY_ID in registry:
    _register_env(_FALLBACK_ID)
else:
    _register_env(_PRIMARY_ID)

