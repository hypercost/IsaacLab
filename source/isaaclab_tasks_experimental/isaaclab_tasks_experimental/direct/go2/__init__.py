# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Unitree Go2 velocity (flat) environment (Direct + Warp).
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Unitree-Go2-Direct-Warp-v0",
    entry_point=f"{__name__}.go2_warp_env:UnitreeGo2WarpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_warp_env:UnitreeGo2WarpEnvCfg",
        # Requested: use rsl-rl PPO for registration.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2DirectWarpPPORunnerCfg",
    },
)

