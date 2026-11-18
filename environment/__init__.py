from gymnasium.envs.registration import register

from environment.custom_env import (
    ACTION_TYPES,
    CHILD_FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    SCENARIOS,
    StuntingCareEnv,
)

ENV_ID = "StuntingCare-v0"

register(id=ENV_ID, entry_point="environment.custom_env:StuntingCareEnv")

__all__ = [
    "ACTION_TYPES",
    "CHILD_FEATURE_NAMES",
    "ENV_ID",
    "GLOBAL_FEATURE_NAMES",
    "SCENARIOS",
    "StuntingCareEnv",
]
