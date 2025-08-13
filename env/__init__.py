"""
SPEOS Environment Package

This package contains the SPEOS optical simulation environment for reinforcement learning.
"""

from gymnasium.envs.registration import register
from .env_speos_v1 import SpeosEnv

# Register the SPEOS environment
register(
    id='SpeosEnv-v1',
    entry_point='env.env_speos_v1:SpeosEnv',
    max_episode_steps=100,
    kwargs={},
)

__all__ = ['SpeosEnv']
