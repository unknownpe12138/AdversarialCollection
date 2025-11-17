"""
Core components for VQ-HC-SAC.
"""

from .role_manager import RoleManager
from .replay_buffer import MultiAgentReplayBuffer, PrioritizedReplayBuffer
from .temperature import TemperatureManager, FixedTemperatureManager

__all__ = [
    'RoleManager',
    'MultiAgentReplayBuffer',
    'PrioritizedReplayBuffer',
    'TemperatureManager',
    'FixedTemperatureManager',
]
