"""
Vector Quantized Hierarchical Clustered Soft Actor-Critic (VQ-HC-SAC).

A novel multi-agent RL framework that:
1. Dynamically clusters N agents into M roles using Vector Quantization
2. Trains M shared role-specific policies with Soft Actor-Critic
3. Uses PBRS for dense rewards and STE for end-to-end training
"""

from .config import VQHCSACConfig
from .trainer import VQHCSACTrainer
from .agent import VQHCSACAgent

__all__ = [
    'VQHCSACConfig',
    'VQHCSACTrainer',
    'VQHCSACAgent',
]
