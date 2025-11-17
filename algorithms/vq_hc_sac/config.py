"""
Configuration for VQ-HC-SAC algorithm.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VQHCSACConfig:
    """
    Hyperparameter configuration for VQ-HC-SAC.
    
    Algorithm Components:
    - VQ-Role: Dynamic role assignment via vector quantization
    - Hierarchical SAC: M role-specific Actor-Critic networks
    - PBRS: Potential-based reward shaping (handled by environment)
    """
    
    # ==================== Role Configuration ====================
    n_roles: int = 3
    """Number of roles M (M << N agents)"""
    
    embedding_dim: int = 64
    """Dimension D of role embeddings z_k ∈ R^D"""
    
    # ==================== Network Architecture ====================
    # Encoder: s_k → z_k
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    """Hidden layer sizes for state encoder E_φ"""
    
    # Actor: π_c(a|s_k)
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    """Hidden layer sizes for role actors {π_c}"""
    
    # Critic: Q_c(s_k, a_k, s_env, {s̄_j})
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    """Hidden layer sizes for role critics {Q_c}"""
    
    # ==================== SAC Hyperparameters ====================
    gamma: float = 0.99
    """Discount factor γ"""
    
    tau: float = 0.005
    """Soft update coefficient for target networks"""
    
    actor_lr: float = 3e-4
    """Learning rate for actors"""
    
    critic_lr: float = 3e-4
    """Learning rate for critics"""
    
    alpha_lr: float = 3e-4
    """Learning rate for temperature parameters"""
    
    initial_alpha: float = 0.2
    """Initial value for temperature α"""
    
    target_entropy_scale: float = -1.0
    """Scale for target entropy: H_target = scale * action_dim"""
    
    # ==================== VQ-VAE Parameters ====================
    vq_beta: float = 0.25
    """Commitment loss weight β in L_commit = β·||z - sg(e)||^2"""
    
    encoder_lr: float = 3e-4
    """Learning rate for state encoder E_φ"""
    
    codebook_lr: float = 1e-3
    """Learning rate for role codebook E"""
    
    use_ema_codebook: bool = False
    """Whether to use EMA updates for codebook (alternative to gradient)"""
    
    codebook_ema_decay: float = 0.99
    """EMA decay for codebook updates (if use_ema_codebook=True)"""
    
    # ==================== Training Configuration ====================
    buffer_size: int = 1_000_000
    """Replay buffer capacity"""
    
    batch_size: int = 256
    """Batch size for network updates"""
    
    warmup_steps: int = 10_000
    """Random exploration steps before training"""
    
    update_frequency: int = 1
    """Update networks every N environment steps"""
    
    updates_per_step: int = 1
    """Number of gradient updates per environment step"""
    
    target_update_frequency: int = 1
    """Update target networks every N training steps"""
    
    # ==================== Exploration ====================
    exploration_noise: float = 0.1
    """Noise scale for exploration during training"""
    
    # ==================== Logging & Evaluation ====================
    log_interval: int = 1000
    """Log training metrics every N steps"""
    
    eval_interval: int = 10_000
    """Evaluate agent every N steps"""
    
    eval_episodes: int = 5
    """Number of episodes for evaluation"""
    
    save_interval: int = 50_000
    """Save checkpoint every N steps"""
    
    # ==================== Device Configuration ====================
    device: str = 'cuda'
    """Device for training: 'cuda' or 'cpu'"""
    
    seed: Optional[int] = None
    """Random seed for reproducibility"""
    
    # ==================== Advanced Options ====================
    use_double_critic: bool = True
    """Use twin critics to reduce overestimation (SAC standard)"""
    
    clip_grad_norm: Optional[float] = 10.0
    """Gradient clipping norm (None to disable)"""
    
    role_aggregation_method: str = 'mean'
    """Method for role state aggregation: 'mean', 'max', 'sum'"""
    
    normalize_observations: bool = True
    """Whether to normalize observations"""
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_roles > 0, "n_roles must be positive"
        assert self.embedding_dim > 0, "embedding_dim must be positive"
        assert 0 < self.gamma <= 1, "gamma must be in (0, 1]"
        assert 0 < self.tau <= 1, "tau must be in (0, 1]"
        assert self.vq_beta >= 0, "vq_beta must be non-negative"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.role_aggregation_method in ['mean', 'max', 'sum'], \
            "role_aggregation_method must be 'mean', 'max', or 'sum'"
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            key: getattr(self, key) 
            for key in self.__dataclass_fields__.keys()
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'VQHCSACConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
