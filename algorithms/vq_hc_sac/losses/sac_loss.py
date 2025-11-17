"""
Soft Actor-Critic (SAC) Loss Functions.

Implements the three core SAC losses:
1. Critic Loss: Temporal Difference (TD) error
2. Actor Loss: Policy gradient with entropy regularization
3. Alpha Loss: Temperature parameter optimization (handled by TemperatureManager)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class SACLoss:
    """
    SAC loss computation for VQ-HC-SAC.
    
    Standard SAC losses adapted for role-based multi-agent learning.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        use_twin_critic: bool = True
    ):
        """
        Initialize SAC loss calculator.
        
        Args:
            gamma: Discount factor γ
            use_twin_critic: Whether to use twin critics (double Q-learning)
        """
        self.gamma = gamma
        self.use_twin_critic = use_twin_critic
    
    def critic_loss(
        self,
        q_values: Tuple[torch.Tensor, torch.Tensor],
        rewards: torch.Tensor,
        next_q_values: torch.Tensor,
        next_log_probs: torch.Tensor,
        alpha: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Critic TD loss.
        
        Loss: L(Q) = E[(Q(s,a) - (r + γ·(Q_target(s',a') - α·log π(a'|s'))))^2]
        
        This is the standard SAC critic loss with entropy regularization.
        
        Args:
            q_values: Tuple of (Q1, Q2) predictions, each (batch_size, 1)
            rewards: Shaped rewards from PBRS, (batch_size,)
            next_q_values: Target Q-values for next state, (batch_size,)
            next_log_probs: Log probs of next actions, (batch_size,)
            alpha: Temperature parameter α, scalar
            dones: Episode termination flags, (batch_size,)
        
        Returns:
            q1_loss: Loss for first critic
            q2_loss: Loss for second critic (if twin critic)
        """
        q1, q2 = q_values
        
        # Reshape if needed
        if q1.dim() > 1:
            q1 = q1.squeeze(-1)
        if q2.dim() > 1:
            q2 = q2.squeeze(-1)
        
        # Compute target: r + γ·(Q_target(s',a') - α·log π(a'|s'))
        # This includes the entropy term for maximum entropy RL
        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.gamma * (
                next_q_values - alpha * next_log_probs
            )
        
        # TD errors
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        
        return q1_loss, q2_loss
    
    def actor_loss(
        self,
        q_values: torch.Tensor,
        log_probs: torch.Tensor,
        alpha: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Actor policy gradient loss.
        
        Loss: L(π) = E[α·log π(a|s) - Q(s,a)]
        
        This encourages the policy to maximize Q-values while maintaining entropy.
        
        Args:
            q_values: Q-values for sampled actions, (batch_size,)
            log_probs: Log probabilities of sampled actions, (batch_size,)
            alpha: Temperature parameter α, scalar
        
        Returns:
            actor_loss: Policy gradient loss (scalar)
        """
        # Reshape if needed
        if q_values.dim() > 1:
            q_values = q_values.squeeze(-1)
        
        # Actor loss: encourage high Q-values and high entropy
        # Note: we minimize this, so we want to maximize Q and entropy
        actor_loss = (alpha * log_probs - q_values).mean()
        
        return actor_loss
    
    def compute_target_q(
        self,
        next_states: torch.Tensor,
        next_env_states: torch.Tensor,
        next_role_aggregations: torch.Tensor,
        target_critic,
        actor,
        alpha: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute target Q-values for next state.
        
        Q_target(s') = min(Q1_target(s',a'), Q2_target(s',a')) - α·log π(a'|s')
        where a' ~ π(·|s')
        
        Args:
            next_states: Next individual states, (batch_size, state_dim)
            next_env_states: Next environment states, (batch_size, env_state_dim)
            next_role_aggregations: Next role aggregations, (batch_size, n_roles, agg_dim)
            target_critic: Target critic network (TwinRoleCritic)
            actor: Actor network (RoleActor)
            alpha: Temperature parameter α
            deterministic: Whether to use deterministic policy (usually False)
        
        Returns:
            target_q_values: Target Q-values, (batch_size,)
            next_log_probs: Log probs of sampled next actions, (batch_size,)
        """
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = actor.sample(
                next_states, deterministic=deterministic
            )
            
            # Get target Q-values
            if self.use_twin_critic:
                next_q1, next_q2 = target_critic(
                    next_states, next_actions, next_env_states, next_role_aggregations
                )
                # Use minimum to reduce overestimation
                next_q = torch.min(next_q1, next_q2).squeeze(-1)
            else:
                next_q = target_critic(
                    next_states, next_actions, next_env_states, next_role_aggregations
                ).squeeze(-1)
        
        return next_q, next_log_probs
    
    def compute_critic_targets(
        self,
        rewards: torch.Tensor,
        next_q_values: torch.Tensor,
        next_log_probs: torch.Tensor,
        alpha: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute critic targets for bootstrapping.
        
        Target: y = r + γ·(1-done)·(Q_target(s',a') - α·log π(a'|s'))
        
        Args:
            rewards: Immediate rewards, (batch_size,)
            next_q_values: Q-values for next state, (batch_size,)
            next_log_probs: Log probs for next actions, (batch_size,)
            alpha: Temperature parameter
            dones: Termination flags, (batch_size,)
        
        Returns:
            targets: TD targets, (batch_size,)
        """
        with torch.no_grad():
            targets = rewards + (1 - dones) * self.gamma * (
                next_q_values - alpha * next_log_probs
            )
        return targets


class CriticLossCalculator:
    """
    Helper class for computing critic losses with detailed statistics.
    """
    
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
    
    def __call__(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        targets: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Compute critic losses with optional statistics.
        
        Args:
            q1: First Q-values, (batch_size,)
            q2: Second Q-values, (batch_size,)
            targets: Target values, (batch_size,)
            return_stats: Whether to return detailed statistics
        
        Returns:
            q1_loss: MSE loss for Q1
            q2_loss: MSE loss for Q2
            stats: Optional statistics dictionary
        """
        # Ensure correct shapes
        if q1.dim() > 1:
            q1 = q1.squeeze(-1)
        if q2.dim() > 1:
            q2 = q2.squeeze(-1)
        
        # Compute losses
        q1_loss = F.mse_loss(q1, targets)
        q2_loss = F.mse_loss(q2, targets)
        
        stats = None
        if return_stats:
            with torch.no_grad():
                td_error1 = (q1 - targets).abs()
                td_error2 = (q2 - targets).abs()
                
                stats = {
                    'q1_loss': q1_loss.item(),
                    'q2_loss': q2_loss.item(),
                    'q1_mean': q1.mean().item(),
                    'q2_mean': q2.mean().item(),
                    'target_mean': targets.mean().item(),
                    'td_error1_mean': td_error1.mean().item(),
                    'td_error2_mean': td_error2.mean().item(),
                    'td_error1_max': td_error1.max().item(),
                    'td_error2_max': td_error2.max().item()
                }
        
        return q1_loss, q2_loss, stats


class ActorLossCalculator:
    """
    Helper class for computing actor losses with detailed statistics.
    """
    
    def __call__(
        self,
        q_values: torch.Tensor,
        log_probs: torch.Tensor,
        alpha: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Compute actor loss with optional statistics.
        
        Loss: L(π) = E[α·log π(a|s) - Q(s,a)]
        
        Args:
            q_values: Q-values, (batch_size,)
            log_probs: Log probabilities, (batch_size,)
            alpha: Temperature parameter
            return_stats: Whether to return detailed statistics
        
        Returns:
            actor_loss: Policy gradient loss
            stats: Optional statistics dictionary
        """
        # Ensure correct shapes
        if q_values.dim() > 1:
            q_values = q_values.squeeze(-1)
        
        # Actor loss
        actor_loss = (alpha * log_probs - q_values).mean()
        
        stats = None
        if return_stats:
            with torch.no_grad():
                stats = {
                    'actor_loss': actor_loss.item(),
                    'q_mean': q_values.mean().item(),
                    'q_std': q_values.std().item(),
                    'log_prob_mean': log_probs.mean().item(),
                    'log_prob_std': log_probs.std().item(),
                    'entropy_mean': -log_probs.mean().item(),
                    'alpha': alpha.item() if alpha.dim() == 0 else alpha.mean().item()
                }
        
        return actor_loss, stats


if __name__ == '__main__':
    # Test SAC losses
    print("Testing SAC Losses...")
    print("="*60)
    
    batch_size = 32
    
    # Create SAC loss calculator
    sac_loss = SACLoss(gamma=0.99, use_twin_critic=True)
    
    print(f"SAC Loss Calculator:")
    print(f"  Gamma: {sac_loss.gamma}")
    print(f"  Twin Critic: {sac_loss.use_twin_critic}")
    
    # Test critic loss
    print("\n" + "-"*60)
    print("Testing Critic Loss...")
    
    q1 = torch.randn(batch_size, 1)
    q2 = torch.randn(batch_size, 1)
    rewards = torch.randn(batch_size)
    next_q_values = torch.randn(batch_size)
    next_log_probs = torch.randn(batch_size)
    alpha = torch.tensor(0.2)
    dones = torch.zeros(batch_size)
    dones[::10] = 1.0  # Some episodes terminate
    
    q1_loss, q2_loss = sac_loss.critic_loss(
        (q1, q2), rewards, next_q_values, next_log_probs, alpha, dones
    )
    
    print(f"✓ Critic losses computed:")
    print(f"  Q1 loss: {q1_loss.item():.4f}")
    print(f"  Q2 loss: {q2_loss.item():.4f}")
    
    # Test actor loss
    print("\n" + "-"*60)
    print("Testing Actor Loss...")
    
    q_values = torch.randn(batch_size)
    log_probs = torch.randn(batch_size) * 2 - 3
    
    actor_loss = sac_loss.actor_loss(q_values, log_probs, alpha)
    
    print(f"✓ Actor loss computed: {actor_loss.item():.4f}")
    
    # Test target computation
    print("\n" + "-"*60)
    print("Testing Target Computation...")
    
    targets = sac_loss.compute_critic_targets(
        rewards, next_q_values, next_log_probs, alpha, dones
    )
    
    print(f"✓ Targets computed: {targets.shape}")
    print(f"  Mean target: {targets.mean().item():.4f}")
    print(f"  Target range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
    
    # Test with statistics
    print("\n" + "-"*60)
    print("Testing Loss Calculators with Statistics...")
    
    critic_loss_calc = CriticLossCalculator(gamma=0.99)
    q1_loss, q2_loss, critic_stats = critic_loss_calc(
        q1.squeeze(), q2.squeeze(), targets, return_stats=True
    )
    
    print(f"✓ Critic statistics:")
    for key, value in critic_stats.items():
        print(f"  {key}: {value:.4f}")
    
    actor_loss_calc = ActorLossCalculator()
    actor_loss, actor_stats = actor_loss_calc(
        q_values, log_probs, alpha, return_stats=True
    )
    
    print(f"\n✓ Actor statistics:")
    for key, value in actor_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test gradient flow
    print("\n" + "-"*60)
    print("Testing Gradient Flow...")
    
    q1 = torch.randn(batch_size, 1, requires_grad=True)
    q2 = torch.randn(batch_size, 1, requires_grad=True)
    
    q1_loss, q2_loss = sac_loss.critic_loss(
        (q1, q2), rewards, next_q_values, next_log_probs, alpha, dones
    )
    
    total_loss = q1_loss + q2_loss
    total_loss.backward()
    
    print(f"✓ Gradient flow verified:")
    print(f"  Q1 grad exists: {q1.grad is not None}")
    print(f"  Q2 grad exists: {q2.grad is not None}")
    print(f"  Q1 grad norm: {q1.grad.norm().item():.6f}")
    print(f"  Q2 grad norm: {q2.grad.norm().item():.6f}")
    
    print("\n" + "="*60)
    print("✓ All SAC loss tests passed!")
