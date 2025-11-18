"""
Role Critic Network Q_c: Evaluates state-action values for each role.

Input: (s_k, a_k, s_env, {s̄_j})
  - Individual state and action: (s_k, a_k)
  - Global environment state: s_env
  - Role aggregations: {s̄_1, ..., s̄_M}

This design decouples critic input from N agents.
"""

import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


class RoleCritic(nn.Module):
    """
    Role-specific critic Q_c(s_k, a_k, s_env, {s̄_j}).
    
    Architecture:
        [s_k, a_k, s_env, s̄_1, ..., s̄_M] → [Linear → ReLU]* → Q-value
    
    The input dimension is fixed regardless of N (number of agents),
    scaling only with M (number of roles).
    """
    
    def __init__(
        self,
        individual_state_dim: int,
        action_dim: int,
        env_state_dim: int,
        n_roles: int,
        role_aggregation_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        layer_norm: bool = True,
        action_type: str = 'discrete'
    ):
        """
        Initialize role critic.
        
        Args:
            individual_state_dim: Dimension of individual state s_k
            action_dim: Dimension of action a_k
            env_state_dim: Dimension of environment state s_env
            n_roles: Number of roles M
            role_aggregation_dim: Dimension of each role aggregation s̄_j
            hidden_dims: Hidden layer sizes
            activation: Activation function
            layer_norm: Whether to use layer normalization
            action_type: 'discrete' or 'continuous'
        """
        super().__init__()
        
        self.individual_state_dim = individual_state_dim
        self.action_dim = action_dim
        self.env_state_dim = env_state_dim
        self.n_roles = n_roles
        self.role_aggregation_dim = role_aggregation_dim
        self.action_type = action_type
        
        # Debug flag: only print once
        self._debug_printed = False
        
        # For discrete actions, use one-hot encoding
        if action_type == 'discrete':
            action_input_dim = action_dim
        else:
            action_input_dim = action_dim
        
        # Total input dimension
        # [s_k, a_k, s_env, s̄_1, ..., s̄_M]
        self.input_dim = (
            individual_state_dim +  # Individual state
            action_input_dim +       # Action
            env_state_dim +          # Environment state
            n_roles * role_aggregation_dim  # M role aggregations
        )
        
        # Build network
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        # Output Q-value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        individual_state: torch.Tensor,
        action: torch.Tensor,
        env_state: torch.Tensor,
        role_aggregations: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute Q-value.
        
        Args:
            individual_state: Individual state s_k of shape (batch_size, individual_state_dim)
            action: Action a_k of shape (batch_size,) for discrete or (batch_size, action_dim) for continuous
            env_state: Environment state s_env of shape (batch_size, env_state_dim)
            role_aggregations: Role aggregations {s̄_j} of shape (batch_size, n_roles, role_aggregation_dim)
        
        Returns:
            q_value: Q-value of shape (batch_size, 1)
        """
        batch_size = individual_state.shape[0]
        
        # Debug: Print input shapes before processing (only once)
        if not self._debug_printed:
            print(f"\n[DEBUG Critic] Input shapes (first call):")
            print(f"  individual_state: {individual_state.shape}")
            print(f"  action (original): {action.shape}")
            print(f"  env_state: {env_state.shape}")
            print(f"  role_aggregations: {role_aggregations.shape}")
            print(f"  action type: {self.action_type}, action_dim: {self.action_dim}")
            print(f"  Expected role_aggregations: (batch={batch_size}, n_roles={self.n_roles}, dim={self.role_aggregation_dim})")
        
        # Process action
        if self.action_type == 'discrete':
            # Convert discrete action to one-hot
            if action.dim() == 1:
                # Shape: [batch_size] -> [batch_size, action_dim]
                action = action.long()
                action_input = torch.zeros(batch_size, self.action_dim, device=action.device)
                action_input.scatter_(1, action.unsqueeze(1), 1.0)
            elif action.shape[-1] == 1:
                # Shape: [batch_size, 1] -> [batch_size, action_dim]
                action = action.long().squeeze(-1)
                action_input = torch.zeros(batch_size, self.action_dim, device=action.device)
                action_input.scatter_(1, action.unsqueeze(1), 1.0)
            else:
                # Already one-hot encoded
                action_input = action
        else:
            action_input = action
        
        # Flatten role aggregations: (batch_size, n_roles, dim) → (batch_size, n_roles * dim)
        role_aggregations_flat = role_aggregations.reshape(batch_size, -1)
        
        if not self._debug_printed:
            print(f"  role_aggregations_flat: {role_aggregations_flat.shape}")
        
        # Concatenate all inputs
        if not self._debug_printed:
            print(f"  action_input (after processing): {action_input.shape}")
        
        critic_input = torch.cat([
            individual_state,
            action_input,
            env_state,
            role_aggregations_flat
        ], dim=-1)
        
        if not self._debug_printed:
            print(f"  critic_input (after concat): {critic_input.shape}")
            print(f"  Expected input dim: {self.input_dim}")
            print(f"  Network first layer weight: {list(self.network.children())[0].weight.shape}")
            print(f"  ✓ Debug info printed once, suppressing future outputs\n")
            self._debug_printed = True
        
        # Forward through network
        q_value = self.network(critic_input)
        
        return q_value
    
    def get_input_dim(self) -> int:
        """Get total input dimension."""
        return self.input_dim


class TwinRoleCritic(nn.Module):
    """
    Twin critics for reduced overestimation (standard in SAC).
    
    Uses minimum of two Q-values: Q = min(Q1, Q2)
    """
    
    def __init__(
        self,
        individual_state_dim: int,
        action_dim: int,
        env_state_dim: int,
        n_roles: int,
        role_aggregation_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        layer_norm: bool = True,
        action_type: str = 'discrete'
    ):
        """Initialize twin critics."""
        super().__init__()
        
        self.q1 = RoleCritic(
            individual_state_dim=individual_state_dim,
            action_dim=action_dim,
            env_state_dim=env_state_dim,
            n_roles=n_roles,
            role_aggregation_dim=role_aggregation_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            action_type=action_type
        )
        
        self.q2 = RoleCritic(
            individual_state_dim=individual_state_dim,
            action_dim=action_dim,
            env_state_dim=env_state_dim,
            n_roles=n_roles,
            role_aggregation_dim=role_aggregation_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            action_type=action_type
        )
    
    def forward(
        self,
        individual_state: torch.Tensor,
        action: torch.Tensor,
        env_state: torch.Tensor,
        role_aggregations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both Q-values.
        
        Returns:
            (q1, q2): Tuple of Q-values
        """
        q1 = self.q1(individual_state, action, env_state, role_aggregations)
        q2 = self.q2(individual_state, action, env_state, role_aggregations)
        
        return q1, q2
    
    def q_min(
        self,
        individual_state: torch.Tensor,
        action: torch.Tensor,
        env_state: torch.Tensor,
        role_aggregations: torch.Tensor
    ) -> torch.Tensor:
        """Compute minimum Q-value."""
        q1, q2 = self.forward(individual_state, action, env_state, role_aggregations)
        return torch.min(q1, q2)


if __name__ == '__main__':
    # Test critic
    print("Testing RoleCritic...")
    
    individual_state_dim = 10
    action_dim = 8
    env_state_dim = 20
    n_roles = 3
    role_aggregation_dim = 64
    batch_size = 32
    
    critic = RoleCritic(
        individual_state_dim=individual_state_dim,
        action_dim=action_dim,
        env_state_dim=env_state_dim,
        n_roles=n_roles,
        role_aggregation_dim=role_aggregation_dim,
        hidden_dims=[256, 256],
        action_type='discrete'
    )
    
    print(f"Critic input dim: {critic.get_input_dim()}")
    print(f"  = {individual_state_dim} (s_k) + {action_dim} (a_k) + "
          f"{env_state_dim} (s_env) + {n_roles}×{role_aggregation_dim} (roles)")
    
    # Test single forward pass
    individual_state = torch.randn(batch_size, individual_state_dim)
    action = torch.randint(0, action_dim, (batch_size,))
    env_state = torch.randn(batch_size, env_state_dim)
    role_aggregations = torch.randn(batch_size, n_roles, role_aggregation_dim)
    
    q_value = critic(individual_state, action, env_state, role_aggregations)
    print(f"\nBatch forward pass:")
    print(f"  Individual state: {individual_state.shape}")
    print(f"  Action: {action.shape}")
    print(f"  Env state: {env_state.shape}")
    print(f"  Role aggregations: {role_aggregations.shape}")
    print(f"  Q-value: {q_value.shape}")
    print(f"  Q-value range: [{q_value.min().item():.4f}, {q_value.max().item():.4f}]")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    individual_state = torch.randn(10, individual_state_dim, requires_grad=True)
    action = torch.randint(0, action_dim, (10,))
    env_state = torch.randn(10, env_state_dim)
    role_aggregations = torch.randn(10, n_roles, role_aggregation_dim)
    
    q_value = critic(individual_state, action, env_state, role_aggregations)
    loss = q_value.mean()
    loss.backward()
    print(f"  Gradient on individual_state: {individual_state.grad is not None}")
    print(f"  Gradient norm: {individual_state.grad.norm().item():.6f}")
    
    print("\n" + "="*50)
    print("Testing TwinRoleCritic...")
    
    twin_critic = TwinRoleCritic(
        individual_state_dim=individual_state_dim,
        action_dim=action_dim,
        env_state_dim=env_state_dim,
        n_roles=n_roles,
        role_aggregation_dim=role_aggregation_dim,
        hidden_dims=[256, 256],
        action_type='discrete'
    )
    
    # Test twin critics
    individual_state = torch.randn(batch_size, individual_state_dim)
    action = torch.randint(0, action_dim, (batch_size,))
    env_state = torch.randn(batch_size, env_state_dim)
    role_aggregations = torch.randn(batch_size, n_roles, role_aggregation_dim)
    
    q1, q2 = twin_critic(individual_state, action, env_state, role_aggregations)
    q_min = twin_critic.q_min(individual_state, action, env_state, role_aggregations)
    
    print(f"\nTwin critics:")
    print(f"  Q1: {q1.shape}, mean: {q1.mean().item():.4f}")
    print(f"  Q2: {q2.shape}, mean: {q2.mean().item():.4f}")
    print(f"  Q_min: {q_min.shape}, mean: {q_min.mean().item():.4f}")
    
    # Verify Q_min is indeed minimum
    assert torch.allclose(q_min, torch.min(q1, q2)), "Q_min should equal min(Q1, Q2)"
    
    print("\n✓ All tests passed!")
