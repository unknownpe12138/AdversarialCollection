"""
Role Actor Network π_c: Maps state to action distribution.

For discrete action spaces: outputs categorical distribution
For continuous action spaces: outputs Gaussian distribution with tanh squashing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import List, Tuple, Optional
import numpy as np


class RoleActor(nn.Module):
    """
    Role-specific policy network π_c(a|s_k).
    
    For discrete actions (our case):
        Outputs logits for categorical distribution
        
    For continuous actions:
        Outputs mean and log_std for Gaussian distribution
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = 'relu',
        action_type: str = 'discrete',
        log_std_min: float = -20,
        log_std_max: float = 2
    ):
        """
        Initialize role actor.
        
        Args:
            state_dim: Dimension of input state s_k
            action_dim: Dimension of action space
            hidden_dims: Hidden layer sizes
            activation: Activation function
            action_type: 'discrete' or 'continuous'
            log_std_min: Minimum log std (for continuous)
            log_std_max: Maximum log std (for continuous)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build shared feature extractor
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Build action head
        if action_type == 'discrete':
            # Output logits for categorical distribution
            self.action_head = nn.Linear(prev_dim, action_dim)
        elif action_type == 'continuous':
            # Output mean and log_std for Gaussian
            self.mean_head = nn.Linear(prev_dim, action_dim)
            self.log_std_head = nn.Linear(prev_dim, action_dim)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
        
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
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get action distribution parameters.
        
        Args:
            state: State tensor of shape (batch_size, state_dim) or (state_dim,)
        
        Returns:
            For discrete: logits of shape (batch_size, action_dim)
            For continuous: (mean, log_std) tuple
        """
        features = self.feature_extractor(state)
        
        if self.action_type == 'discrete':
            logits = self.action_head(features)
            return logits
        else:  # continuous
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy π_c(a|s_k).
        
        Args:
            state: State tensor
            deterministic: If True, return mode of distribution
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        if self.action_type == 'discrete':
            return self._sample_discrete(state, deterministic)
        else:
            return self._sample_continuous(state, deterministic)
    
    def _sample_discrete(
        self,
        state: torch.Tensor,
        deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from categorical distribution."""
        logits = self.forward(state)
        
        if deterministic:
            # Greedy action
            action = torch.argmax(logits, dim=-1)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
        else:
            # Sample from categorical
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def _sample_continuous(
        self,
        state: torch.Tensor,
        deterministic: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from Gaussian with tanh squashing."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            # Use mean
            action_raw = mean
        else:
            # Sample from normal
            dist = Normal(mean, std)
            action_raw = dist.rsample()  # Reparameterization trick
        
        # Squash with tanh
        action = torch.tanh(action_raw)
        
        # Compute log probability with correction for tanh
        if deterministic:
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
        else:
            log_prob = dist.log_prob(action_raw).sum(dim=-1)
        
        # Correction for tanh squashing: log_prob - log(1 - tanh^2(a))
        log_prob = log_prob - torch.sum(
            torch.log(1 - action**2 + 1e-6), dim=-1
        )
        
        return action, log_prob
    
    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given state-action pairs.
        
        Args:
            state: State tensor
            action: Action tensor
        
        Returns:
            log_prob: Log probability of actions
            entropy: Entropy of distribution
        """
        if self.action_type == 'discrete':
            return self._evaluate_discrete(state, action)
        else:
            return self._evaluate_continuous(state, action)
    
    def _evaluate_discrete(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate discrete actions."""
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return log_prob, entropy
    
    def _evaluate_continuous(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate continuous actions."""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        # Inverse tanh to get raw action
        action_raw = torch.atanh(torch.clamp(action, -0.999, 0.999))
        
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        # Correction for tanh
        log_prob = log_prob - torch.sum(
            torch.log(1 - action**2 + 1e-6), dim=-1
        )
        
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy
    
    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities (for discrete actions only).
        
        Args:
            state: State tensor
        
        Returns:
            Action probabilities of shape (batch_size, action_dim)
        """
        if self.action_type != 'discrete':
            raise ValueError("Action probabilities only available for discrete actions")
        
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs


if __name__ == '__main__':
    # Test discrete actor
    print("Testing RoleActor (discrete)...")
    
    state_dim = 15
    action_dim = 10
    batch_size = 32
    
    actor = RoleActor(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        action_type='discrete'
    )
    
    print(f"Actor: {actor}")
    
    # Test single state
    state = torch.randn(state_dim)
    logits = actor(state)
    print(f"\nSingle state:")
    print(f"  State: {state.shape}")
    print(f"  Logits: {logits.shape}")
    
    # Test sampling
    action, log_prob = actor.sample(state, deterministic=False)
    print(f"  Sampled action: {action.item()}")
    print(f"  Log prob: {log_prob.item():.4f}")
    
    # Test batch
    states = torch.randn(batch_size, state_dim)
    actions, log_probs = actor.sample(states)
    print(f"\nBatch:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Log probs: {log_probs.shape}")
    
    # Test evaluation
    log_probs_eval, entropies = actor.evaluate(states, actions)
    print(f"  Evaluated log probs: {log_probs_eval.shape}")
    print(f"  Entropies: {entropies.shape}")
    print(f"  Mean entropy: {entropies.mean().item():.4f}")
    
    # Test action probabilities
    probs = actor.get_action_probabilities(states)
    print(f"  Action probs: {probs.shape}")
    print(f"  Prob sum: {probs[0].sum().item():.4f}")
    
    # Test gradient flow
    print("\nTesting gradient flow...")
    states = torch.randn(10, state_dim, requires_grad=True)
    actions, log_probs = actor.sample(states)
    loss = -log_probs.mean()
    loss.backward()
    print(f"  Gradient on states: {states.grad is not None}")
    print(f"  Gradient norm: {states.grad.norm().item():.6f}")
    
    print("\n✓ All discrete tests passed!")
    
    # Test continuous actor
    print("\n" + "="*50)
    print("Testing RoleActor (continuous)...")
    
    actor_continuous = RoleActor(
        state_dim=state_dim,
        action_dim=5,
        hidden_dims=[256, 256],
        action_type='continuous'
    )
    
    state = torch.randn(state_dim)
    mean, log_std = actor_continuous(state)
    print(f"  Mean: {mean.shape}")
    print(f"  Log std: {log_std.shape}")
    
    action, log_prob = actor_continuous.sample(state)
    print(f"  Sampled action: {action.shape}, range: [{action.min():.2f}, {action.max():.2f}]")
    print(f"  Log prob: {log_prob.item():.4f}")
    
    print("\n✓ All continuous tests passed!")
