"""
Temperature Parameter Management for VQ-HC-SAC.

Each role c has its own learnable temperature α_c and target entropy H_target,c.
This enables role-differentiated exploration:
- High-risk roles (explorers): high α_c, high entropy
- Low-risk roles (transporters): low α_c, low entropy
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np


class TemperatureManager:
    """
    Manages M learnable temperature parameters {α_c} for M roles.
    
    In SAC, the temperature α controls the exploration-exploitation trade-off:
    - Higher α → more exploration (higher entropy)
    - Lower α → more exploitation (lower entropy)
    
    Loss: L(α_c) = -α_c * (log π_c(a|s) + H_target,c)
    where H_target,c is the target entropy for role c.
    """
    
    def __init__(
        self,
        n_roles: int,
        action_dim: int,
        target_entropy_scale: float = -1.0,
        initial_alpha: float = 0.2,
        lr: float = 3e-4,
        per_role_entropy: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize temperature manager.
        
        Args:
            n_roles: Number of roles M
            action_dim: Dimension of action space
            target_entropy_scale: Scale for automatic target entropy
                                 H_target = scale * action_dim
            initial_alpha: Initial value for log(α)
            lr: Learning rate for α optimization
            per_role_entropy: If True, each role has different target entropy
                            If False, all roles share the same target entropy
            device: Device for tensors
        """
        self.n_roles = n_roles
        self.action_dim = action_dim
        self.device = device
        self.per_role_entropy = per_role_entropy
        
        # Log-space α for unconstrained optimization: α = exp(log_α)
        self.log_alphas = nn.ParameterList([
            nn.Parameter(
                torch.tensor(np.log(initial_alpha), dtype=torch.float32)
            ).to(device)
            for _ in range(n_roles)
        ])
        
        # Target entropies for each role
        if per_role_entropy:
            # Different target entropy for each role
            # Could be learned or manually set based on role
            self.target_entropies = torch.tensor([
                target_entropy_scale * action_dim for _ in range(n_roles)
            ], dtype=torch.float32).to(device)
        else:
            # Same target entropy for all roles
            target_entropy = target_entropy_scale * action_dim
            self.target_entropies = torch.tensor([
                target_entropy for _ in range(n_roles)
            ], dtype=torch.float32).to(device)
        
        # Optimizer for α parameters
        self.optimizer = torch.optim.Adam(self.log_alphas, lr=lr)
        
        print(f"TemperatureManager initialized:")
        print(f"  Roles: {n_roles}")
        print(f"  Initial α: {initial_alpha:.4f}")
        print(f"  Target entropies: {self.target_entropies.tolist()}")
    
    def get_alpha(self, role_id: int) -> torch.Tensor:
        """
        Get temperature α_c for a specific role.
        
        Args:
            role_id: Role index
        
        Returns:
            α_c = exp(log_α_c)
        """
        return torch.exp(self.log_alphas[role_id])
    
    def get_all_alphas(self) -> List[torch.Tensor]:
        """Get all temperature parameters."""
        return [self.get_alpha(i) for i in range(self.n_roles)]
    
    def get_target_entropy(self, role_id: int) -> torch.Tensor:
        """Get target entropy for a specific role."""
        return self.target_entropies[role_id]
    
    def compute_alpha_loss(
        self,
        role_id: int,
        log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute α loss for a specific role.
        
        L(α_c) = -α_c * E[log π_c(a|s) + H_target,c]
               = -α_c * (mean_log_prob + H_target,c)
        
        Args:
            role_id: Role index
            log_probs: Log probabilities from policy, shape (batch_size,)
        
        Returns:
            Alpha loss (scalar)
        """
        alpha = self.get_alpha(role_id).detach()  # Detach α from computation graph
        target_entropy = self.get_target_entropy(role_id)
        
        # Loss: -α * (log_prob + H_target)
        # Gradient w.r.t. log_α: -exp(log_α) * (log_prob + H_target)
        alpha_loss = -(self.log_alphas[role_id] * (log_probs + target_entropy).detach()).mean()
        
        return alpha_loss
    
    def update(
        self,
        role_losses: Dict[int, torch.Tensor]
    ) -> Dict[int, float]:
        """
        Update all α parameters.
        
        Args:
            role_losses: Dictionary mapping role_id to alpha loss
        
        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()
        
        # Sum all role losses
        total_loss = sum(role_losses.values())
        
        # Backward and step
        total_loss.backward()
        self.optimizer.step()
        
        # Return loss values
        return {role_id: loss.item() for role_id, loss in role_losses.items()}
    
    def set_target_entropy(self, role_id: int, target_entropy: float):
        """
        Manually set target entropy for a role.
        
        Useful for role-specific entropy tuning:
        - Explorer role: higher target entropy (more exploration)
        - Transporter role: lower target entropy (more exploitation)
        """
        self.target_entropies[role_id] = target_entropy
    
    def get_statistics(self) -> Dict[str, float]:
        """Get temperature statistics."""
        alphas = [self.get_alpha(i).item() for i in range(self.n_roles)]
        
        return {
            f'alpha_{i}': alphas[i] for i in range(self.n_roles)
        } | {
            'alpha_mean': np.mean(alphas),
            'alpha_std': np.std(alphas),
            'alpha_min': np.min(alphas),
            'alpha_max': np.max(alphas)
        }
    
    def save_state_dict(self, path: str):
        """Save temperature parameters."""
        state_dict = {
            'log_alphas': [log_alpha.data for log_alpha in self.log_alphas],
            'target_entropies': self.target_entropies,
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state_dict, path)
    
    def load_state_dict(self, path: str):
        """Load temperature parameters."""
        state_dict = torch.load(path, map_location=self.device)
        
        for i, log_alpha_data in enumerate(state_dict['log_alphas']):
            self.log_alphas[i].data = log_alpha_data.to(self.device)
        
        self.target_entropies = state_dict['target_entropies'].to(self.device)
        self.optimizer.load_state_dict(state_dict['optimizer'])
    
    def __repr__(self) -> str:
        alphas = [f"{self.get_alpha(i).item():.4f}" for i in range(self.n_roles)]
        return f"TemperatureManager(n_roles={self.n_roles}, alphas={alphas})"


class FixedTemperatureManager:
    """
    Fixed (non-learnable) temperature manager.
    
    Alternative to learnable α when you want to manually control exploration.
    """
    
    def __init__(
        self,
        n_roles: int,
        alpha_values: Optional[List[float]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize fixed temperature manager.
        
        Args:
            n_roles: Number of roles
            alpha_values: Fixed α values for each role (default: all 0.2)
            device: Device for tensors
        """
        self.n_roles = n_roles
        self.device = device
        
        if alpha_values is None:
            alpha_values = [0.2] * n_roles
        
        assert len(alpha_values) == n_roles, \
            f"alpha_values length ({len(alpha_values)}) must match n_roles ({n_roles})"
        
        self.alphas = [
            torch.tensor(alpha, dtype=torch.float32).to(device)
            for alpha in alpha_values
        ]
        
        print(f"FixedTemperatureManager initialized with alphas: {alpha_values}")
    
    def get_alpha(self, role_id: int) -> torch.Tensor:
        """Get fixed α for a role."""
        return self.alphas[role_id]
    
    def get_all_alphas(self) -> List[torch.Tensor]:
        """Get all α values."""
        return self.alphas
    
    def compute_alpha_loss(self, role_id: int, log_probs: torch.Tensor) -> torch.Tensor:
        """No loss for fixed temperature."""
        return torch.tensor(0.0, device=self.device)
    
    def update(self, role_losses: Dict[int, torch.Tensor]) -> Dict[int, float]:
        """No update for fixed temperature."""
        return {role_id: 0.0 for role_id in range(self.n_roles)}
    
    def get_statistics(self) -> Dict[str, float]:
        """Get temperature statistics."""
        alphas = [alpha.item() for alpha in self.alphas]
        return {f'alpha_{i}': alphas[i] for i in range(self.n_roles)}


if __name__ == '__main__':
    # Test TemperatureManager
    print("Testing TemperatureManager...")
    print("="*60)
    
    n_roles = 3
    action_dim = 8
    batch_size = 32
    
    # Create manager
    temp_manager = TemperatureManager(
        n_roles=n_roles,
        action_dim=action_dim,
        target_entropy_scale=-1.0,
        initial_alpha=0.2,
        lr=3e-4,
        per_role_entropy=True,
        device='cpu'
    )
    
    print(f"\n{temp_manager}")
    
    # Test getting α
    print("\n" + "-"*60)
    print("Testing alpha retrieval...")
    
    for role_id in range(n_roles):
        alpha = temp_manager.get_alpha(role_id)
        target_entropy = temp_manager.get_target_entropy(role_id)
        print(f"  Role {role_id}: α={alpha.item():.4f}, H_target={target_entropy.item():.4f}")
    
    # Test alpha loss computation
    print("\n" + "-"*60)
    print("Testing alpha loss computation...")
    
    role_losses = {}
    for role_id in range(n_roles):
        # Simulate log probabilities from policy
        log_probs = torch.randn(batch_size) * 2 - 3  # Around -3
        
        alpha_loss = temp_manager.compute_alpha_loss(role_id, log_probs)
        role_losses[role_id] = alpha_loss
        
        print(f"  Role {role_id}: loss={alpha_loss.item():.4f}")
    
    # Test update
    print("\n" + "-"*60)
    print("Testing alpha update...")
    
    alphas_before = [temp_manager.get_alpha(i).item() for i in range(n_roles)]
    print(f"  Alphas before: {[f'{a:.4f}' for a in alphas_before]}")
    
    # Update
    loss_values = temp_manager.update(role_losses)
    
    alphas_after = [temp_manager.get_alpha(i).item() for i in range(n_roles)]
    print(f"  Alphas after: {[f'{a:.4f}' for a in alphas_after]}")
    
    # Verify α changed
    changed = any(abs(a - b) > 1e-6 for a, b in zip(alphas_before, alphas_after))
    print(f"  Alphas changed: {changed}")
    
    # Test multiple updates
    print("\n" + "-"*60)
    print("Testing multiple updates (simulating training)...")
    
    for step in range(10):
        role_losses = {}
        for role_id in range(n_roles):
            log_probs = torch.randn(batch_size) * 2 - 3
            alpha_loss = temp_manager.compute_alpha_loss(role_id, log_probs)
            role_losses[role_id] = alpha_loss
        
        temp_manager.update(role_losses)
    
    alphas_final = [temp_manager.get_alpha(i).item() for i in range(n_roles)]
    print(f"  Alphas after 10 updates: {[f'{a:.4f}' for a in alphas_final]}")
    
    # Test statistics
    print("\n" + "-"*60)
    print("Testing statistics...")
    
    stats = temp_manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test manual target entropy setting
    print("\n" + "-"*60)
    print("Testing manual target entropy setting...")
    
    # Set different entropies for different roles
    temp_manager.set_target_entropy(0, -5.0)  # Explorer: high entropy target
    temp_manager.set_target_entropy(1, -8.0)  # Balanced
    temp_manager.set_target_entropy(2, -10.0)  # Transporter: low entropy target
    
    for role_id in range(n_roles):
        target_entropy = temp_manager.get_target_entropy(role_id)
        print(f"  Role {role_id}: H_target={target_entropy.item():.4f}")
    
    # Test FixedTemperatureManager
    print("\n" + "="*60)
    print("Testing FixedTemperatureManager...")
    
    fixed_temp_manager = FixedTemperatureManager(
        n_roles=n_roles,
        alpha_values=[0.3, 0.2, 0.1],  # Different α for each role
        device='cpu'
    )
    
    print(f"\nFixed alphas:")
    for role_id in range(n_roles):
        alpha = fixed_temp_manager.get_alpha(role_id)
        print(f"  Role {role_id}: α={alpha.item():.4f}")
    
    # Test that update does nothing
    log_probs = torch.randn(batch_size)
    alpha_loss = fixed_temp_manager.compute_alpha_loss(0, log_probs)
    print(f"\nFixed alpha loss: {alpha_loss.item():.4f} (should be 0.0)")
    
    stats = fixed_temp_manager.get_statistics()
    print(f"Fixed stats: {stats}")
    
    print("\n" + "="*60)
    print("✓ All TemperatureManager tests passed!")
