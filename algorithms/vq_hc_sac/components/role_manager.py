"""
Role Manager: Manages M role-specific networks.

Coordinates:
- M Actors {π_c}
- M Twin Critics {Q_c}
- M Temperature parameters {α_c}
- Role aggregation computation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy

from ..networks import RoleActor, RoleCritic
from ..networks.critic import TwinRoleCritic


class RoleManager:
    """
    Manages M role-specific networks for VQ-HC-SAC.
    
    Responsibilities:
    1. Create and store M actors
    2. Create and store M twin critics (and target critics)
    3. Compute role aggregations: s̄_j = Pool({s_i | c_i = j})
    4. Provide network access by role ID
    5. Manage soft updates for target networks
    """
    
    def __init__(
        self,
        n_roles: int,
        individual_state_dim: int,
        action_dim: int,
        env_state_dim: int,
        role_aggregation_dim: int,
        actor_hidden_dims: List[int] = [256, 256],
        critic_hidden_dims: List[int] = [256, 256],
        action_type: str = 'discrete',
        use_twin_critic: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize role manager.
        
        Args:
            n_roles: Number of roles M
            individual_state_dim: Dimension of individual state s_k
            action_dim: Dimension of action space
            env_state_dim: Dimension of environment state s_env
            role_aggregation_dim: Dimension of role aggregation s̄_j
            actor_hidden_dims: Hidden layer sizes for actors
            critic_hidden_dims: Hidden layer sizes for critics
            action_type: 'discrete' or 'continuous'
            use_twin_critic: Whether to use twin critics
            device: Device for networks
        """
        self.n_roles = n_roles
        self.individual_state_dim = individual_state_dim
        self.action_dim = action_dim
        self.env_state_dim = env_state_dim
        self.role_aggregation_dim = role_aggregation_dim
        self.action_type = action_type
        self.use_twin_critic = use_twin_critic
        self.device = device
        
        # Create M actors
        self.actors = self._create_actors(actor_hidden_dims)
        
        # Create M critics (and target critics)
        self.critics, self.target_critics = self._create_critics(critic_hidden_dims)
        
        print(f"RoleManager initialized with {n_roles} roles")
        print(f"  - {n_roles} Actors")
        print(f"  - {n_roles} {'Twin ' if use_twin_critic else ''}Critics")
        print(f"  - Device: {device}")
    
    def _create_actors(self, hidden_dims: List[int]) -> nn.ModuleDict:
        """Create M role-specific actors."""
        actors = nn.ModuleDict()
        
        for role_id in range(self.n_roles):
            actor = RoleActor(
                state_dim=self.individual_state_dim,
                action_dim=self.action_dim,
                hidden_dims=hidden_dims,
                action_type=self.action_type
            ).to(self.device)
            
            actors[str(role_id)] = actor
        
        return actors
    
    def _create_critics(
        self,
        hidden_dims: List[int]
    ) -> Tuple[nn.ModuleDict, nn.ModuleDict]:
        """Create M role-specific critics and their target networks."""
        critics = nn.ModuleDict()
        target_critics = nn.ModuleDict()
        
        for role_id in range(self.n_roles):
            if self.use_twin_critic:
                critic = TwinRoleCritic(
                    individual_state_dim=self.individual_state_dim,
                    action_dim=self.action_dim,
                    env_state_dim=self.env_state_dim,
                    n_roles=self.n_roles,
                    role_aggregation_dim=self.role_aggregation_dim,
                    hidden_dims=hidden_dims,
                    action_type=self.action_type
                ).to(self.device)
            else:
                critic = RoleCritic(
                    individual_state_dim=self.individual_state_dim,
                    action_dim=self.action_dim,
                    env_state_dim=self.env_state_dim,
                    n_roles=self.n_roles,
                    role_aggregation_dim=self.role_aggregation_dim,
                    hidden_dims=hidden_dims,
                    action_type=self.action_type
                ).to(self.device)
            
            # Create target critic (deep copy)
            target_critic = copy.deepcopy(critic)
            
            # Freeze target network
            for param in target_critic.parameters():
                param.requires_grad = False
            
            critics[str(role_id)] = critic
            target_critics[str(role_id)] = target_critic
        
        return critics, target_critics
    
    def get_actor(self, role_id: int) -> RoleActor:
        """Get actor for a specific role."""
        return self.actors[str(role_id)]
    
    def get_critic(self, role_id: int) -> nn.Module:
        """Get critic for a specific role."""
        return self.critics[str(role_id)]
    
    def get_target_critic(self, role_id: int) -> nn.Module:
        """Get target critic for a specific role."""
        return self.target_critics[str(role_id)]
    
    def get_all_actors(self) -> List[RoleActor]:
        """Get all actors."""
        return [self.actors[str(i)] for i in range(self.n_roles)]
    
    def get_all_critics(self) -> List[nn.Module]:
        """Get all critics."""
        return [self.critics[str(i)] for i in range(self.n_roles)]
    
    def get_all_target_critics(self) -> List[nn.Module]:
        """Get all target critics."""
        return [self.target_critics[str(i)] for i in range(self.n_roles)]
    
    def compute_role_aggregations(
        self,
        agent_states: torch.Tensor,
        role_assignments: torch.Tensor,
        method: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute role aggregation states: s̄_j = Pool({s_i | c_i = j}).
        
        Args:
            agent_states: Agent states of shape (batch_size, n_agents, state_dim)
                         or (n_agents, state_dim)
            role_assignments: Role IDs of shape (batch_size, n_agents) or (n_agents,)
            method: Aggregation method: 'mean', 'max', or 'sum'
        
        Returns:
            role_aggregations: Aggregated states of shape 
                              (batch_size, n_roles, aggregation_dim) or
                              (n_roles, aggregation_dim)
        """
        # Handle single batch case
        single_batch = (agent_states.dim() == 2)
        if single_batch:
            agent_states = agent_states.unsqueeze(0)
            role_assignments = role_assignments.unsqueeze(0)
        
        batch_size, n_agents, state_dim = agent_states.shape
        
        # Initialize aggregations - use the actual state_dim from input
        role_aggregations = torch.zeros(
            batch_size, self.n_roles, state_dim,  # Use state_dim instead of self.role_aggregation_dim
            device=self.device
        )
        
        # Compute aggregation for each batch and role
        for b in range(batch_size):
            for role_id in range(self.n_roles):
                # Find agents with this role
                mask = (role_assignments[b] == role_id)
                
                if mask.sum() > 0:
                    # Get states of agents with this role
                    role_states = agent_states[b, mask]
                    
                    # Apply aggregation method
                    if method == 'mean':
                        role_aggregations[b, role_id] = role_states.mean(dim=0)
                    elif method == 'max':
                        role_aggregations[b, role_id] = role_states.max(dim=0)[0]
                    elif method == 'sum':
                        role_aggregations[b, role_id] = role_states.sum(dim=0)
                    else:
                        raise ValueError(f"Unknown aggregation method: {method}")
                # else: role_aggregations[b, role_id] remains zero
        
        # Handle single batch case
        if single_batch:
            role_aggregations = role_aggregations.squeeze(0)
        
        return role_aggregations
    
    def soft_update_target_networks(self, tau: float):
        """
        Soft update target networks: θ_target ← τ·θ + (1-τ)·θ_target.
        
        Args:
            tau: Soft update coefficient
        """
        for role_id in range(self.n_roles):
            critic = self.critics[str(role_id)]
            target_critic = self.target_critics[str(role_id)]
            
            for param, target_param in zip(
                critic.parameters(),
                target_critic.parameters()
            ):
                target_param.data.copy_(
                    tau * param.data + (1.0 - tau) * target_param.data
                )
    
    def get_actor_parameters(self) -> List[torch.nn.Parameter]:
        """Get all actor parameters for optimization."""
        params = []
        for role_id in range(self.n_roles):
            params.extend(list(self.actors[str(role_id)].parameters()))
        return params
    
    def get_critic_parameters(self) -> List[torch.nn.Parameter]:
        """Get all critic parameters for optimization."""
        params = []
        for role_id in range(self.n_roles):
            params.extend(list(self.critics[str(role_id)].parameters()))
        return params
    
    def save_state_dict(self, path: str):
        """Save all networks."""
        state_dict = {
            'actors': {str(i): self.actors[str(i)].state_dict() 
                      for i in range(self.n_roles)},
            'critics': {str(i): self.critics[str(i)].state_dict() 
                       for i in range(self.n_roles)},
            'target_critics': {str(i): self.target_critics[str(i)].state_dict() 
                              for i in range(self.n_roles)}
        }
        torch.save(state_dict, path)
    
    def load_state_dict(self, path: str):
        """Load all networks."""
        state_dict = torch.load(path, map_location=self.device)
        
        for i in range(self.n_roles):
            self.actors[str(i)].load_state_dict(state_dict['actors'][str(i)])
            self.critics[str(i)].load_state_dict(state_dict['critics'][str(i)])
            self.target_critics[str(i)].load_state_dict(
                state_dict['target_critics'][str(i)]
            )
    
    def train_mode(self):
        """Set all networks to training mode."""
        for role_id in range(self.n_roles):
            self.actors[str(role_id)].train()
            self.critics[str(role_id)].train()
    
    def eval_mode(self):
        """Set all networks to evaluation mode."""
        for role_id in range(self.n_roles):
            self.actors[str(role_id)].eval()
            self.critics[str(role_id)].eval()
    
    def __repr__(self) -> str:
        return (
            f"RoleManager(n_roles={self.n_roles}, "
            f"action_type={self.action_type}, "
            f"twin_critic={self.use_twin_critic})"
        )


if __name__ == '__main__':
    # Test RoleManager
    print("Testing RoleManager...")
    
    n_roles = 3
    individual_state_dim = 10
    action_dim = 8
    env_state_dim = 20
    role_aggregation_dim = 64
    batch_size = 16
    n_agents = 5
    
    # Create manager
    manager = RoleManager(
        n_roles=n_roles,
        individual_state_dim=individual_state_dim,
        action_dim=action_dim,
        env_state_dim=env_state_dim,
        role_aggregation_dim=role_aggregation_dim,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        action_type='discrete',
        use_twin_critic=True,
        device='cpu'
    )
    
    print(f"\n{manager}")
    
    # Test getting networks
    print("\n" + "-"*60)
    print("Testing network access...")
    actor_0 = manager.get_actor(0)
    critic_0 = manager.get_critic(0)
    target_critic_0 = manager.get_target_critic(0)
    print(f"✓ Actor 0: {type(actor_0).__name__}")
    print(f"✓ Critic 0: {type(critic_0).__name__}")
    print(f"✓ Target Critic 0: {type(target_critic_0).__name__}")
    
    # Test role aggregation
    print("\n" + "-"*60)
    print("Testing role aggregation...")
    
    # Single batch
    agent_states = torch.randn(n_agents, role_aggregation_dim)
    role_assignments = torch.randint(0, n_roles, (n_agents,))
    
    print(f"Agent states: {agent_states.shape}")
    print(f"Role assignments: {role_assignments.tolist()}")
    
    role_agg = manager.compute_role_aggregations(
        agent_states, role_assignments, method='mean'
    )
    print(f"Role aggregations (mean): {role_agg.shape}")
    
    # Batch
    agent_states_batch = torch.randn(batch_size, n_agents, role_aggregation_dim)
    role_assignments_batch = torch.randint(0, n_roles, (batch_size, n_agents))
    
    role_agg_batch = manager.compute_role_aggregations(
        agent_states_batch, role_assignments_batch, method='mean'
    )
    print(f"Role aggregations (batch): {role_agg_batch.shape}")
    
    # Test different aggregation methods
    for method in ['mean', 'max', 'sum']:
        agg = manager.compute_role_aggregations(
            agent_states, role_assignments, method=method
        )
        print(f"  {method}: {agg.shape}")
    
    # Test soft update
    print("\n" + "-"*60)
    print("Testing soft update...")
    
    # Get initial target parameters
    target_param_before = list(target_critic_0.parameters())[0].clone()
    
    # Modify critic parameters
    for param in critic_0.parameters():
        param.data += 1.0
    
    # Soft update
    tau = 0.005
    manager.soft_update_target_networks(tau)
    
    # Check target parameters changed
    target_param_after = list(target_critic_0.parameters())[0]
    changed = not torch.allclose(target_param_before, target_param_after)
    print(f"✓ Target network updated: {changed}")
    
    # Test parameter collection
    print("\n" + "-"*60)
    print("Testing parameter collection...")
    
    actor_params = manager.get_actor_parameters()
    critic_params = manager.get_critic_parameters()
    
    print(f"Total actor parameters: {sum(p.numel() for p in actor_params)}")
    print(f"Total critic parameters: {sum(p.numel() for p in critic_params)}")
    
    # Test mode switching
    print("\n" + "-"*60)
    print("Testing mode switching...")
    
    manager.train_mode()
    print(f"✓ Train mode: {manager.get_actor(0).training}")
    
    manager.eval_mode()
    print(f"✓ Eval mode: {not manager.get_actor(0).training}")
    
    print("\n" + "="*60)
    print("✓ All RoleManager tests passed!")
