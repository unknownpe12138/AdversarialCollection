"""
VQ-HC-SAC Agent: Inference interface for trained policies.

Provides a simple interface for using trained VQ-HC-SAC models.
"""

import torch
import numpy as np
from typing import Union, Dict, Any
from pathlib import Path

from .config import VQHCSACConfig
from .networks import StateEncoder, VQRoleModule
from .components import RoleManager


class VQHCSACAgent:
    """
    Agent interface for using trained VQ-HC-SAC policies.
    
    Usage:
        # Load trained agent
        agent = VQHCSACAgent.load('checkpoints/best_model.pt')
        
        # Select action
        action = agent.select_action(state, deterministic=True)
        
        # Get role assignment
        role_id = agent.get_role(state)
    """
    
    def __init__(
        self,
        vq_module: VQRoleModule,
        role_manager: RoleManager,
        config: VQHCSACConfig,
        device: str = 'cpu'
    ):
        """
        Initialize agent.
        
        Args:
            vq_module: Trained VQ role module
            role_manager: Trained role manager with actors
            config: Training configuration
            device: Device for inference
        """
        self.vq_module = vq_module
        self.role_manager = role_manager
        self.config = config
        self.device = torch.device(device)
        
        # Set to evaluation mode
        self.vq_module.eval()
        self.role_manager.eval_mode()
    
    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True
    ) -> Union[int, np.ndarray]:
        """
        Select action for a single agent.
        
        Args:
            state: Agent state
            deterministic: If True, use deterministic policy (no exploration)
        
        Returns:
            action: Selected action (int for discrete, array for continuous)
        """
        with torch.no_grad():
            # Convert to tensor
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            
            # Ensure correct shape
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # Assign role
            role_id = self.vq_module.get_role_assignment(state)
            
            # Get actor for this role
            actor = self.role_manager.get_actor(role_id.item())
            
            # Sample action
            action, _ = actor.sample(state.squeeze(0), deterministic=deterministic)
            
            # Convert to numpy if needed
            if isinstance(action, torch.Tensor):
                if action.dim() == 0:
                    action = action.item()
                else:
                    action = action.cpu().numpy()
            
            return action
    
    def select_actions(
        self,
        states: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True
    ) -> np.ndarray:
        """
        Select actions for multiple agents.
        
        Args:
            states: Agent states of shape (n_agents, state_dim)
            deterministic: If True, use deterministic policy
        
        Returns:
            actions: Selected actions of shape (n_agents,) or (n_agents, action_dim)
        """
        with torch.no_grad():
            # Convert to tensor
            if isinstance(states, np.ndarray):
                states = torch.FloatTensor(states).to(self.device)
            
            n_agents = states.shape[0]
            actions = []
            
            for i in range(n_agents):
                action = self.select_action(states[i], deterministic=deterministic)
                actions.append(action)
            
            return np.array(actions)
    
    def get_role(self, state: Union[np.ndarray, torch.Tensor]) -> int:
        """
        Get role assignment for a state.
        
        Args:
            state: Agent state
        
        Returns:
            role_id: Assigned role ID
        """
        with torch.no_grad():
            # Convert to tensor
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            
            # Assign role
            role_id = self.vq_module.get_role_assignment(state)
            
            return role_id.item()
    
    def get_roles(self, states: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get role assignments for multiple states.
        
        Args:
            states: Agent states of shape (n_agents, state_dim)
        
        Returns:
            role_ids: Assigned role IDs of shape (n_agents,)
        """
        with torch.no_grad():
            # Convert to tensor
            if isinstance(states, np.ndarray):
                states = torch.FloatTensor(states).to(self.device)
            
            n_agents = states.shape[0]
            role_ids = []
            
            for i in range(n_agents):
                role_id = self.get_role(states[i])
                role_ids.append(role_id)
            
            return np.array(role_ids)
    
    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        device: str = 'cpu'
    ) -> 'VQHCSACAgent':
        """
        Load trained agent from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device for inference ('cpu' or 'cuda')
        
        Returns:
            Loaded agent
        """
        device_obj = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
        
        # Load configuration
        config = VQHCSACConfig.from_dict(checkpoint['config'])
        
        # Extract dimensions from checkpoint
        # (This is a simplified version; might need adjustment based on actual checkpoint)
        example_actor = checkpoint['role_manager']['actors'][0]
        state_dim = None
        action_dim = None
        
        for key in example_actor.keys():
            if 'feature_extractor.0.weight' in key:
                state_dim = example_actor[key].shape[1]
            if 'action_head.weight' in key:
                action_dim = example_actor[key].shape[0]
        
        if state_dim is None or action_dim is None:
            raise ValueError("Could not extract dimensions from checkpoint")
        
        # Create networks
        encoder = StateEncoder(
            input_dim=state_dim,
            embedding_dim=config.embedding_dim,
            hidden_dims=config.encoder_hidden_dims
        ).to(device_obj)
        
        vq_module = VQRoleModule(
            n_roles=config.n_roles,
            embedding_dim=config.embedding_dim,
            encoder=encoder,
            commitment_cost=config.vq_beta
        ).to(device_obj)
        
        # Load VQ module state
        encoder.load_state_dict(checkpoint['encoder'])
        vq_module.load_state_dict(checkpoint['vq_module'])
        
        # Create role manager (simplified, assuming known dimensions)
        role_manager = RoleManager(
            n_roles=config.n_roles,
            individual_state_dim=state_dim,
            action_dim=action_dim,
            env_state_dim=state_dim * 5,  # Placeholder
            role_aggregation_dim=config.embedding_dim,
            actor_hidden_dims=config.actor_hidden_dims,
            critic_hidden_dims=config.critic_hidden_dims,
            action_type='discrete',
            use_twin_critic=config.use_double_critic,
            device=device
        )
        
        # Load actor states
        for i in range(config.n_roles):
            role_manager.get_actor(i).load_state_dict(
                checkpoint['role_manager']['actors'][i]
            )
        
        # Create agent
        agent = cls(vq_module, role_manager, config, device)
        
        print(f"Agent loaded from: {checkpoint_path}")
        print(f"  Roles: {config.n_roles}")
        print(f"  Device: {device}")
        
        return agent
    
    def to(self, device: str) -> 'VQHCSACAgent':
        """Move agent to device."""
        self.device = torch.device(device)
        self.vq_module.to(self.device)
        # Role manager networks are already on device
        return self


if __name__ == '__main__':
    # Test agent interface
    print("Testing VQHCSACAgent...")
    print("="*60)
    
    # This would normally load a trained checkpoint
    # For testing, we'll create a dummy agent
    
    from .config import VQHCSACConfig
    
    config = VQHCSACConfig(n_roles=3, embedding_dim=64)
    
    state_dim = 10
    action_dim = 8
    
    # Create dummy networks
    encoder = StateEncoder(state_dim, config.embedding_dim, [128, 128])
    vq_module = VQRoleModule(config.n_roles, config.embedding_dim, encoder)
    
    role_manager = RoleManager(
        n_roles=config.n_roles,
        individual_state_dim=state_dim,
        action_dim=action_dim,
        env_state_dim=50,
        role_aggregation_dim=config.embedding_dim,
        action_type='discrete'
    )
    
    agent = VQHCSACAgent(vq_module, role_manager, config, device='cpu')
    
    print(f"✓ Agent created")
    
    # Test single action selection
    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(state, deterministic=True)
    print(f"\n✓ Single action: {action}")
    
    # Test multiple actions
    states = np.random.randn(5, state_dim).astype(np.float32)
    actions = agent.select_actions(states, deterministic=True)
    print(f"✓ Multiple actions: {actions}")
    
    # Test role assignment
    role_id = agent.get_role(state)
    print(f"\n✓ Single role: {role_id}")
    
    role_ids = agent.get_roles(states)
    print(f"✓ Multiple roles: {role_ids}")
    
    print("\n" + "="*60)
    print("✓ All agent tests passed!")
