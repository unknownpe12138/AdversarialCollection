"""
Multi-Agent Replay Buffer for VQ-HC-SAC.

Stores transitions: (s, a, r', s', done, role_assignments)
where r' is the shaped reward from PBRS.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
import random


class MultiAgentReplayBuffer:
    """
    Experience replay buffer for multi-agent systems.
    
    Stores:
    - states: (batch, n_agents, state_dim)
    - actions: (batch, n_agents) for discrete or (batch, n_agents, action_dim) for continuous
    - shaped_rewards: (batch, n_agents) - PBRS shaped rewards
    - next_states: (batch, n_agents, state_dim)
    - dones: (batch,) - episode termination
    - role_assignments: (batch, n_agents) - assigned roles
    - env_states: (batch, env_state_dim) - global environment state
    """
    
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        individual_state_dim: int,
        env_state_dim: int,
        action_dim: int,
        action_type: str = 'discrete',
        device: str = 'cpu'
    ):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            n_agents: Number of agents
            individual_state_dim: Dimension of individual agent state
            env_state_dim: Dimension of environment state
            action_dim: Dimension of action space
            action_type: 'discrete' or 'continuous'
            device: Device for tensors
        """
        self.capacity = capacity
        self.n_agents = n_agents
        self.individual_state_dim = individual_state_dim
        self.env_state_dim = env_state_dim
        self.action_dim = action_dim
        self.action_type = action_type
        self.device = device
        
        # Storage
        self.individual_states = np.zeros(
            (capacity, n_agents, individual_state_dim), dtype=np.float32
        )
        self.env_states = np.zeros(
            (capacity, env_state_dim), dtype=np.float32
        )
        
        if action_type == 'discrete':
            self.actions = np.zeros((capacity, n_agents), dtype=np.int64)
        else:
            self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        
        self.shaped_rewards = np.zeros((capacity, n_agents), dtype=np.float32)
        self.next_individual_states = np.zeros(
            (capacity, n_agents, individual_state_dim), dtype=np.float32
        )
        self.next_env_states = np.zeros(
            (capacity, env_state_dim), dtype=np.float32
        )
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.role_assignments = np.zeros((capacity, n_agents), dtype=np.int64)
        
        # Pointers
        self.position = 0
        self.size = 0
    
    def add(
        self,
        individual_states: np.ndarray,
        env_state: np.ndarray,
        actions: np.ndarray,
        shaped_rewards: np.ndarray,
        next_individual_states: np.ndarray,
        next_env_state: np.ndarray,
        done: bool,
        role_assignments: np.ndarray
    ):
        """
        Add a transition to the buffer.
        
        Args:
            individual_states: (n_agents, state_dim)
            env_state: (env_state_dim,)
            actions: (n_agents,) or (n_agents, action_dim)
            shaped_rewards: (n_agents,) - PBRS shaped rewards
            next_individual_states: (n_agents, state_dim)
            next_env_state: (env_state_dim,)
            done: bool - episode termination
            role_assignments: (n_agents,) - assigned roles
        """
        # Store transition
        self.individual_states[self.position] = individual_states
        self.env_states[self.position] = env_state
        self.actions[self.position] = actions
        self.shaped_rewards[self.position] = shaped_rewards
        self.next_individual_states[self.position] = next_individual_states
        self.next_env_states[self.position] = next_env_state
        self.dones[self.position] = float(done)
        self.role_assignments[self.position] = role_assignments
        
        # Update pointers
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Dictionary containing batched transitions as torch tensors
        """
        # Sample indices
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # Gather batch
        batch = {
            'individual_states': torch.FloatTensor(
                self.individual_states[indices]
            ).to(self.device),
            'env_states': torch.FloatTensor(
                self.env_states[indices]
            ).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device)
                      if self.action_type == 'discrete'
                      else torch.FloatTensor(self.actions[indices]).to(self.device),
            'shaped_rewards': torch.FloatTensor(
                self.shaped_rewards[indices]
            ).to(self.device),
            'next_individual_states': torch.FloatTensor(
                self.next_individual_states[indices]
            ).to(self.device),
            'next_env_states': torch.FloatTensor(
                self.next_env_states[indices]
            ).to(self.device),
            'dones': torch.FloatTensor(
                self.dones[indices]
            ).to(self.device),
            'role_assignments': torch.LongTensor(
                self.role_assignments[indices]
            ).to(self.device)
        }
        
        return batch
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= min_size
    
    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if self.size == 0:
            return {
                'size': 0,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'mean_episode_length': 0.0
            }
        
        valid_rewards = self.shaped_rewards[:self.size]
        
        return {
            'size': self.size,
            'capacity_used': self.size / self.capacity,
            'mean_reward': float(np.mean(valid_rewards)),
            'std_reward': float(np.std(valid_rewards)),
            'min_reward': float(np.min(valid_rewards)),
            'max_reward': float(np.max(valid_rewards))
        }
    
    def __repr__(self) -> str:
        return (
            f"MultiAgentReplayBuffer(size={self.size}/{self.capacity}, "
            f"n_agents={self.n_agents}, "
            f"action_type={self.action_type})"
        )


class PrioritizedReplayBuffer(MultiAgentReplayBuffer):
    """
    Prioritized Experience Replay (PER) for multi-agent systems.
    
    Samples transitions based on TD error: P(i) ∝ |δ_i|^α
    
    Optional: Can be used to improve sample efficiency.
    """
    
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        individual_state_dim: int,
        env_state_dim: int,
        action_dim: int,
        action_type: str = 'discrete',
        device: str = 'cpu',
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames to anneal beta to 1.0
        """
        super().__init__(
            capacity, n_agents, individual_state_dim, env_state_dim,
            action_dim, action_type, device
        )
        
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        # Priority tree (simple array implementation)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        individual_states: np.ndarray,
        env_state: np.ndarray,
        actions: np.ndarray,
        shaped_rewards: np.ndarray,
        next_individual_states: np.ndarray,
        next_env_state: np.ndarray,
        done: bool,
        role_assignments: np.ndarray
    ):
        """Add transition with maximum priority."""
        super().add(
            individual_states, env_state, actions, shaped_rewards,
            next_individual_states, next_env_state, done, role_assignments
        )
        
        # Assign maximum priority to new transition
        self.priorities[self.position - 1] = self.max_priority
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample batch with prioritized sampling.
        
        Returns:
            batch: Dictionary of transitions
            indices: Sampled indices (for updating priorities)
            weights: Importance sampling weights
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        # Gather batch
        batch = super().sample(batch_size)  # This won't work, need to fix
        
        # Manual gathering
        batch = {
            'individual_states': torch.FloatTensor(
                self.individual_states[indices]
            ).to(self.device),
            'env_states': torch.FloatTensor(
                self.env_states[indices]
            ).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device)
                      if self.action_type == 'discrete'
                      else torch.FloatTensor(self.actions[indices]).to(self.device),
            'shaped_rewards': torch.FloatTensor(
                self.shaped_rewards[indices]
            ).to(self.device),
            'next_individual_states': torch.FloatTensor(
                self.next_individual_states[indices]
            ).to(self.device),
            'next_env_states': torch.FloatTensor(
                self.next_env_states[indices]
            ).to(self.device),
            'dones': torch.FloatTensor(
                self.dones[indices]
            ).to(self.device),
            'role_assignments': torch.LongTensor(
                self.role_assignments[indices]
            ).to(self.device)
        }
        
        self.frame += 1
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


if __name__ == '__main__':
    # Test replay buffer
    print("Testing MultiAgentReplayBuffer...")
    
    capacity = 1000
    n_agents = 5
    individual_state_dim = 10
    env_state_dim = 20
    action_dim = 8
    batch_size = 32
    
    # Create buffer
    buffer = MultiAgentReplayBuffer(
        capacity=capacity,
        n_agents=n_agents,
        individual_state_dim=individual_state_dim,
        env_state_dim=env_state_dim,
        action_dim=action_dim,
        action_type='discrete',
        device='cpu'
    )
    
    print(f"{buffer}")
    
    # Add transitions
    print("\n" + "-"*60)
    print("Adding transitions...")
    
    for i in range(100):
        individual_states = np.random.randn(n_agents, individual_state_dim).astype(np.float32)
        env_state = np.random.randn(env_state_dim).astype(np.float32)
        actions = np.random.randint(0, action_dim, size=n_agents)
        shaped_rewards = np.random.randn(n_agents).astype(np.float32)
        next_individual_states = np.random.randn(n_agents, individual_state_dim).astype(np.float32)
        next_env_state = np.random.randn(env_state_dim).astype(np.float32)
        done = (i % 20 == 19)  # Episode ends every 20 steps
        role_assignments = np.random.randint(0, 3, size=n_agents)
        
        buffer.add(
            individual_states, env_state, actions, shaped_rewards,
            next_individual_states, next_env_state, done, role_assignments
        )
    
    print(f"✓ Added 100 transitions")
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Ready for training: {buffer.is_ready(batch_size)}")
    
    # Sample batch
    print("\n" + "-"*60)
    print("Sampling batch...")
    
    batch = buffer.sample(batch_size)
    
    print(f"Batch keys: {list(batch.keys())}")
    print(f"  individual_states: {batch['individual_states'].shape}")
    print(f"  env_states: {batch['env_states'].shape}")
    print(f"  actions: {batch['actions'].shape}")
    print(f"  shaped_rewards: {batch['shaped_rewards'].shape}")
    print(f"  next_individual_states: {batch['next_individual_states'].shape}")
    print(f"  next_env_states: {batch['next_env_states'].shape}")
    print(f"  dones: {batch['dones'].shape}")
    print(f"  role_assignments: {batch['role_assignments'].shape}")
    
    # Check data types
    print("\n" + "-"*60)
    print("Checking data types...")
    print(f"  individual_states: {batch['individual_states'].dtype}")
    print(f"  actions: {batch['actions'].dtype}")
    print(f"  shaped_rewards: {batch['shaped_rewards'].dtype}")
    print(f"  dones: {batch['dones'].dtype}")
    
    # Get statistics
    print("\n" + "-"*60)
    print("Buffer statistics:")
    stats = buffer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Test buffer overflow
    print("\n" + "-"*60)
    print("Testing buffer overflow...")
    
    for i in range(1000):
        individual_states = np.random.randn(n_agents, individual_state_dim).astype(np.float32)
        env_state = np.random.randn(env_state_dim).astype(np.float32)
        actions = np.random.randint(0, action_dim, size=n_agents)
        shaped_rewards = np.random.randn(n_agents).astype(np.float32)
        next_individual_states = np.random.randn(n_agents, individual_state_dim).astype(np.float32)
        next_env_state = np.random.randn(env_state_dim).astype(np.float32)
        done = False
        role_assignments = np.random.randint(0, 3, size=n_agents)
        
        buffer.add(
            individual_states, env_state, actions, shaped_rewards,
            next_individual_states, next_env_state, done, role_assignments
        )
    
    print(f"✓ Added 1000 more transitions")
    print(f"  Buffer size: {len(buffer)} (should be {capacity})")
    assert len(buffer) == capacity, "Buffer should be at capacity"
    
    # Test PrioritizedReplayBuffer
    print("\n" + "="*60)
    print("Testing PrioritizedReplayBuffer...")
    
    per_buffer = PrioritizedReplayBuffer(
        capacity=capacity,
        n_agents=n_agents,
        individual_state_dim=individual_state_dim,
        env_state_dim=env_state_dim,
        action_dim=action_dim,
        action_type='discrete',
        device='cpu',
        alpha=0.6,
        beta_start=0.4
    )
    
    # Add transitions
    for i in range(100):
        individual_states = np.random.randn(n_agents, individual_state_dim).astype(np.float32)
        env_state = np.random.randn(env_state_dim).astype(np.float32)
        actions = np.random.randint(0, action_dim, size=n_agents)
        shaped_rewards = np.random.randn(n_agents).astype(np.float32)
        next_individual_states = np.random.randn(n_agents, individual_state_dim).astype(np.float32)
        next_env_state = np.random.randn(env_state_dim).astype(np.float32)
        done = False
        role_assignments = np.random.randint(0, 3, size=n_agents)
        
        per_buffer.add(
            individual_states, env_state, actions, shaped_rewards,
            next_individual_states, next_env_state, done, role_assignments
        )
    
    # Sample with priorities
    batch, indices, weights = per_buffer.sample(batch_size)
    
    print(f"✓ Sampled with priorities")
    print(f"  Indices: {indices[:5]}")
    print(f"  Weights: {weights[:5]}")
    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    # Update priorities
    new_priorities = np.abs(np.random.randn(batch_size)) + 0.01
    per_buffer.update_priorities(indices, new_priorities)
    print(f"✓ Updated priorities")
    
    print("\n" + "="*60)
    print("✓ All ReplayBuffer tests passed!")
