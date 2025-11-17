"""
State Encoder E_φ: Maps agent state to low-dimensional embedding.

Transforms individual state s_k = (position, carried_utility, is_alive)
into embedding z_k ∈ R^D for role assignment.
"""

import torch
import torch.nn as nn
from typing import List


class StateEncoder(nn.Module):
    """
    State encoder network E_φ.
    
    Architecture:
        s_k → [Linear → ReLU]* → Linear → z_k
    
    Input: Individual agent state
    Output: Low-dimensional embedding for VQ role assignment
    """
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dims: List[int] = [128, 128],
        activation: str = 'relu',
        layer_norm: bool = True
    ):
        """
        Initialize state encoder.
        
        Args:
            input_dim: Dimension of input state s_k
            embedding_dim: Dimension D of output embedding z_k
            hidden_dims: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
            layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
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
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: s_k → z_k.
        
        Args:
            state: Agent state tensor of shape (batch_size, input_dim)
                   or (input_dim,) for single agent
        
        Returns:
            embedding: Embedding tensor of shape (batch_size, embedding_dim)
                      or (embedding_dim,)
        """
        return self.network(state)
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


class MultiAgentStateEncoder(nn.Module):
    """
    Wrapper for encoding multiple agents simultaneously.
    
    Processes N agent states in parallel using shared encoder.
    """
    
    def __init__(self, encoder: StateEncoder):
        """
        Initialize multi-agent encoder.
        
        Args:
            encoder: Single-agent state encoder to share
        """
        super().__init__()
        self.encoder = encoder
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encode multiple agent states.
        
        Args:
            states: Tensor of shape (batch_size, n_agents, state_dim)
        
        Returns:
            embeddings: Tensor of shape (batch_size, n_agents, embedding_dim)
        """
        batch_size, n_agents, state_dim = states.shape
        
        # Flatten batch and agent dimensions
        states_flat = states.view(batch_size * n_agents, state_dim)
        
        # Encode all states
        embeddings_flat = self.encoder(states_flat)
        
        # Reshape back
        embedding_dim = embeddings_flat.shape[-1]
        embeddings = embeddings_flat.view(batch_size, n_agents, embedding_dim)
        
        return embeddings


if __name__ == '__main__':
    # Test encoder
    print("Testing StateEncoder...")
    
    input_dim = 10  # Example: agent observation dimension
    embedding_dim = 64
    batch_size = 32
    
    encoder = StateEncoder(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dims=[128, 128]
    )
    
    # Test single state
    state = torch.randn(input_dim)
    embedding = encoder(state)
    print(f"Single state: {state.shape} → {embedding.shape}")
    assert embedding.shape == (embedding_dim,)
    
    # Test batch
    states = torch.randn(batch_size, input_dim)
    embeddings = encoder(states)
    print(f"Batch states: {states.shape} → {embeddings.shape}")
    assert embeddings.shape == (batch_size, embedding_dim)
    
    # Test multi-agent encoder
    print("\nTesting MultiAgentStateEncoder...")
    n_agents = 5
    multi_encoder = MultiAgentStateEncoder(encoder)
    
    states = torch.randn(batch_size, n_agents, input_dim)
    embeddings = multi_encoder(states)
    print(f"Multi-agent: {states.shape} → {embeddings.shape}")
    assert embeddings.shape == (batch_size, n_agents, embedding_dim)
    
    print("\n✓ All tests passed!")
