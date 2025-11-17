"""
Vector Quantization Role Assignment Module Ψ.

Implements dynamic role assignment via VQ:
    c_k(t) = Ψ(s_k) = argmin_j ||E_φ(s_k) - e_j||^2

Uses Straight-Through Estimator (STE) for end-to-end training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np

from .encoder import StateEncoder


class VQRoleModule(nn.Module):
    """
    Vector Quantization Role Assignment Module.
    
    Components:
    1. State encoder E_φ: s_k → z_k ∈ R^D
    2. Role codebook E ∈ R^{M×D}: M role prototype vectors
    3. VQ operation: argmin_j ||z_k - e_j||^2
    
    Training:
    - Uses Straight-Through Estimator for gradients
    - Codebook loss: ||sg(z) - e||^2
    - Commitment loss: β·||z - sg(e)||^2
    """
    
    def __init__(
        self,
        n_roles: int,
        embedding_dim: int,
        encoder: StateEncoder,
        commitment_cost: float = 0.25,
        use_ema: bool = False,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        """
        Initialize VQ role module.
        
        Args:
            n_roles: Number of roles M
            embedding_dim: Dimension D of embeddings
            encoder: State encoder E_φ
            commitment_cost: Commitment loss weight β
            use_ema: Use EMA updates for codebook (alternative to gradient)
            ema_decay: EMA decay rate
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        
        self.n_roles = n_roles
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        
        # State encoder E_φ
        self.encoder = encoder
        
        # Role codebook E ∈ R^{M×D}
        self.codebook = nn.Embedding(n_roles, embedding_dim)
        self._initialize_codebook()
        
        # EMA statistics (if using EMA updates)
        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(n_roles))
            self.register_buffer('ema_embed_avg', self.codebook.weight.data.clone())
        
        # Usage tracking
        self.register_buffer('codebook_usage', torch.zeros(n_roles))
        self.register_buffer('total_assignments', torch.tensor(0))
    
    def _initialize_codebook(self):
        """Initialize codebook with uniform distribution."""
        nn.init.uniform_(self.codebook.weight, -1.0 / self.n_roles, 1.0 / self.n_roles)
    
    def forward(
        self,
        state: torch.Tensor,
        compute_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with VQ role assignment.
        
        Args:
            state: Agent state(s) of shape (batch_size, state_dim) or (state_dim,)
            compute_loss: Whether to compute VQ losses
        
        Returns:
            role_ids: Assigned role indices of shape (batch_size,)
            embeddings: Encoder output z_k of shape (batch_size, embedding_dim)
            quantized_embeddings: Codebook vectors e_{c_k} (with STE)
            losses: Dict with 'codebook_loss' and 'commitment_loss' (if compute_loss=True)
        """
        # Encode state: s_k → z_k
        embeddings = self.encoder(state)
        
        # Handle single state case
        is_single = (embeddings.dim() == 1)
        if is_single:
            embeddings = embeddings.unsqueeze(0)
        
        # Vector quantization: find nearest codebook vector
        role_ids, quantized_embeddings = self._vector_quantize(embeddings)
        
        # Straight-Through Estimator: copy gradients from quantized to embeddings
        quantized_embeddings_ste = embeddings + (quantized_embeddings - embeddings).detach()
        
        # Compute losses
        losses = None
        if compute_loss:
            losses = self._compute_vq_losses(embeddings, quantized_embeddings)
        
        # Update usage statistics
        self._update_usage_stats(role_ids)
        
        # Handle single state case
        if is_single:
            role_ids = role_ids.squeeze(0)
            embeddings = embeddings.squeeze(0)
            quantized_embeddings_ste = quantized_embeddings_ste.squeeze(0)
        
        return role_ids, embeddings, quantized_embeddings_ste, losses
    
    def _vector_quantize(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform vector quantization: argmin_j ||z - e_j||^2.
        
        Args:
            embeddings: Input embeddings of shape (batch_size, embedding_dim)
        
        Returns:
            role_ids: Nearest codebook indices
            quantized: Nearest codebook vectors
        """
        batch_size = embeddings.shape[0]
        
        # Compute distances: ||z - e_j||^2 = ||z||^2 + ||e_j||^2 - 2<z, e_j>
        # Shape: (batch_size, n_roles)
        distances = (
            torch.sum(embeddings**2, dim=1, keepdim=True) +
            torch.sum(self.codebook.weight**2, dim=1) -
            2 * torch.matmul(embeddings, self.codebook.weight.t())
        )
        
        # Find nearest codebook vector
        role_ids = torch.argmin(distances, dim=1)
        
        # Get quantized embeddings
        quantized = self.codebook(role_ids)
        
        return role_ids, quantized
    
    def _compute_vq_losses(
        self,
        embeddings: torch.Tensor,
        quantized: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VQ-VAE losses.
        
        Codebook loss: L_codebook = ||sg(z) - e||^2
        Commitment loss: L_commit = β·||z - sg(e)||^2
        
        Args:
            embeddings: Encoder outputs z
            quantized: Codebook vectors e
        
        Returns:
            Dictionary with codebook and commitment losses
        """
        # Codebook loss: move codebook towards embeddings
        # Gradients only flow to codebook
        codebook_loss = F.mse_loss(quantized, embeddings.detach())
        
        # Commitment loss: encourage encoder to commit to codebook
        # Gradients only flow to encoder
        commitment_loss = self.commitment_cost * F.mse_loss(
            embeddings, quantized.detach()
        )
        
        return {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'total_vq_loss': codebook_loss + commitment_loss
        }
    
    def _update_usage_stats(self, role_ids: torch.Tensor):
        """Update codebook usage statistics."""
        with torch.no_grad():
            # Count role assignments
            for role_id in role_ids:
                self.codebook_usage[role_id] += 1
            self.total_assignments += role_ids.numel()
    
    def get_role_assignment(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get role assignment without computing losses (inference mode).
        
        Args:
            state: Agent state(s)
        
        Returns:
            role_ids: Assigned role indices
        """
        with torch.no_grad():
            role_ids, _, _, _ = self.forward(state, compute_loss=False)
        return role_ids
    
    def get_codebook_usage(self) -> np.ndarray:
        """
        Get normalized codebook usage statistics.
        
        Returns:
            Array of shape (n_roles,) with usage percentages
        """
        if self.total_assignments == 0:
            return np.zeros(self.n_roles)
        
        usage = self.codebook_usage.cpu().numpy()
        total = self.total_assignments.item()
        return usage / total
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.codebook_usage.zero_()
        self.total_assignments.zero_()
    
    def ema_update(self, embeddings: torch.Tensor, role_ids: torch.Tensor):
        """
        Update codebook using Exponential Moving Average.
        
        Alternative to gradient-based updates.
        
        Args:
            embeddings: Encoder outputs
            role_ids: Assigned roles
        """
        if not self.use_ema:
            return
        
        with torch.no_grad():
            # One-hot encoding
            encodings = F.one_hot(role_ids, self.n_roles).float()
            
            # Update cluster sizes
            cluster_size = encodings.sum(0)
            self.ema_cluster_size = (
                self.ema_decay * self.ema_cluster_size +
                (1 - self.ema_decay) * cluster_size
            )
            
            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            cluster_size_smoothed = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.n_roles * self.epsilon) * n
            )
            
            # Update embeddings
            embed_sum = torch.matmul(encodings.t(), embeddings)
            self.ema_embed_avg = (
                self.ema_decay * self.ema_embed_avg +
                (1 - self.ema_decay) * embed_sum
            )
            
            # Update codebook
            self.codebook.weight.data = (
                self.ema_embed_avg / cluster_size_smoothed.unsqueeze(1)
            )
    
    def get_codebook_vectors(self) -> torch.Tensor:
        """
        Get all codebook vectors.
        
        Returns:
            Codebook of shape (n_roles, embedding_dim)
        """
        return self.codebook.weight.data
    
    def __repr__(self) -> str:
        usage = self.get_codebook_usage()
        return (
            f"VQRoleModule(n_roles={self.n_roles}, "
            f"embedding_dim={self.embedding_dim}, "
            f"usage={usage.tolist()})"
        )


if __name__ == '__main__':
    # Test VQ module
    print("Testing VQRoleModule...")
    
    state_dim = 10
    embedding_dim = 64
    n_roles = 3
    batch_size = 32
    
    # Create encoder and VQ module
    encoder = StateEncoder(state_dim, embedding_dim, [128, 128])
    vq_module = VQRoleModule(
        n_roles=n_roles,
        embedding_dim=embedding_dim,
        encoder=encoder,
        commitment_cost=0.25
    )
    
    print(f"VQ Module: {vq_module}")
    
    # Test single state
    state = torch.randn(state_dim)
    role_id, emb, quant_emb, losses = vq_module(state)
    print(f"\nSingle state:")
    print(f"  State: {state.shape}")
    print(f"  Role ID: {role_id.item()}")
    print(f"  Embedding: {emb.shape}")
    print(f"  Losses: {losses}")
    
    # Test batch
    states = torch.randn(batch_size, state_dim)
    role_ids, embs, quant_embs, losses = vq_module(states)
    print(f"\nBatch:")
    print(f"  States: {states.shape}")
    print(f"  Role IDs: {role_ids.shape}, unique: {torch.unique(role_ids).tolist()}")
    print(f"  Embeddings: {embs.shape}")
    print(f"  Losses: {losses}")
    
    # Test usage statistics
    print(f"\nCodebook usage: {vq_module.get_codebook_usage()}")
    
    # Test gradient flow (STE)
    print("\nTesting gradient flow...")
    states = torch.randn(10, state_dim, requires_grad=True)
    role_ids, embs, quant_embs, losses = vq_module(states)
    loss = losses['total_vq_loss']
    loss.backward()
    print(f"  Gradient on states: {states.grad is not None}")
    print(f"  Gradient norm: {states.grad.norm().item():.6f}")
    
    print("\n✓ All tests passed!")
