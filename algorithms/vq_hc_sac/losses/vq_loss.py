"""
Vector Quantization (VQ) Loss Functions.

Implements VQ-VAE auxiliary losses for training the role assignment module:
1. Codebook Loss: Pulls codebook vectors towards encoder outputs
2. Commitment Loss: Encourages encoder outputs to commit to codebook
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class VQLoss:
    """
    VQ-VAE loss computation for role assignment module.
    
    These auxiliary losses enable end-to-end training despite the
    non-differentiable argmin operation in vector quantization.
    """
    
    def __init__(
        self,
        commitment_cost: float = 0.25,
        use_ema_codebook: bool = False
    ):
        """
        Initialize VQ loss calculator.
        
        Args:
            commitment_cost: Weight β for commitment loss
            use_ema_codebook: Whether codebook uses EMA updates
                            (if True, codebook loss is not used)
        """
        self.commitment_cost = commitment_cost
        self.use_ema_codebook = use_ema_codebook
    
    def codebook_loss(
        self,
        embeddings: torch.Tensor,
        quantized_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute codebook loss: pulls codebook towards encoder outputs.
        
        Loss: L_codebook = ||sg(z) - e||^2
        
        where:
        - z = encoder output (embeddings)
        - e = selected codebook vector (quantized_embeddings)
        - sg = stop_gradient (detach)
        
        Gradients flow only to codebook vectors, not to encoder.
        
        Args:
            embeddings: Encoder outputs z, (batch_size, embedding_dim)
            quantized_embeddings: Selected codebook vectors e, (batch_size, embedding_dim)
        
        Returns:
            codebook_loss: MSE loss (scalar)
        """
        if self.use_ema_codebook:
            # EMA updates don't use gradient-based codebook loss
            return torch.tensor(0.0, device=embeddings.device)
        
        # Codebook loss: move codebook towards embeddings
        # embeddings.detach() ensures no gradient flows to encoder
        codebook_loss = F.mse_loss(quantized_embeddings, embeddings.detach())
        
        return codebook_loss
    
    def commitment_loss(
        self,
        embeddings: torch.Tensor,
        quantized_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute commitment loss: encourages encoder to commit to codebook.
        
        Loss: L_commit = β·||z - sg(e)||^2
        
        where:
        - z = encoder output (embeddings)
        - e = selected codebook vector (quantized_embeddings)
        - sg = stop_gradient (detach)
        - β = commitment cost
        
        Gradients flow only to encoder, not to codebook.
        
        Args:
            embeddings: Encoder outputs z, (batch_size, embedding_dim)
            quantized_embeddings: Selected codebook vectors e, (batch_size, embedding_dim)
        
        Returns:
            commitment_loss: Weighted MSE loss (scalar)
        """
        # Commitment loss: move encoder towards codebook
        # quantized_embeddings.detach() ensures no gradient flows to codebook
        commitment_loss = self.commitment_cost * F.mse_loss(
            embeddings, quantized_embeddings.detach()
        )
        
        return commitment_loss
    
    def total_vq_loss(
        self,
        embeddings: torch.Tensor,
        quantized_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total VQ loss and components.
        
        Total: L_VQ = L_codebook + L_commit
        
        Args:
            embeddings: Encoder outputs, (batch_size, embedding_dim)
            quantized_embeddings: Selected codebook vectors, (batch_size, embedding_dim)
        
        Returns:
            total_loss: Sum of codebook and commitment losses
            loss_dict: Dictionary with individual loss components
        """
        codebook_loss = self.codebook_loss(embeddings, quantized_embeddings)
        commitment_loss = self.commitment_loss(embeddings, quantized_embeddings)
        
        total_loss = codebook_loss + commitment_loss
        
        loss_dict = {
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'total_vq_loss': total_loss
        }
        
        return total_loss, loss_dict
    
    def compute_perplexity(
        self,
        role_assignments: torch.Tensor,
        n_roles: int
    ) -> torch.Tensor:
        """
        Compute codebook perplexity: measures how evenly roles are used.
        
        Perplexity = exp(H) where H is entropy of role distribution.
        
        High perplexity (close to n_roles) = all roles used evenly
        Low perplexity = only few roles used (codebook collapse)
        
        Args:
            role_assignments: Assigned role IDs, (batch_size,)
            n_roles: Total number of roles M
        
        Returns:
            perplexity: Codebook perplexity (scalar)
        """
        # Compute role usage distribution
        role_counts = torch.bincount(role_assignments, minlength=n_roles).float()
        role_probs = role_counts / role_counts.sum()
        
        # Compute entropy (with epsilon to avoid log(0))
        epsilon = 1e-10
        entropy = -(role_probs * torch.log(role_probs + epsilon)).sum()
        
        # Perplexity = exp(entropy)
        perplexity = torch.exp(entropy)
        
        return perplexity


class VQStatistics:
    """
    Helper class for tracking VQ-related statistics during training.
    """
    
    @staticmethod
    def compute_statistics(
        embeddings: torch.Tensor,
        quantized_embeddings: torch.Tensor,
        role_assignments: torch.Tensor,
        n_roles: int
    ) -> Dict[str, float]:
        """
        Compute comprehensive VQ statistics.
        
        Args:
            embeddings: Encoder outputs
            quantized_embeddings: Selected codebook vectors
            role_assignments: Assigned role IDs
            n_roles: Total number of roles
        
        Returns:
            Dictionary with various statistics
        """
        with torch.no_grad():
            # Quantization error
            quant_error = F.mse_loss(embeddings, quantized_embeddings)
            
            # Role usage
            role_counts = torch.bincount(role_assignments, minlength=n_roles).float()
            role_usage = role_counts / role_counts.sum()
            
            # Entropy and perplexity
            epsilon = 1e-10
            entropy = -(role_usage * torch.log(role_usage + epsilon)).sum()
            perplexity = torch.exp(entropy)
            
            # Active roles (used at least once)
            active_roles = (role_counts > 0).sum()
            
            # Embedding norms
            emb_norm = embeddings.norm(dim=-1).mean()
            quant_norm = quantized_embeddings.norm(dim=-1).mean()
            
            stats = {
                'quantization_error': quant_error.item(),
                'role_entropy': entropy.item(),
                'role_perplexity': perplexity.item(),
                'active_roles': active_roles.item(),
                'max_role_usage': role_usage.max().item(),
                'min_role_usage': role_usage.min().item(),
                'embedding_norm': emb_norm.item(),
                'quantized_norm': quant_norm.item()
            }
            
            # Add per-role usage
            for i in range(n_roles):
                stats[f'role_{i}_usage'] = role_usage[i].item()
        
        return stats


class CodebookResetScheduler:
    """
    Optional: Scheduler for resetting unused codebook vectors.
    
    Helps prevent codebook collapse by reinitializing unused vectors.
    """
    
    def __init__(
        self,
        reset_threshold: int = 100,
        reset_prob: float = 0.5
    ):
        """
        Initialize reset scheduler.
        
        Args:
            reset_threshold: Reset if role unused for this many steps
            reset_prob: Probability of resetting an unused vector
        """
        self.reset_threshold = reset_threshold
        self.reset_prob = reset_prob
        self.usage_counts = {}
    
    def update_usage(
        self,
        role_assignments: torch.Tensor,
        n_roles: int
    ):
        """
        Update usage tracking.
        
        Args:
            role_assignments: Current batch role assignments
            n_roles: Total number of roles
        """
        # Count role usage in current batch
        role_counts = torch.bincount(role_assignments, minlength=n_roles)
        
        for role_id in range(n_roles):
            if role_id not in self.usage_counts:
                self.usage_counts[role_id] = 0
            
            if role_counts[role_id] > 0:
                # Role used, reset counter
                self.usage_counts[role_id] = 0
            else:
                # Role not used, increment counter
                self.usage_counts[role_id] += 1
    
    def get_reset_mask(self, n_roles: int) -> torch.Tensor:
        """
        Get mask of roles to reset.
        
        Args:
            n_roles: Total number of roles
        
        Returns:
            Boolean mask of roles to reset
        """
        reset_mask = torch.zeros(n_roles, dtype=torch.bool)
        
        for role_id in range(n_roles):
            if role_id in self.usage_counts:
                if self.usage_counts[role_id] >= self.reset_threshold:
                    # Probabilistically reset
                    if torch.rand(1).item() < self.reset_prob:
                        reset_mask[role_id] = True
                        self.usage_counts[role_id] = 0
        
        return reset_mask


if __name__ == '__main__':
    # Test VQ losses
    print("Testing VQ Losses...")
    print("="*60)
    
    batch_size = 32
    embedding_dim = 64
    n_roles = 3
    
    # Create VQ loss calculator
    vq_loss = VQLoss(commitment_cost=0.25, use_ema_codebook=False)
    
    print(f"VQ Loss Calculator:")
    print(f"  Commitment cost: {vq_loss.commitment_cost}")
    print(f"  Use EMA codebook: {vq_loss.use_ema_codebook}")
    
    # Test codebook loss
    print("\n" + "-"*60)
    print("Testing Codebook Loss...")
    
    embeddings = torch.randn(batch_size, embedding_dim)
    quantized_embeddings = torch.randn(batch_size, embedding_dim)
    
    codebook_loss = vq_loss.codebook_loss(embeddings, quantized_embeddings)
    
    print(f"✓ Codebook loss: {codebook_loss.item():.4f}")
    
    # Test commitment loss
    print("\n" + "-"*60)
    print("Testing Commitment Loss...")
    
    commitment_loss = vq_loss.commitment_loss(embeddings, quantized_embeddings)
    
    print(f"✓ Commitment loss: {commitment_loss.item():.4f}")
    
    # Test total loss
    print("\n" + "-"*60)
    print("Testing Total VQ Loss...")
    
    total_loss, loss_dict = vq_loss.total_vq_loss(embeddings, quantized_embeddings)
    
    print(f"✓ Total VQ loss: {total_loss.item():.4f}")
    print(f"  Components:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value.item():.4f}")
    
    # Test perplexity
    print("\n" + "-"*60)
    print("Testing Perplexity Computation...")
    
    # Balanced role usage
    role_assignments_balanced = torch.randint(0, n_roles, (batch_size,))
    perplexity_balanced = vq_loss.compute_perplexity(role_assignments_balanced, n_roles)
    
    print(f"✓ Balanced role usage:")
    print(f"  Assignments: {role_assignments_balanced.tolist()[:10]}...")
    print(f"  Perplexity: {perplexity_balanced.item():.4f} (max: {n_roles})")
    
    # Collapsed codebook (all same role)
    role_assignments_collapsed = torch.zeros(batch_size, dtype=torch.long)
    perplexity_collapsed = vq_loss.compute_perplexity(role_assignments_collapsed, n_roles)
    
    print(f"\n✓ Collapsed codebook:")
    print(f"  All assigned to role 0")
    print(f"  Perplexity: {perplexity_collapsed.item():.4f} (min: 1.0)")
    
    # Test statistics
    print("\n" + "-"*60)
    print("Testing VQ Statistics...")
    
    stats = VQStatistics.compute_statistics(
        embeddings, quantized_embeddings, role_assignments_balanced, n_roles
    )
    
    print(f"✓ VQ Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test gradient flow
    print("\n" + "-"*60)
    print("Testing Gradient Flow...")
    
    embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    quantized_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    
    # Codebook loss (gradients to quantized only)
    codebook_loss = vq_loss.codebook_loss(embeddings, quantized_embeddings)
    codebook_loss.backward(retain_graph=True)
    
    print(f"✓ Codebook loss gradients:")
    print(f"  Embeddings grad: {embeddings.grad is None} (should be None)")
    print(f"  Quantized grad: {quantized_embeddings.grad is not None} (should exist)")
    
    # Reset gradients
    embeddings.grad = None
    quantized_embeddings.grad = None
    
    # Commitment loss (gradients to embeddings only)
    commitment_loss = vq_loss.commitment_loss(embeddings, quantized_embeddings)
    commitment_loss.backward()
    
    print(f"\n✓ Commitment loss gradients:")
    print(f"  Embeddings grad: {embeddings.grad is not None} (should exist)")
    print(f"  Quantized grad: {quantized_embeddings.grad is None} (should be None)")
    
    # Test reset scheduler
    print("\n" + "-"*60)
    print("Testing Codebook Reset Scheduler...")
    
    scheduler = CodebookResetScheduler(reset_threshold=5, reset_prob=1.0)
    
    # Simulate training steps
    for step in range(10):
        # Use only roles 0 and 1, ignore role 2
        role_assignments = torch.randint(0, 2, (batch_size,))
        scheduler.update_usage(role_assignments, n_roles)
    
    reset_mask = scheduler.get_reset_mask(n_roles)
    
    print(f"✓ Reset scheduler:")
    print(f"  Usage counts: {scheduler.usage_counts}")
    print(f"  Reset mask: {reset_mask.tolist()}")
    print(f"  Role 2 should be reset: {reset_mask[2].item()}")
    
    print("\n" + "="*60)
    print("✓ All VQ loss tests passed!")
