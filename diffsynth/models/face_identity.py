"""
Face Identity Mapping Network

A lightweight adapter that projects face embeddings into the model's conditioning space.
This is combined with LoRA training for efficient face identity integration.

Phase 2 of Dual-Input Training: Training Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FaceIdentityMapper(nn.Module):
    """
    Maps face embeddings to conditioning space.
    
    This module projects face identity embeddings (e.g., from CLIP) into the
    model's conditioning space using learnable linear transformations.
    """
    
    def __init__(
        self,
        face_embedding_dim: int = 768,  # CLIP embedding dim
        conditioning_dim: int = 768,     # Model conditioning dim
        hidden_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize the face identity mapper.
        
        Args:
            face_embedding_dim: Dimension of input face embeddings
            conditioning_dim: Dimension of output conditioning
            hidden_dim: Dimension of hidden layers
            num_layers: Number of MLP layers (1-3 recommended)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.face_embedding_dim = face_embedding_dim
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build MLP
        layers = []
        
        if num_layers == 1:
            layers.append(nn.Linear(face_embedding_dim, conditioning_dim))
        else:
            # Input projection
            layers.append(nn.Linear(face_embedding_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            # Hidden layers
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            
            # Output projection
            layers.append(nn.Linear(hidden_dim, conditioning_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, face_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project face embedding to conditioning space.
        
        Args:
            face_embedding: Tensor of shape (B, face_embedding_dim) or (face_embedding_dim,)
            
        Returns:
            Conditioning tensor of shape (B, conditioning_dim) or (conditioning_dim,)
        """
        squeeze = False
        if face_embedding.dim() == 1:
            face_embedding = face_embedding.unsqueeze(0)
            squeeze = True
        
        conditioning = self.mlp(face_embedding)
        
        if squeeze:
            conditioning = conditioning.squeeze(0)
        
        return conditioning


class FaceIdentityAdapter(nn.Module):
    """
    Complete adapter for face identity conditioning.
    
    Combines the mapper with additional operations like normalization,
    scaling, and residual connections.
    """
    
    def __init__(
        self,
        face_embedding_dim: int = 768,
        conditioning_dim: int = 768,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.1,
        scale: float = 1.0,
        use_residual: bool = False,
    ):
        """
        Initialize face identity adapter.
        
        Args:
            face_embedding_dim: Dimension of input face embeddings
            conditioning_dim: Dimension of output conditioning
            hidden_dim: Dimension of hidden layers
            num_layers: Number of MLP layers
            dropout: Dropout rate
            scale: Scaling factor for the face conditioning
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.mapper = FaceIdentityMapper(
            face_embedding_dim=face_embedding_dim,
            conditioning_dim=conditioning_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.scale = scale
        self.use_residual = use_residual and (face_embedding_dim == conditioning_dim)
    
    def forward(self, face_embedding: torch.Tensor) -> torch.Tensor:
        """
        Process face embedding.
        
        Args:
            face_embedding: Input face embedding
            
        Returns:
            Adapted face conditioning
        """
        conditioning = self.mapper(face_embedding)
        
        if self.use_residual and face_embedding.shape[-1] == conditioning.shape[-1]:
            conditioning = conditioning + face_embedding
        
        if self.scale != 1.0:
            conditioning = conditioning * self.scale
        
        return conditioning


class FaceIdentityLoRA(nn.Module):
    """
    LoRA module specifically for face identity conditioning.
    
    This is trained alongside the mapper to efficiently learn face-specific
    adjustments without modifying the base model.
    """
    
    def __init__(
        self,
        base_dim: int = 768,
        rank: int = 64,
        alpha: float = 1.0,
        dropout: float = 0.1,
    ):
        """
        Initialize face identity LoRA.
        
        Args:
            base_dim: Dimension of the base feature
            rank: LoRA rank (lower = more efficient, typically 4-128)
            alpha: Scaling factor
            dropout: Dropout for regularization
        """
        super().__init__()
        
        self.base_dim = base_dim
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_down = nn.Linear(base_dim, rank, bias=False)
        self.lora_up = nn.Linear(rank, base_dim, bias=False)
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        # Initialize with small values
        nn.init.normal_(self.lora_down.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_up.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LoRA to feature.
        
        Args:
            x: Input feature tensor
            
        Returns:
            LoRA-adjusted feature
        """
        if self.dropout is not None:
            x_lora = self.dropout(x)
        else:
            x_lora = x
        
        x_lora = self.lora_down(x_lora)
        x_lora = self.lora_up(x_lora)
        x_lora = x_lora * self.alpha / self.rank
        
        return x + x_lora


class FaceConditioningBlock(nn.Module):
    """
    Complete face conditioning block for integration into the pipeline.
    
    Combines face embedding extraction, mapping, and LoRA adaptation.
    """
    
    def __init__(
        self,
        face_embedding_dim: int = 768,
        conditioning_dim: int = 768,
        hidden_dim: int = 1024,
        num_mapper_layers: int = 2,
        use_lora: bool = True,
        lora_rank: int = 64,
        dropout: float = 0.1,
        scale: float = 1.0,
    ):
        """
        Initialize face conditioning block.
        
        Args:
            face_embedding_dim: Dimension of input face embeddings
            conditioning_dim: Dimension of output conditioning
            hidden_dim: Hidden dimension for mapper
            num_mapper_layers: Number of mapper layers
            use_lora: Whether to use LoRA
            lora_rank: LoRA rank
            dropout: Dropout rate
            scale: Scaling factor
        """
        super().__init__()
        
        self.adapter = FaceIdentityAdapter(
            face_embedding_dim=face_embedding_dim,
            conditioning_dim=conditioning_dim,
            hidden_dim=hidden_dim,
            num_layers=num_mapper_layers,
            dropout=dropout,
            scale=scale,
        )
        
        self.use_lora = use_lora
        if use_lora:
            self.lora = FaceIdentityLoRA(
                base_dim=conditioning_dim,
                rank=lora_rank,
                alpha=1.0,
                dropout=dropout,
            )
    
    def forward(self, face_embedding: torch.Tensor) -> torch.Tensor:
        """
        Process face embedding through complete conditioning pipeline.
        
        Args:
            face_embedding: Input face embedding
            
        Returns:
            Face conditioning tensor
        """
        conditioning = self.adapter(face_embedding)
        
        if self.use_lora:
            conditioning = self.lora(conditioning)
        
        return conditioning
    
    def get_trainable_parameters(self):
        """Get parameters that should be trained."""
        params = list(self.adapter.parameters())
        if self.use_lora:
            params += list(self.lora.parameters())
        return params
    
    def freeze_base_mapper(self):
        """Freeze base mapper (keep LoRA trainable for fine-tuning)."""
        for param in self.adapter.mapper.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


# Utility functions for multi-face conditioning
def blend_face_embeddings(
    face_embeddings: list,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Blend multiple face embeddings (for creative effects).
    
    Args:
        face_embeddings: List of face embedding tensors
        weights: Optional weights for blending (will be normalized)
        
    Returns:
        Blended face embedding
    """
    if weights is None:
        weights = torch.ones(len(face_embeddings), device=face_embeddings[0].device)
    
    weights = F.softmax(weights, dim=0)
    
    stacked = torch.stack(face_embeddings, dim=0)
    blended = (stacked * weights.unsqueeze(-1)).sum(dim=0)
    
    return F.normalize(blended, p=2, dim=-1)


def interpolate_face_embeddings(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    t: float = 0.5,
) -> torch.Tensor:
    """
    Interpolate between two face embeddings (for smooth transitions).
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        t: Interpolation parameter (0 = embedding1, 1 = embedding2)
        
    Returns:
        Interpolated face embedding
    """
    # Normalize for slerp
    e1 = F.normalize(embedding1, p=2, dim=-1)
    e2 = F.normalize(embedding2, p=2, dim=-1)
    
    # Compute angle between embeddings
    cos_angle = torch.sum(e1 * e2)
    angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
    
    # Slerp (spherical linear interpolation)
    if angle.abs() < 1e-6:
        # Embeddings are nearly identical
        result = (1 - t) * e1 + t * e2
    else:
        sin_angle = torch.sin(angle)
        result = (torch.sin((1 - t) * angle) / sin_angle) * e1 + (torch.sin(t * angle) / sin_angle) * e2
    
    return F.normalize(result, p=2, dim=-1)
