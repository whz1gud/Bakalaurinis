"""
Similarity Model using DINOv2 as backbone.

The model consists of:
1. DINOv2 backbone (pretrained, can be frozen or fine-tuned)
2. Optional projection head (linear or MLP)
3. L2 normalization for embedding space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


class SimilarityModel(nn.Module):
    """
    Image similarity model using DINOv2 backbone.
    
    Args:
        backbone_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', etc.)
        embedding_dim: Output embedding dimension (default: 128)
        freeze_backbone: Whether to freeze backbone weights (default: False)
        head_type: Type of projection head ('linear', 'mlp', 'none')
    """
    
    DINOV2_DIMS = {
        'dinov2_vits14': 384,   # Small
        'dinov2_vitb14': 768,   # Base
        'dinov2_vitl14': 1024,  # Large
        'dinov2_vitg14': 1536,  # Giant
    }
    
    def __init__(
        self,
        backbone_name: str = 'dinov2_vits14',
        embedding_dim: int = 128,
        freeze_backbone: bool = False,
        head_type: Literal['linear', 'mlp', 'none'] = 'linear',
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone
        
        # Load DINOv2 backbone
        print(f"Loading DINOv2 backbone: {backbone_name}...")
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
        
        # Get backbone output dimension
        self.backbone_dim = self.DINOV2_DIMS.get(backbone_name, 384)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone weights frozen.")
        
        # Create projection head
        if head_type == 'none':
            self.head = nn.Identity()
            self.embedding_dim = self.backbone_dim
        elif head_type == 'linear':
            self.head = nn.Linear(self.backbone_dim, embedding_dim)
        elif head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim),
                nn.ReLU(),
                nn.Linear(self.backbone_dim, embedding_dim),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")
        
        print(f"Model created: {backbone_name} -> {embedding_dim}D embeddings")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized embeddings from images.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            L2-normalized embeddings [B, embedding_dim]
        """
        # Extract backbone features
        features = self.backbone(x)  # [B, backbone_dim]
        
        # Project to embedding space
        embeddings = self.head(features)  # [B, embedding_dim]
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw backbone features without projection head."""
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.freeze_backbone = False
        print("Backbone weights unfrozen.")
    
    def freeze_backbone_layers(self, num_layers_to_freeze: int = 6):
        """
        Partially freeze backbone (freeze first N layers).
        Useful for gradual unfreezing during training.
        """
        # DINOv2 ViT has blocks we can freeze
        if hasattr(self.backbone, 'blocks'):
            for i, block in enumerate(self.backbone.blocks):
                if i < num_layers_to_freeze:
                    for param in block.parameters():
                        param.requires_grad = False
                else:
                    for param in block.parameters():
                        param.requires_grad = True
            print(f"Froze first {num_layers_to_freeze} transformer blocks.")


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head for metric learning.
    
    Used during training with ArcFace loss.
    Maps embeddings to class logits with angular margin.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.5,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes
        
        # Weight matrix (class centers on unit hypersphere)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ArcFace logits.
        
        Args:
            embeddings: L2-normalized embeddings [B, D]
            labels: Class labels [B] (required during training)
            
        Returns:
            Logits [B, num_classes]
        """
        # Normalize weights
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight_norm)  # [B, num_classes]
        
        if labels is not None and self.training:
            # Add angular margin to target class
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            target_logits = torch.cos(theta + self.margin)
            
            # Create one-hot mask
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1)
            
            # Apply margin only to target class
            output = cosine * (1 - one_hot) + target_logits * one_hot
            output *= self.scale
        else:
            output = cosine * self.scale
        
        return output


def create_model(
    backbone: str = 'dinov2_vits14',
    embedding_dim: int = 128,
    freeze_backbone: bool = False,
    device: str = 'cuda',
) -> SimilarityModel:
    """
    Factory function to create a similarity model.
    
    Args:
        backbone: DINOv2 variant name
        embedding_dim: Output embedding dimension
        freeze_backbone: Whether to freeze backbone
        device: Device to place model on
        
    Returns:
        SimilarityModel instance
    """
    model = SimilarityModel(
        backbone_name=backbone,
        embedding_dim=embedding_dim,
        freeze_backbone=freeze_backbone,
    )
    model = model.to(device)
    return model


