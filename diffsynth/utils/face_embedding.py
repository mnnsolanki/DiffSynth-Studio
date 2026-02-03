"""
Face Identity Embedding Extraction Module

This module provides utilities to extract facial identity embeddings from images
using various encoders (CLIP, DINOv3, or specialized face encoders).

Phase 1 of Dual-Input Training: Face Embedding Context
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union, Optional, Tuple
from tqdm import tqdm


class FaceEmbeddingExtractor:
    """
    Extract facial identity embeddings from images using pre-trained encoders.
    
    Supports multiple encoder types:
    - "clip": OpenAI CLIP (recommended for general face understanding)
    - "dinov3": Meta's DINOv3 (excellent for fine-grained identity features)
    """
    
    def __init__(
        self,
        encoder_type: str = "clip",
        model_id: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the face embedding extractor.
        
        Args:
            encoder_type: Type of encoder ("clip" or "dinov3")
            model_id: Model identifier for HuggingFace models
            device: Device to run on ("cuda" or "cpu")
            torch_dtype: Torch dtype for computations
        """
        self.encoder_type = encoder_type
        self.device = device
        self.torch_dtype = torch_dtype
        self.encoder = None
        self.processor = None
        
        self._load_encoder(encoder_type, model_id)
    
    def _load_encoder(self, encoder_type: str, model_id: str):
        """Load the appropriate encoder."""
        if encoder_type == "clip":
            try:
                from transformers import CLIPProcessor, CLIPModel
                self.processor = CLIPProcessor.from_pretrained(model_id)
                self.encoder = CLIPModel.from_pretrained(model_id).to(self.device).to(self.torch_dtype)
                self.encoder.eval()
                self.embedding_dim = self.encoder.config.projection_dim
            except ImportError:
                raise ImportError("transformers library required for CLIP encoder")
        
        elif encoder_type == "dinov3":
            try:
                import timm
                self.encoder = timm.create_model('vit_large_patch14_dinov3.lvim1m_lp7', 
                                                 pretrained=True).to(self.device)
                self.encoder.eval()
                self.embedding_dim = 1024  # DINOv3 large embedding dimension
            except ImportError:
                raise ImportError("timm library required for DINOv3 encoder")
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    @torch.no_grad()
    def extract_embedding(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Extract a single embedding from an image.
        
        Args:
            image: PIL Image or tensor of shape (C, H, W)
            
        Returns:
            Embedding tensor of shape (embedding_dim,)
        """
        if isinstance(image, Image.Image):
            if self.encoder_type == "clip":
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device).to(self.torch_dtype) for k, v in inputs.items()}
                image_features = self.encoder.get_image_features(**inputs)
            else:  # dinov3
                inputs = self.processor(image).unsqueeze(0).to(self.device).to(self.torch_dtype)
                image_features = self.encoder.forward_features(inputs)
                image_features = image_features[:, 0]  # CLS token
        else:
            # Assume it's a tensor
            if image.ndim == 2:
                image = image.unsqueeze(0)
            image = image.to(self.device).to(self.torch_dtype)
            if self.encoder_type == "clip":
                image_features = self.encoder.get_image_features(image)
            else:
                image_features = self.encoder.forward_features(image)
                image_features = image_features[:, 0]
        
        # Normalize embedding
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features.squeeze(0)
    
    @torch.no_grad()
    def extract_embeddings_batch(
        self,
        images: List[Union[Image.Image, str, Path]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract embeddings from a batch of images.
        
        Args:
            images: List of PIL Images, image paths, or Path objects
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Tensor of shape (num_images, embedding_dim)
        """
        embeddings = []
        
        for image_input in tqdm(images, desc="Extracting face embeddings"):
            if isinstance(image_input, (str, Path)):
                image = Image.open(image_input).convert("RGB")
            else:
                image = image_input
            
            embedding = self.extract_embedding(image)
            embeddings.append(embedding)
        
        embeddings = torch.stack(embeddings, dim=0)
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def compute_identity_embedding(
        self,
        images: List[Union[Image.Image, str, Path]],
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """
        Compute an aggregated identity embedding from multiple images.
        
        Args:
            images: List of face images
            aggregation: Aggregation method ("mean", "median", or "weighted_mean")
            
        Returns:
            Single aggregated embedding tensor of shape (embedding_dim,)
        """
        embeddings = self.extract_embeddings_batch(images)
        
        if aggregation == "mean":
            identity_embedding = embeddings.mean(dim=0)
        elif aggregation == "median":
            identity_embedding = embeddings.median(dim=0)[0]
        elif aggregation == "weighted_mean":
            # Weight by similarity to mean
            mean_emb = embeddings.mean(dim=0)
            weights = F.cosine_similarity(embeddings, mean_emb.unsqueeze(0))
            weights = F.softmax(weights, dim=0)
            identity_embedding = (embeddings * weights.unsqueeze(-1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        return F.normalize(identity_embedding, p=2, dim=-1)
    
    def save_embedding(self, embedding: torch.Tensor, path: Union[str, Path]):
        """Save embedding to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding.cpu(), path)
    
    def load_embedding(self, path: Union[str, Path]) -> torch.Tensor:
        """Load embedding from disk."""
        return torch.load(path, map_location=self.device)


class FaceEmbeddingDataset:
    """
    Dataset class for managing face embeddings and their metadata.
    """
    
    def __init__(self, name: str = "identity"):
        """
        Initialize face embedding dataset.
        
        Args:
            name: Name/identifier for this face identity
        """
        self.name = name
        self.embeddings: List[torch.Tensor] = []
        self.images: List[Union[str, Path]] = []
        self.identity_embedding: Optional[torch.Tensor] = None
    
    def add_embedding(
        self,
        embedding: torch.Tensor,
        image_path: Optional[Union[str, Path]] = None,
    ):
        """Add an embedding to the dataset."""
        self.embeddings.append(embedding.cpu())
        if image_path is not None:
            self.images.append(image_path)
    
    def compute_identity_vector(self, aggregation: str = "mean") -> torch.Tensor:
        """Compute aggregated identity vector."""
        if not self.embeddings:
            raise ValueError("No embeddings in dataset")
        
        stacked = torch.stack(self.embeddings)
        
        if aggregation == "mean":
            self.identity_embedding = stacked.mean(dim=0)
        elif aggregation == "median":
            self.identity_embedding = stacked.median(dim=0)[0]
        elif aggregation == "weighted_mean":
            mean_emb = stacked.mean(dim=0)
            weights = F.cosine_similarity(stacked, mean_emb.unsqueeze(0))
            weights = F.softmax(weights, dim=0)
            self.identity_embedding = (stacked * weights.unsqueeze(-1)).sum(dim=0)
        
        return F.normalize(self.identity_embedding, p=2, dim=-1)
    
    def save(self, directory: Union[str, Path]):
        """Save dataset to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save individual embeddings
        embeddings_dir = directory / "embeddings"
        embeddings_dir.mkdir(exist_ok=True)
        for i, emb in enumerate(self.embeddings):
            torch.save(emb, embeddings_dir / f"embedding_{i:04d}.pt")
        
        # Save identity embedding
        if self.identity_embedding is not None:
            torch.save(self.identity_embedding, directory / "identity_embedding.pt")
        
        # Save metadata
        metadata = {
            "name": self.name,
            "num_embeddings": len(self.embeddings),
            "embedding_dim": self.embeddings[0].shape[0] if self.embeddings else None,
        }
        torch.save(metadata, directory / "metadata.pt")
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> "FaceEmbeddingDataset":
        """Load dataset from disk."""
        directory = Path(directory)
        metadata = torch.load(directory / "metadata.pt")
        
        dataset = cls(name=metadata["name"])
        
        embeddings_dir = directory / "embeddings"
        if embeddings_dir.exists():
            for emb_file in sorted(embeddings_dir.glob("embedding_*.pt")):
                dataset.add_embedding(torch.load(emb_file))
        
        if (directory / "identity_embedding.pt").exists():
            dataset.identity_embedding = torch.load(directory / "identity_embedding.pt")
        
        return dataset


# Convenience function for quick embedding extraction
def extract_face_embeddings(
    image_paths: List[Union[str, Path]],
    encoder_type: str = "clip",
    device: str = "cuda",
    aggregation: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quick utility to extract and aggregate face embeddings.
    
    Args:
        image_paths: List of paths to face images
        encoder_type: Encoder type ("clip" or "dinov3")
        device: Device to use
        aggregation: Aggregation method
        
    Returns:
        Tuple of (identity_embedding, all_embeddings)
    """
    extractor = FaceEmbeddingExtractor(encoder_type=encoder_type, device=device)
    all_embeddings = extractor.extract_embeddings_batch(image_paths)
    identity_embedding = extractor.compute_identity_embedding(image_paths, aggregation=aggregation)
    
    return identity_embedding, all_embeddings
