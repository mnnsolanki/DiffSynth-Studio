"""
Face Embedding Processing Units for WAN Video Pipeline

These pipeline units handle the integration of face identity embeddings
into the WAN video generation process.
"""

from typing import Optional
import torch
from ..diffusion.base_pipeline import PipelineUnit


class WanVideoUnit_FaceEmbedding(PipelineUnit):
    """
    Process face identity embeddings for conditioning.
    
    This unit handles:
    1. Loading or computing face embeddings
    2. Projecting them to the model's conditioning space
    3. Scaling and normalizing for stable training/inference
    """
    
    def __init__(self):
        super().__init__(
            input_params=("face_embedding", "face_scale"),
            output_params=("face_conditioning", "face_scale"),
            onload_model_names=("face_conditioning_block",)
        )
    
    def process(self, pipe, face_embedding, face_scale):
        """
        Process face embedding into conditioning.
        
        Args:
            pipe: WanVideoPipeline instance
            face_embedding: Tensor of shape (embedding_dim,) or (B, embedding_dim)
            face_scale: Scaling factor for face conditioning
            
        Returns:
            Dictionary with face_conditioning and face_scale
        """
        if face_embedding is None:
            return {"face_conditioning": None, "face_scale": face_scale}
        
        # Ensure model is loaded
        if hasattr(pipe, 'face_conditioning_block') and pipe.face_conditioning_block is not None:
            pipe.load_models_to_device(self.onload_model_names)
            
            # Move embedding to device and correct dtype
            face_embedding = face_embedding.to(pipe.device).to(pipe.torch_dtype)
            
            # Process through face conditioning block
            face_conditioning = pipe.face_conditioning_block(face_embedding)
            face_conditioning = face_conditioning.to(dtype=pipe.torch_dtype, device=pipe.device)
            
            return {"face_conditioning": face_conditioning, "face_scale": face_scale}
        
        return {"face_conditioning": None, "face_scale": face_scale}


class WanVideoUnit_FaceIdentityLora(PipelineUnit):
    """
    Load and apply face identity LoRA weights.
    
    Handles loading pre-trained face identity LoRA adapters that were
    trained on specific face identities.
    """
    
    def __init__(self):
        super().__init__(
            input_params=("face_identity_lora_path",),
            output_params=(),
        )
    
    def process(self, pipe, face_identity_lora_path):
        """
        Load face identity LoRA.
        
        Args:
            pipe: WanVideoPipeline instance
            face_identity_lora_path: Path to saved LoRA weights
            
        Returns:
            Empty dictionary (LoRA is applied in-place)
        """
        if face_identity_lora_path is None:
            return {}
        
        if hasattr(pipe, 'face_conditioning_block') and pipe.face_conditioning_block is not None:
            try:
                lora_state = torch.load(face_identity_lora_path, map_location=pipe.device)
                pipe.face_conditioning_block.load_state_dict(lora_state, strict=False)
            except Exception as e:
                print(f"Warning: Could not load face identity LoRA from {face_identity_lora_path}: {e}")
        
        return {}


# Integration code for wan_video.py
# This should be added to the imports section:
"""
from ..models.face_identity import FaceConditioningBlock
"""

# This should be added to WanVideoPipeline.__init__:
"""
        self.face_conditioning_block: FaceConditioningBlock = None
"""

# This should be added to the units list in WanVideoPipeline.__init__:
"""
            WanVideoUnit_FaceEmbedding(),
            WanVideoUnit_FaceIdentityLora(),
"""

# This should be added to the __call__ method signature:
"""
        face_embedding: Optional[torch.Tensor] = None,
        face_scale: Optional[float] = 1.0,
        face_identity_lora_path: Optional[str] = None,
"""
