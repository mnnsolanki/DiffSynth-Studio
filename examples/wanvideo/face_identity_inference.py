"""
Face Identity Video Generation - Inference Example

This script demonstrates how to generate videos with consistent face identity
using pre-trained face LoRA adapters.

Phase 3: Inference with Face Identity Conditioning
"""

import torch
from PIL import Image
from pathlib import Path
from diffsynth.utils.data import save_video, VideoData
from diffsynth.utils.face_embedding import FaceEmbeddingExtractor, extract_face_embeddings
from diffsynth.models.face_identity import FaceConditioningBlock
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def generate_video_with_face_identity(
    # Face identity inputs
    face_images_dir: str,
    face_lora_path: str = None,
    # Motion inputs
    pose_video_path: str = None,
    # Model paths
    model_path: str = "models/Wan-AI/Wan2.1-VACE-1.3B",
    # Generation settings
    prompt: str = "A person is dancing following the pose video exactly. Natural and smooth movement.",
    negative_prompt: str = "blurry face, inconsistent appearance, static, multiple people, distorted",
    num_frames: int = 65,
    height: int = 480,
    width: int = 832,
    seed: int = 1,
    cfg_scale: float = 5.0,
    num_inference_steps: int = 50,
    face_scale: float = 1.0,
    # Output
    output_path: str = "results/face_identity_video.mp4",
    # Device
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    Generate a video with consistent face identity.
    
    Args:
        face_images_dir: Directory containing 3-15 reference images of the target face
        face_lora_path: Optional path to pre-trained face LoRA weights
        pose_video_path: Path to pose/motion video
        model_path: Path to base Wan model
        prompt: Text prompt for generation
        negative_prompt: Negative prompt
        num_frames: Number of frames to generate
        height: Video height
        width: Video width
        seed: Random seed
        cfg_scale: Classifier-free guidance scale
        num_inference_steps: Number of diffusion steps
        face_scale: How strongly to apply face conditioning (0.5-1.5)
        output_path: Where to save the generated video
        device: Device to use
        torch_dtype: Torch dtype
    """
    print("=" * 80)
    print("FACE IDENTITY VIDEO GENERATION")
    print("=" * 80)
    
    # Step 1: Extract face embeddings
    print("\n[1/5] Extracting face embeddings from reference images...")
    face_images_dir = Path(face_images_dir)
    face_image_paths = list(face_images_dir.glob("*.jpg")) + list(face_images_dir.glob("*.png"))
    
    if len(face_image_paths) == 0:
        raise ValueError(f"No face images found in {face_images_dir}")
    
    print(f"  Found {len(face_image_paths)} face reference images")
    
    # Extract and aggregate face embeddings
    identity_embedding, all_embeddings = extract_face_embeddings(
        image_paths=face_image_paths,
        encoder_type="clip",
        device=device,
        aggregation="weighted_mean",  # Use weighted mean for best results
    )
    
    print(f"  ✓ Face embedding extracted: shape {identity_embedding.shape}")
    
    # Step 2: Initialize the Wan pipeline
    print("\n[2/5] Loading Wan video pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(path=f"{model_path}/diffusion_pytorch_model.safetensors"),
            ModelConfig(path=f"{model_path}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{model_path}/Wan2.1_VAE.pth"),
        ],
    )
    print("  ✓ Pipeline loaded")
    
    # Step 3: Initialize and load face conditioning block
    print("\n[3/5] Setting up face conditioning...")
    face_block = FaceConditioningBlock(
        face_embedding_dim=768,  # CLIP embedding dimension
        conditioning_dim=768,     # Match your model's conditioning dimension
        hidden_dim=1024,
        num_mapper_layers=2,
        use_lora=True,
        lora_rank=64,
        dropout=0.0,  # No dropout during inference
        scale=face_scale,
    ).to(device).to(torch_dtype)
    
    # Load pre-trained face LoRA if provided
    if face_lora_path:
        print(f"  Loading face LoRA from: {face_lora_path}")
        face_block.load_state_dict(torch.load(face_lora_path, map_location=device))
        print("  ✓ Face LoRA loaded")
    else:
        print("  ⚠ No face LoRA provided - using untrained mapper")
    
    # Attach to pipeline
    pipe.face_conditioning_block = face_block
    face_block.eval()
    
    # Step 4: Load pose video
    print("\n[4/5] Loading pose/motion video...")
    if pose_video_path:
        pose_video = VideoData(pose_video_path, height=height, width=width)
        pose_frames = [pose_video[i] for i in range(min(num_frames, len(pose_video)))]
        print(f"  ✓ Loaded {len(pose_frames)} pose frames")
    else:
        pose_frames = None
        print("  ⚠ No pose video provided - generating text-to-video")
    
    # Step 5: Generate video
    print("\n[5/5] Generating video with face identity...")
    print(f"  Prompt: {prompt}")
    print(f"  Face scale: {face_scale}")
    print(f"  Frames: {num_frames}, Size: {width}x{height}")
    print(f"  CFG scale: {cfg_scale}, Steps: {num_inference_steps}")
    
    with torch.no_grad():
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            vace_video=pose_frames,
            face_embedding=identity_embedding,
            face_scale=face_scale,
            num_frames=num_frames if pose_frames is None else len(pose_frames),
            height=height,
            width=width,
            seed=seed,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            tiled=True,
        )
    
    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(output_path), fps=15, quality=9)
    
    print(f"\n✓ Video generated successfully!")
    print(f"  Saved to: {output_path}")
    print("=" * 80)


def batch_generate_variations(
    face_images_dir: str,
    pose_videos_dir: str,
    face_lora_path: str,
    output_dir: str = "results/face_variations",
    **kwargs
):
    """
    Generate multiple video variations with the same face identity.
    
    Useful for testing consistency across different motions.
    """
    print("\n" + "=" * 80)
    print("BATCH GENERATION WITH FACE IDENTITY")
    print("=" * 80)
    
    pose_videos_dir = Path(pose_videos_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all pose videos
    pose_videos = sorted(pose_videos_dir.glob("*.mp4"))
    print(f"\nFound {len(pose_videos)} pose videos to process")
    
    for idx, pose_video_path in enumerate(pose_videos, 1):
        print(f"\n--- Processing video {idx}/{len(pose_videos)}: {pose_video_path.name} ---")
        
        output_path = output_dir / f"{pose_video_path.stem}_face_identity.mp4"
        
        try:
            generate_video_with_face_identity(
                face_images_dir=face_images_dir,
                face_lora_path=face_lora_path,
                pose_video_path=str(pose_video_path),
                output_path=str(output_path),
                **kwargs
            )
        except Exception as e:
            print(f"✗ Error processing {pose_video_path.name}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"✓ Batch generation complete! All videos saved to: {output_dir}")
    print("=" * 80)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate videos with face identity conditioning")
    
    # Face identity
    parser.add_argument("--face_images_dir", type=str, required=True,
                        help="Directory with face reference images")
    parser.add_argument("--face_lora_path", type=str, default=None,
                        help="Path to trained face LoRA weights")
    parser.add_argument("--face_scale", type=float, default=1.0,
                        help="Face conditioning strength (0.5-1.5)")
    
    # Motion
    parser.add_argument("--pose_video", type=str, default=None,
                        help="Path to pose/motion video")
    parser.add_argument("--pose_videos_dir", type=str, default=None,
                        help="Directory of pose videos for batch generation")
    
    # Model
    parser.add_argument("--model_path", type=str,
                        default="models/Wan-AI/Wan2.1-VACE-1.3B",
                        help="Path to Wan model")
    
    # Generation
    parser.add_argument("--prompt", type=str,
                        default="A person is dancing following the pose video exactly. Natural and smooth movement.",
                        help="Generation prompt")
    parser.add_argument("--negative_prompt", type=str,
                        default="blurry face, inconsistent appearance, static, multiple people, distorted",
                        help="Negative prompt")
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    
    # Output
    parser.add_argument("--output", type=str, default="results/face_identity_video.mp4")
    parser.add_argument("--output_dir", type=str, default="results/face_variations")
    
    # Mode
    parser.add_argument("--batch", action="store_true",
                        help="Batch process multiple pose videos")
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.pose_videos_dir:
            raise ValueError("--pose_videos_dir required for batch mode")
        
        batch_generate_variations(
            face_images_dir=args.face_images_dir,
            pose_videos_dir=args.pose_videos_dir,
            face_lora_path=args.face_lora_path,
            output_dir=args.output_dir,
            model_path=args.model_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            face_scale=args.face_scale,
        )
    else:
        generate_video_with_face_identity(
            face_images_dir=args.face_images_dir,
            face_lora_path=args.face_lora_path,
            pose_video_path=args.pose_video,
            model_path=args.model_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            face_scale=args.face_scale,
            output_path=args.output,
        )
