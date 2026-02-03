"""
Updated Wan2.1-VACE-1.3B Validation Script with Face Identity

This is an enhanced version of the original validation script that includes
face identity conditioning for consistent facial features.
"""

import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.face_identity import FaceConditioningBlock
from diffsynth.utils.face_embedding import extract_face_embeddings
from pathlib import Path


# ========== Configuration ==========
USE_FACE_IDENTITY = True  # Set to False to disable face identity
FACE_IMAGES_DIR = "data_infer/face_reference"  # Directory with face images
FACE_LORA_PATH = None  # Optional: path to trained face LoRA
FACE_SCALE = 1.0  # Face conditioning strength (0.5-1.5)

# ========== Load Pipeline ==========
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"),
        ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth"),
    ],
)

# Load the original VACE LoRA (for pose conditioning)
pipe.load_lora(pipe.vace, "models/train/Wan2.1-VACE-1.3B_lora/step-2800.safetensors", alpha=1)

# ========== Setup Face Identity ==========
face_embedding = None
if USE_FACE_IDENTITY:
    face_images_dir = Path(FACE_IMAGES_DIR)
    
    # Check if face images directory exists
    if face_images_dir.exists():
        print("=" * 80)
        print("EXTRACTING FACE EMBEDDINGS")
        print("=" * 80)
        
        # Get face images
        face_image_paths = list(face_images_dir.glob("*.jpg")) + \
                          list(face_images_dir.glob("*.png"))
        
        if len(face_image_paths) > 0:
            print(f"Found {len(face_image_paths)} face reference images")
            
            # Extract face embeddings
            identity_embedding, _ = extract_face_embeddings(
                image_paths=face_image_paths,
                encoder_type="clip",
                device="cuda",
                aggregation="weighted_mean",
            )
            face_embedding = identity_embedding
            print(f"Face embedding extracted: shape {face_embedding.shape}")
            
            # Initialize face conditioning block
            face_block = FaceConditioningBlock(
                face_embedding_dim=768,
                conditioning_dim=768,
                hidden_dim=1024,
                num_mapper_layers=2,
                use_lora=True,
                lora_rank=64,
                dropout=0.0,
                scale=FACE_SCALE,
            ).to("cuda").to(torch.bfloat16)
            
            # Load trained face LoRA if provided
            if FACE_LORA_PATH and Path(FACE_LORA_PATH).exists():
                print(f"Loading face LoRA from: {FACE_LORA_PATH}")
                face_block.load_state_dict(torch.load(FACE_LORA_PATH, map_location="cuda"))
                print("Face LoRA loaded successfully")
            else:
                print("⚠ No face LoRA provided - using untrained mapper")
            
            # Attach to pipeline
            pipe.face_conditioning_block = face_block
            face_block.eval()
            
            print("✓ Face identity system ready")
            print("=" * 80)
        else:
            print(f"⚠ No face images found in {face_images_dir}")
            USE_FACE_IDENTITY = False
    else:
        print(f"⚠ Face images directory not found: {face_images_dir}")
        USE_FACE_IDENTITY = False

# ========== Load Videos ==========
print("\nLoading pose video and reference images...")

# Load pose video
video = VideoData("data_infer/processed/pose/dance-1_1_pose.mp4", height=480, width=832)
video = [video[i] for i in range(65)]

# Load reference images (for VACE)
ref_img = VideoData("data_infer/ref_img.jpg", height=480, width=832)[0]
ref_img_1 = VideoData("data_infer/ref_img_1.jpg", height=480, width=832)[0]
ref_img_2 = VideoData("data_infer/ref_img_2.jpg", height=480, width=832)[0]
ref_img_3 = VideoData("data_infer/ref_img_3.jpg", height=480, width=832)[0]
ref_img_4 = VideoData("data_infer/ref_img_4.jpg", height=480, width=832)[0]

print("✓ Videos and images loaded")

# ========== Generate Video ==========
print("\nGenerating video...")
print(f"  Face Identity: {'ENABLED' if USE_FACE_IDENTITY else 'DISABLED'}")
print(f"  Face Scale: {FACE_SCALE if USE_FACE_IDENTITY else 'N/A'}")

# Enhanced prompt for face identity
prompt = "Person is dancing by following the video pose exactly. Dance is natural and smooth. "
if USE_FACE_IDENTITY:
    prompt += "Clear face with consistent identity and features. Maintain exact facial appearance throughout. "
prompt += "Maintain the exact facial features, hair, clothing, and background from the reference images. Keep the background consistent with the reference images."

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
if USE_FACE_IDENTITY:
    negative_prompt += ", inconsistent face, blurry face, distorted face, multiple people"

video_output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    vace_video=video,
    vace_reference_image=[ref_img, ref_img_1, ref_img_2, ref_img_3, ref_img_4],
    face_embedding=face_embedding if USE_FACE_IDENTITY else None,
    face_scale=FACE_SCALE if USE_FACE_IDENTITY else 1.0,
    num_frames=65,
    seed=1,
    tiled=True
)

# Save output
output_filename = "results/lora_dance_1_with_face.mp4" if USE_FACE_IDENTITY else "results/lora_dance_1.mp4"
save_video(video_output, output_filename, fps=15, quality=9)

print(f"\n✓ Video saved to: {output_filename}")
print("=" * 80)
