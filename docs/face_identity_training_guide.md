# Face Identity Training for WAN Video Generation

## 🎯 Overview

This system enables you to generate videos where a specific person (with a clear, consistent face) performs dance movements from your AIST pose dataset. It uses a **dual-input training approach**:

1. **Face Embedding Context**: 10-15 high-quality face images of your target identity
2. **Motion Context**: Your existing AIST dance pose videos

During inference, you provide both a pose video and the face embedding to generate videos with consistent facial identity.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [Phase 1: Data Preparation](#phase-1-data-preparation)
4. [Phase 2: Training](#phase-2-training)
5. [Phase 3: Inference](#phase-3-inference)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Install additional dependencies
pip install transformers timm
```

### Step 1: Prepare Face Embeddings (5 minutes)

Collect 10-15 high-quality images of your target face:
- Front-facing, well-lit portraits
- Variations in expression are good
- Consistent person across all images

```bash
python examples/wanvideo/prepare_face_embeddings.py \
    --face_images_dir data/face_images/my_identity \
    --output_dir data/face_embeddings/my_identity \
    --identity_name "my_identity"
```

**Output**: Face embeddings saved to `data/face_embeddings/my_identity/`

### Step 2: Train Face LoRA (Several hours depending on dataset size)

```bash
python examples/wanvideo/model_training/train_face_lora.py \
    --face_images_dir data/face_images/my_identity \
    --dance_videos_dir data/videos_5sec \
    --pose_videos_dir data/processed/pose \
    --model_path models/Wan-AI/Wan2.1-VACE-1.3B \
    --output_dir models/face_lora/my_identity \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --lora_rank 64
```

**Output**: Trained LoRA weights in `models/face_lora/my_identity/`

### Step 3: Generate Videos with Face Identity

```bash
python examples/wanvideo/face_identity_inference.py \
    --face_images_dir data/face_images/my_identity \
    --face_lora_path models/face_lora/my_identity/face_lora_final.pth \
    --pose_video data_infer/processed/pose/dance-1_1_pose.mp4 \
    --output results/my_dance_with_face.mp4 \
    --face_scale 1.0
```

**Output**: Video with your target face dancing!

---

## 🏗️ System Architecture

### Components

1. **Face Embedding Extractor** (`diffsynth/utils/face_embedding.py`)
   - Uses CLIP or DINOv3 to extract facial features
   - Aggregates multiple images into a single identity embedding
   - L2-normalized 768-dimensional vectors

2. **Face Identity Mapper** (`diffsynth/models/face_identity.py`)
   - Projects face embeddings to model's conditioning space
   - Lightweight MLP (2-3 layers)
   - Combined with LoRA for efficient training

3. **Pipeline Integration** (`diffsynth/pipelines/wan_video.py`)
   - `WanVideoUnit_FaceEmbedding`: Processes face embeddings
   - `WanVideoUnit_FaceIdentityLora`: Loads trained adapters
   - `model_fn_wan_video`: Injects face conditioning into generation

### Data Flow

```
Face Images → CLIP Encoder → Identity Embedding
                                    ↓
                            Face Mapper + LoRA
                                    ↓
Pose Video → VACE Encoder ← Face Conditioning → DiT Model → Generated Video
                Text Prompt ←──────────────────┘
```

---

## 📊 Phase 1: Data Preparation

### 1.1 Collect Face Reference Images

**Requirements:**
- **Quantity**: 10-15 images (minimum 3, maximum 20)
- **Quality**: High-resolution, well-lit, sharp
- **Consistency**: Same person across all images
- **Variety**: Different expressions/angles help

**Good examples:**
- Professional headshots
- Clear selfies with good lighting
- Portrait photos

**Avoid:**
- Blurry or low-resolution images
- Heavy makeup or filters
- Occluded faces (sunglasses, masks)
- Multiple people in frame

### 1.2 Organize Directory Structure

```
data/
├── face_images/
│   └── my_identity/
│       ├── photo_001.jpg
│       ├── photo_002.jpg
│       ├── ...
│       └── photo_015.jpg
├── videos_5sec/           # Your dance videos
│   ├── dance_001.mp4
│   └── ...
└── processed/
    └── pose/              # Corresponding pose videos
        ├── dance_001_pose.mp4
        └── ...
```

### 1.3 Extract Face Embeddings

```bash
python examples/wanvideo/prepare_face_embeddings.py \
    --face_images_dir data/face_images/my_identity \
    --output_dir data/face_embeddings/my_identity \
    --identity_name "my_identity" \
    --encoder clip \
    --aggregation weighted_mean
```

**Parameters:**
- `--encoder`: `clip` (recommended) or `dinov3`
- `--aggregation`: `weighted_mean` (best), `mean`, or `median`

**Output Files:**
- `identity_embedding.pt`: Aggregated face embedding (use this for inference)
- `embeddings/`: Individual embeddings from each image
- `metadata.pt`: Dataset metadata

---

## 🎓 Phase 2: Training

### 2.1 Training Strategy

The system uses a **two-stage training approach**:

1. **Stage 1 (Epochs 0-20)**: Train both mapper and LoRA
   - Learn general face-to-conditioning projection
   - Adapt to your specific identity

2. **Stage 2 (Epochs 20-100)**: Freeze mapper, train LoRA only
   - Fine-tune for better consistency
   - Prevents overfitting to training videos

### 2.2 Run Training

```bash
python examples/wanvideo/model_training/train_face_lora.py \
    --face_images_dir data/face_images/my_identity \
    --dance_videos_dir data/videos_5sec \
    --pose_videos_dir data/processed/pose \
    --model_path models/Wan-AI/Wan2.1-VACE-1.3B \
    --output_dir models/face_lora/my_identity \
    --num_epochs 100 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 64 \
    --freeze_mapper_after_epochs 20 \
    --num_frames 65 \
    --height 480 \
    --width 832
```

**Key Parameters:**

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `num_epochs` | Total training epochs | 100 |
| `learning_rate` | Initial learning rate | 1e-4 |
| `lora_rank` | LoRA rank (higher = more capacity) | 64 |
| `freeze_mapper_after_epochs` | When to freeze mapper | 20 |
| `validate_every` | Validation frequency | 10 |
| `save_every` | Checkpoint save frequency | 10 |

### 2.3 Training Tips

**For Better Results:**
- Use more face images (10-15 is optimal)
- Train for 100+ epochs
- Use cosine learning rate schedule (built-in)
- Monitor validation videos to check for overfitting

**Hardware Requirements:**
- GPU: 24GB+ VRAM recommended (RTX 3090/4090, A6000)
- Can train with lower VRAM by:
  - Reducing `num_frames`
  - Using gradient checkpointing
  - Training with mixed precision

**Training Time Estimates:**
- 10 videos: ~2-3 hours (RTX 4090)
- 50 videos: ~8-10 hours
- 100 videos: ~15-20 hours

### 2.4 Monitoring Training

Training outputs:
```
Epoch 1/100 - Average Loss: 0.023456 - LR: 1.00e-04
>>> Running validation at epoch 10
  Validation video saved to: models/face_lora/my_identity/validation_epoch_10/validation.mp4
>>> Saved checkpoint: models/face_lora/my_identity/face_lora_epoch_10.pth
```

Check validation videos to ensure:
- Face is becoming consistent
- Identity is preserved
- No artifacts or distortions

---

## 🎬 Phase 3: Inference

### 3.1 Single Video Generation

```bash
python examples/wanvideo/face_identity_inference.py \
    --face_images_dir data/face_images/my_identity \
    --face_lora_path models/face_lora/my_identity/face_lora_final.pth \
    --pose_video data_infer/processed/pose/dance-1_1_pose.mp4 \
    --output results/my_dance.mp4 \
    --prompt "A person is dancing following the pose video exactly. Natural and smooth movement." \
    --face_scale 1.0 \
    --cfg_scale 5.0 \
    --num_inference_steps 50 \
    --seed 42
```

### 3.2 Batch Generation

Process multiple pose videos with the same face:

```bash
python examples/wanvideo/face_identity_inference.py \
    --batch \
    --face_images_dir data/face_images/my_identity \
    --face_lora_path models/face_lora/my_identity/face_lora_final.pth \
    --pose_videos_dir data_infer/processed/pose \
    --output_dir results/batch_generation \
    --face_scale 1.0
```

### 3.3 Key Parameters

**Face Control:**
- `--face_scale`: Controls face identity strength
  - `0.5`: Subtle face influence
  - `1.0`: Balanced (recommended)
  - `1.5`: Strong face influence

**Generation Quality:**
- `--cfg_scale`: Classifier-free guidance
  - `3.0-5.0`: More creative
  - `5.0-7.0`: Balanced
  - `7.0-10.0`: Stronger prompt adherence

- `--num_inference_steps`: 
  - `30-40`: Faster, slightly lower quality
  - `50`: Balanced (recommended)
  - `70-100`: Highest quality, slower

**Prompt Engineering:**

Good prompts:
```
"A person is dancing following the pose video exactly. Natural and smooth movement. Clear face with consistent identity."
```

Negative prompts:
```
"blurry face, inconsistent appearance, static, multiple people, distorted, low quality"
```

---

## 🔧 Advanced Usage

### Using Pre-extracted Embeddings

If you already have embeddings saved:

```python
import torch
from diffsynth.pipelines.wan_video import WanVideoPipeline

# Load pre-saved embedding
identity_embedding = torch.load("data/face_embeddings/my_identity/identity_embedding.pt")

# Use in pipeline
pipe = WanVideoPipeline.from_pretrained(...)
video = pipe(
    face_embedding=identity_embedding,
    face_scale=1.0,
    ...
)
```

### Face Blending

Blend multiple identities:

```python
from diffsynth.models.face_identity import blend_face_embeddings

# Load multiple identities
identity1 = torch.load("identity1.pt")
identity2 = torch.load("identity2.pt")

# Blend (50-50)
blended = blend_face_embeddings(
    [identity1, identity2],
    weights=torch.tensor([0.5, 0.5])
)
```

### Face Interpolation

Smooth transitions between faces:

```python
from diffsynth.models.face_identity import interpolate_face_embeddings

# Interpolate
interpolated = interpolate_face_embeddings(
    identity1, identity2, t=0.5  # 0=identity1, 1=identity2
)
```

### Custom Mapper Configuration

Train with different architecture:

```python
from diffsynth.models.face_identity import FaceConditioningBlock

face_block = FaceConditioningBlock(
    face_embedding_dim=768,
    conditioning_dim=768,
    hidden_dim=2048,          # Larger hidden layer
    num_mapper_layers=3,      # More layers
    use_lora=True,
    lora_rank=128,            # Higher rank
    dropout=0.2,              # More dropout
)
```

---

## 🔍 Troubleshooting

### Issue: Face is not consistent across frames

**Solutions:**
1. Increase `face_scale` (try 1.2-1.5)
2. Use more reference images (aim for 12-15)
3. Train for more epochs
4. Increase LoRA rank to 128

### Issue: Face looks distorted or unnatural

**Solutions:**
1. Decrease `face_scale` (try 0.7-0.9)
2. Improve reference image quality
3. Adjust negative prompt to include "distorted, unnatural, blurry face"
4. Reduce CFG scale

### Issue: Training is too slow

**Solutions:**
1. Reduce `num_frames` to 33 or 49
2. Use fewer training videos initially
3. Reduce `num_inference_steps` during training (use 20)
4. Enable gradient checkpointing (if available)

### Issue: Out of memory during training

**Solutions:**
1. Reduce `batch_size` to 1
2. Reduce `num_frames`
3. Use smaller model variant
4. Enable CPU offloading (if supported)

### Issue: Generated videos don't follow pose

**Solutions:**
1. Check VACE conditioning is working
2. Adjust prompt to emphasize pose following
3. Ensure pose videos are correctly preprocessed
4. Try lower `face_scale` to give pose more influence

### Issue: Face looks good but movement is stiff

**Solutions:**
1. This is a limitation of the pose conditioning
2. Use higher quality pose videos
3. Adjust prompts to emphasize "natural, smooth movement"
4. Lower `cfg_scale` for more creative movement

---

## 📝 Best Practices

### Data Quality
✅ **DO:**
- Use high-resolution face images (at least 512x512)
- Ensure good lighting in face images
- Use diverse expressions (smile, neutral, etc.)
- Keep consistent identity across all images

❌ **DON'T:**
- Use heavily edited/filtered images
- Mix different people in face dataset
- Use images with occlusions
- Use low-resolution or blurry images

### Training
✅ **DO:**
- Start with a small dataset to test
- Monitor validation videos regularly
- Save checkpoints frequently
- Use weighted_mean aggregation for embeddings

❌ **DON'T:**
- Train on too few face images (< 5)
- Skip validation checks
- Use too high learning rates
- Overtrain (watch for identity drift)

### Inference
✅ **DO:**
- Start with face_scale=1.0 and adjust
- Use detailed prompts
- Generate multiple seeds to find best results
- Use negative prompts effectively

❌ **DON'T:**
- Use extreme face_scale values (< 0.3 or > 2.0)
- Expect perfection on first try
- Ignore pose video quality
- Use generic prompts

---

## 📚 Technical Details

### Face Embedding Dimensions
- **CLIP**: 768-dimensional vectors
- **DINOv3**: 1024-dimensional vectors
- **Normalized**: L2 normalization applied

### Model Architecture
```
Face Embedding (768)
    ↓
Linear (768 → 1024) + GELU + Dropout
    ↓
Linear (1024 → 768)
    ↓
LoRA (rank 64)
    ↓
Face Conditioning (768)
```

### Memory Requirements
- **Training**: ~20GB VRAM (for 65 frames, 480p)
- **Inference**: ~16GB VRAM
- **Face Embedding Extraction**: ~4GB VRAM

---

## 🆘 Support & Community

For issues, questions, or contributions:
- Check existing issues in the repository
- Review this documentation thoroughly
- Check the troubleshooting section
- Provide detailed error messages and logs when asking for help

---

## 📜 License & Credits

This implementation builds upon:
- WAN Video Generation Pipeline
- CLIP (OpenAI)
- LoRA (Microsoft)
- DiffSynth-Studio

---

## 🎉 Congratulations!

You now have a complete system for generating videos with consistent face identity. Experiment with different faces, poses, and settings to get the best results!

**Happy generating! 🚀**
