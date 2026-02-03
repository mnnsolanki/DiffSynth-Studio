# Face Identity for WAN Video - Quick Start

Generate dance videos with consistent facial identity using dual-input training.

## 🎯 What This Does

Transform AIST dance poses into videos where **your chosen person** performs the dance with a **clear, consistent face**.

**Before**: Multi-person dance videos → Blurry/inconsistent faces  
**After**: Same dance poses → Your target person with clear, consistent face

## ⚡ Quick Start (3 Steps)

### 1. Extract Face Embeddings (5 min)

```bash
# Put 10-15 face images in data/face_images/my_identity/
python examples/wanvideo/prepare_face_embeddings.py \
    --face_images_dir data/face_images/my_identity \
    --output_dir data/face_embeddings/my_identity
```

### 2. Train Face LoRA (2-10 hours)

```bash
python examples/wanvideo/model_training/train_face_lora.py \
    --face_images_dir data/face_images/my_identity \
    --dance_videos_dir data/videos_5sec \
    --pose_videos_dir data/processed/pose \
    --model_path models/Wan-AI/Wan2.1-VACE-1.3B \
    --output_dir models/face_lora/my_identity \
    --num_epochs 100
```

### 3. Generate Videos (2-5 min per video)

```bash
python examples/wanvideo/face_identity_inference.py \
    --face_images_dir data/face_images/my_identity \
    --face_lora_path models/face_lora/my_identity/face_lora_final.pth \
    --pose_video data_infer/processed/pose/dance-1_1_pose.mp4 \
    --output results/my_dance.mp4
```

## 📋 Requirements

```bash
pip install transformers timm
```

**Hardware:**
- GPU with 16GB+ VRAM (training: 20GB+ recommended)
- CUDA-compatible GPU

## 📁 Data Preparation

### Face Images
- **Quantity**: 10-15 images (minimum 3)
- **Quality**: High-res, well-lit, front-facing
- **Format**: JPG or PNG

**Example structure:**
```
data/face_images/my_identity/
├── face_001.jpg
├── face_002.jpg
...
└── face_015.jpg
```

### Dance Videos
Your existing AIST dataset:
```
data/
├── videos_5sec/          # Dance videos
│   └── dance_001.mp4
└── processed/pose/       # Pose videos
    └── dance_001_pose.mp4
```

## 🎛️ Key Parameters

### Training
- `--num_epochs 100`: More epochs = better quality
- `--lora_rank 64`: Higher rank = more capacity
- `--learning_rate 1e-4`: Adjust if training is unstable

### Inference
- `--face_scale 1.0`: Controls face identity strength (0.5-1.5)
- `--cfg_scale 5.0`: Prompt adherence (3.0-10.0)
- `--num_inference_steps 50`: Quality vs speed (30-100)

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Face not consistent | Increase `--face_scale` to 1.2-1.5 |
| Face looks distorted | Decrease `--face_scale` to 0.7-0.9 |
| Out of memory | Reduce `--num_frames` or `--batch_size` |
| Training too slow | Reduce `--num_inference_steps` during training |

## 📖 Full Documentation

See [Face Identity Training Guide](docs/face_identity_training_guide.md) for:
- Detailed architecture explanation
- Advanced configuration options
- Best practices and tips
- Complete troubleshooting guide

## 🎨 What You Can Do

1. **Single Identity Videos**: One person dancing across multiple poses
2. **Batch Generation**: Process all your pose videos with same face
3. **Face Blending**: Mix multiple identities (creative effects)
4. **Face Interpolation**: Smooth transitions between identities

## 🚀 Example Workflow

```bash
# 1. Prepare embeddings
python examples/wanvideo/prepare_face_embeddings.py \
    --face_images_dir data/faces/person_a \
    --output_dir data/embeddings/person_a

# 2. Train (can skip if you have pre-trained weights)
python examples/wanvideo/model_training/train_face_lora.py \
    --face_images_dir data/faces/person_a \
    --dance_videos_dir data/videos_5sec \
    --pose_videos_dir data/processed/pose \
    --model_path models/Wan-AI/Wan2.1-VACE-1.3B \
    --output_dir models/lora/person_a \
    --num_epochs 100

# 3. Generate multiple videos
python examples/wanvideo/face_identity_inference.py \
    --batch \
    --face_images_dir data/faces/person_a \
    --face_lora_path models/lora/person_a/face_lora_final.pth \
    --pose_videos_dir data_infer/processed/pose \
    --output_dir results/person_a_dances
```

## 💡 Tips

**For Best Results:**
1. Use 10-15 high-quality face images
2. Train for 100+ epochs
3. Monitor validation videos during training
4. Adjust `face_scale` based on results
5. Use descriptive prompts

**Recommended Settings:**
- Face Scale: 1.0 (adjust 0.8-1.2 as needed)
- CFG Scale: 5.0
- Inference Steps: 50
- Training Epochs: 100

## 🔗 Related Files

- **Face Embedding Utils**: `diffsynth/utils/face_embedding.py`
- **Face Mapper Model**: `diffsynth/models/face_identity.py`
- **Pipeline Integration**: `diffsynth/pipelines/wan_video.py`
- **Training Script**: `examples/wanvideo/model_training/train_face_lora.py`
- **Inference Script**: `examples/wanvideo/face_identity_inference.py`

## ⚠️ Important Notes

1. **Training Time**: Varies based on dataset size (2-20 hours typical)
2. **VRAM Requirements**: 20GB+ for training, 16GB+ for inference
3. **Quality**: Results depend heavily on face image quality
4. **Consistency**: More face images = better consistency

## 🎯 Next Steps

1. ✅ Collect face images
2. ✅ Extract embeddings
3. ✅ Train face LoRA
4. ✅ Generate videos
5. 🔄 Iterate on face_scale and prompts
6. 🎨 Experiment with different identities

---

**Need Help?** See the full documentation or check the troubleshooting section.

**Ready to start?** Follow the 3 steps above and you'll be generating personalized dance videos in no time! 🚀
