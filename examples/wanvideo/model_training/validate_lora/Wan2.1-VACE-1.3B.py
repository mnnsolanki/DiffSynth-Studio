import torch
import wandb
from datetime import datetime
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

ALPHA=2

PROMPT="""
A man is dancing by following the pose video exactly. Natural and smooth movements.
Clear face with consistent identity from the reference image.
Wearing blue pants, white shoes, and black t-shirt.
Consistent clothing, hair, and background throughout.
High quality, detailed, and background looks same as reference image.
"""

# Create dynamic run name with date and time
run_name = f"lora-dance-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Initialize W&B
wandb.init(
    project="skeleton2Video",
    name=run_name,
    config={
        "alpha": ALPHA,
        "prompt": PROMPT if 'PROMPT' in dir() else "TBD",
        "num_frames": 65,
        "height": 480,
        "width": 832,
        "seed": 1,
    }
)

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"),
        ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth"),
    ],
)
pipe.load_lora(pipe.vace, "models/train/Wan2.1-VACE-1.3B_lora/step-2800.safetensors", alpha=ALPHA)

video = VideoData("data_infer/processed/pose/dance-2_1_pose.mp4", height=480, width=832)
video = [video[i] for i in range(65)]

ref_img = VideoData("data_infer/ref_img.jpg", height=480, width=832)[0]
ref_img_1 = VideoData("data_infer/ref_img_1.jpg", height=480, width=832)[0]
ref_img_2 = VideoData("data_infer/ref_img_2.jpg", height=480, width=832)[0]
ref_img_3 = VideoData("data_infer/ref_img_3.jpg", height=480, width=832)[0]
ref_img_4 = VideoData("data_infer/ref_img_4.jpg", height=480, width=832)[0]

vace_reference_video=[ref_img, ref_img_1, ref_img_2, ref_img_3, ref_img_4]
# vace_reference_video=ref_img

video = pipe(
    prompt=PROMPT,
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=video, 
    vace_reference_image=vace_reference_video,
    num_frames=65, seed=1, tiled=True
)
save_video(video, "lora_dance.mp4", fps=15, quality=9)

# Log the video to W&B with proper format
wandb.log({
    "generated_video": wandb.Video("lora_dance.mp4", format="mp4"),
    "alpha": ALPHA,
})

# Also log video with caption for better visualization in charts
wandb.log({
    "video_output": wandb.Video("lora_dance.mp4", format="mp4", caption=f"Generated dance video - Alpha: {ALPHA}")
})

# Finish the W&B run
wandb.finish()
