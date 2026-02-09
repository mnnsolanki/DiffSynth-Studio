"""
Simple runner script for batch LoRA inference
Customize parameters here and run the batch tests
"""

from batch_lora_inference import BatchLoRAInference


# ============================================================================
# Configuration
# ============================================================================

# LoRA model settings
LORA_PATH = "models/train/Wan2.1-VACE-1.3B_lora/step-2800.safetensors"
LORA_ALPHA = 1.0

# W&B settings
WANDB_PROJECT = "skeleton2Video-batch-lora"

# Output directory (None for auto-generated timestamp)
BASE_SAVE_DIR = None  # or specify like "results/my_batch_test"

# Default inference parameters (can be overridden per test)
DEFAULT_SRC_VIDEO = "data_infer/processed/pose/dance-4_1_pose.mp4"
DEFAULT_REF_IMAGES = [
    "data_infer/ref_img.jpg",
    "data_infer/ref_img_1.jpg",
    "data_infer/ref_img_2.jpg",
    "data_infer/ref_img_3.jpg",
    "data_infer/ref_img_4.jpg",
]
DEFAULT_PROMPT = """A high-quality video of a young adult man with medium brown skin, an oval face, and dark brown almond-shaped eyes. He has thick black wavy hair, short on the sides, and wears modern thin metal-framed rounded rectangular glasses. He is dressed in a plain black crew-neck t-shirt, slim-fit dark washed jeans, and light beige sneakers. He performs a smooth, natural dance following the skeleton pose exactly in a clean, minimalist indoor setting with soft lighting. 4k, photorealistic, consistent character."""

DEFAULT_NEGATIVE_PROMPT = """色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"""

# Video generation parameters
DEFAULT_NUM_FRAMES = 65
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_SEED = 1
DEFAULT_FPS = 15
DEFAULT_QUALITY = 9

# Fixed parameters for test variations
FIXED_CFG_SCALE = 5.0
FIXED_STEPS = 50
FIXED_SHIFT = 16.0

# Test configurations
LORA_ALPHAS = [0.3, 0.7, 1.0, 1.5]
GUIDANCE_SCALES = [3.0, 5.0, 7.5]
SAMPLING_STEPS = [20, 30, 50]
SAMPLE_SHIFTS = [8.0, 16.0]

# Prompt variations for testing
PROMPT_VARIATIONS = [
    # Prompt 1: Default/Standard
    """A high-quality video of a young adult man with medium brown skin, an oval face, and dark brown almond-shaped eyes. He has thick black wavy hair, short on the sides, and wears modern thin metal-framed rounded rectangular glasses. He is dressed in a plain black crew-neck t-shirt, slim-fit dark washed jeans, and light beige sneakers. He performs a smooth, natural dance following the skeleton pose exactly in a clean, minimalist indoor setting with soft lighting. 4k, photorealistic, consistent character.""",
    
    # Prompt 2: Cinematic
    """Cinematic footage of a man with medium brown skin and wavy black hair, wearing metal-framed glasses and a casual black t-shirt with dark jeans. The lighting is professional studio quality, casting soft shadows that define his jawline and approachable presence. He is dancing with fluid, grounded movements. The camera remains steady, capturing the fine textures of his denim jeans and beige sneakers. Extremely detailed, 8k, filmic look.""",
    
    # Prompt 3: Portrait focus
    """Detailed portrait-style video of a man with almond-shaped eyes, neatly shaped eyebrows, and a thin mustache with light stubble. He wears rounded rectangular metal glasses. The subject is wearing a black t-shirt and dark slim jeans, performing a modern dance. Focus on maintaining the exact facial proportions and short wavy hairstyle throughout the motion. The background is a stable, modern gray room. High temporal consistency, sharp focus on the face.""",
    
    # Prompt 4: Movement focus
    """A man with medium brown skin and short black hair dancing in a minimalist space. He wears a black cotton t-shirt and dark slim-fit denim jeans. As he moves, the fabric of his shirt and jeans shows realistic folds and creases. His light beige low-top sneakers stay grounded with no sliding. The motion is energetic yet controlled, matching the pose guidance perfectly. Photorealistic rendering, natural movement.""",
    
    # Prompt 5: Simple
    """A young man with an approachable presence, wearing glasses, a black t-shirt, and dark jeans, dancing gracefully. He has short wavy black hair and a slim build. The setting is clean and modern. The video captures a high-fidelity performance with smooth transitions and consistent character identity. 4k, professional video.""",
]

# Which tests to run
RUN_LORA_ALPHA_TESTS = True
RUN_GUIDANCE_TESTS = True
RUN_STEPS_TESTS = True
RUN_SHIFT_TESTS = True
RUN_PROMPT_TESTS = True


# ============================================================================
# Main
# ============================================================================

def main():
    """Run batch inference with configured parameters"""
    
    # Initialize batch runner
    batch_runner = BatchLoRAInference(
        lora_path=LORA_PATH,
        lora_alpha=LORA_ALPHA,
        base_save_dir=BASE_SAVE_DIR,
        wandb_project=WANDB_PROJECT,
    )
    
    # Set default parameters
    batch_runner.default_params.update({
        "src_video": DEFAULT_SRC_VIDEO,
        "src_ref_images": DEFAULT_REF_IMAGES,
        "prompt": DEFAULT_PROMPT,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "num_frames": DEFAULT_NUM_FRAMES,
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
        "seed": DEFAULT_SEED,
        "cfg_scale": FIXED_CFG_SCALE,
        "num_inference_steps": FIXED_STEPS,
        "sigma_shift": FIXED_SHIFT,
        "fps": DEFAULT_FPS,
        "quality": DEFAULT_QUALITY,
    })
    
    # Run tests
    try:
        if RUN_LORA_ALPHA_TESTS:
            batch_runner.run_lora_alpha_tests(
                lora_alphas=LORA_ALPHAS,
                fixed_cfg_scale=FIXED_CFG_SCALE,
                fixed_steps=FIXED_STEPS,
                fixed_shift=FIXED_SHIFT
            )
            # Reload pipeline with default alpha for subsequent tests
            if any([RUN_GUIDANCE_TESTS, RUN_STEPS_TESTS, RUN_SHIFT_TESTS, RUN_PROMPT_TESTS]):
                batch_runner.load_pipeline(lora_alpha=LORA_ALPHA)
        else:
            # Load pipeline if not using lora alpha tests
            batch_runner.load_pipeline(lora_alpha=LORA_ALPHA)
        
        if RUN_GUIDANCE_TESTS:
            batch_runner.run_guidance_scale_tests(
                guide_scales=GUIDANCE_SCALES,
                fixed_steps=FIXED_STEPS,
                fixed_shift=FIXED_SHIFT
            )
        
        if RUN_STEPS_TESTS:
            batch_runner.run_sampling_steps_tests(
                sampling_steps=SAMPLING_STEPS,
                fixed_cfg_scale=FIXED_CFG_SCALE,
                fixed_shift=FIXED_SHIFT
            )
        
        if RUN_SHIFT_TESTS:
            batch_runner.run_sample_shift_tests(
                sample_shifts=SAMPLE_SHIFTS,
                fixed_cfg_scale=FIXED_CFG_SCALE,
                fixed_steps=FIXED_STEPS
            )
        
        if RUN_PROMPT_TESTS:
            batch_runner.run_prompt_variations_tests(
                prompts=PROMPT_VARIATIONS,
                fixed_cfg_scale=FIXED_CFG_SCALE,
                fixed_steps=FIXED_STEPS,
                fixed_shift=FIXED_SHIFT
            )
    
    finally:
        batch_runner.unload_pipeline()
    
    # Print summary
    total_runs = (
        (len(LORA_ALPHAS) if RUN_LORA_ALPHA_TESTS else 0) +
        (len(GUIDANCE_SCALES) if RUN_GUIDANCE_TESTS else 0) +
        (len(SAMPLING_STEPS) if RUN_STEPS_TESTS else 0) +
        (len(SAMPLE_SHIFTS) if RUN_SHIFT_TESTS else 0) +
        (len(PROMPT_VARIATIONS) if RUN_PROMPT_TESTS else 0)
    )
    
    batch_runner.print_summary(total_runs)


if __name__ == "__main__":
    main()
