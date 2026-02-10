"""
Example: Run a single custom LoRA inference test
Perfect for quick testing of specific parameter combinations
"""

from batch_lora_inference import BatchLoRAInference


def main():
    """Run a single custom inference test"""
    
    # Initialize batch runner
    batch_runner = BatchLoRAInference(
        lora_path="models/train/Wan2.1-VACE-1.3B_lora/step-2800.safetensors",
        lora_alpha=1.0,
        wandb_project="skeleton2Video-single-test",
        base_save_dir="results/single_test",
    )
    
    # Set default parameters (required before running inference)
    batch_runner.default_params.update({
        "src_video": "data_infer/processed/pose/dance-4_1_pose.mp4",
        "src_ref_images": [
            "data_infer/ref_img.jpg",
            "data_infer/ref_img_1.jpg",
            "data_infer/ref_img_2.jpg",
            "data_infer/ref_img_3.jpg",
            "data_infer/ref_img_4.jpg",
        ],
        "prompt": """Cinematic 4k footage of a young man with medium brown skin, wearing metal-framed glasses and a black t-shirt with dark jeans. He performs an energetic, smooth dance with natural movements following the skeleton pose exactly. Professional studio lighting, clean minimalist background. High detail, photorealistic.""",
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "num_frames": 65,
        "height": 480,
        "width": 832,
        "seed": 42,
        "cfg_scale": 6.0,
        "num_inference_steps": 60,
        "sigma_shift": 14.0,
        "fps": 15,
        "quality": 9,
    })
    
    # Load pipeline
    batch_runner.load_pipeline()
    
    try:
        # Run a single inference (parameters can still be overridden if needed)
        batch_runner.run_inference(
            test_name="custom_experiment",
            config_name="high_quality_test",
            override_params={}  # All params already set in default_params
        )
        
        print("\n Single test completed successfully!")
        print(f"Results saved to: {batch_runner.base_save_dir}/custom_experiment/high_quality_test/")
        
    finally:
        # Cleanup
        batch_runner.unload_pipeline()


if __name__ == "__main__":
    main()
