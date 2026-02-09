"""
Batch LoRA Inference Testing Script
Runs multiple LoRA inference jobs with different parameter configurations
"""

import torch
import wandb
import os
from pathlib import Path
from datetime import datetime
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


class BatchLoRAInference:
    def __init__(
        self,
        lora_path: str = "models/train/Wan2.1-VACE-1.3B_lora/step-2800.safetensors",
        lora_alpha: float = 1.0,
        model_configs: list = None,
        base_save_dir: str = None,
        wandb_project: str = "skeleton2Video-batch",
        device: str = "cuda",
        torch_dtype = torch.bfloat16,
    ):
        self.lora_path = lora_path
        self.lora_alpha = lora_alpha
        self.device = device
        self.torch_dtype = torch_dtype
        self.wandb_project = wandb_project
        
        # Set up base save directory
        if base_save_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.base_save_dir = Path(f"results/batch_lora_tests_{timestamp}")
        else:
            self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Default model configs
        if model_configs is None:
            self.model_configs = [
                ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"),
                ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(path="models/Wan-AI/Wan2.1-VACE-1.3B/Wan2.1_VAE.pth"),
            ]
        else:
            self.model_configs = model_configs
        
        self.pipe = None
        self.default_params = {}
    
    def load_pipeline(self):
        """Load or reload the pipeline with LoRA"""
        print("Loading pipeline...")
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self.torch_dtype,
            device=self.device,
            model_configs=self.model_configs,
        )
        self.pipe.load_lora(self.pipe.vace, self.lora_path, alpha=self.lora_alpha)
        print("Pipeline loaded successfully")
    
    def unload_pipeline(self):
        """Unload pipeline to free memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
    
    def load_video_frames(self, video_path: str, num_frames: int, height: int, width: int):
        """Load video frames"""
        video = VideoData(video_path, height=height, width=width)
        return [video[i] for i in range(min(num_frames, len(video)))]
    
    def load_reference_images(self, ref_image_paths: list, height: int, width: int):
        """Load reference images"""
        ref_images = []
        for img_path in ref_image_paths:
            img = VideoData(img_path, height=height, width=width)[0]
            ref_images.append(img)
        return ref_images
    
    def run_inference(
        self,
        test_name: str,
        config_name: str,
        override_params: dict = None,
    ):
        """
        Run a single inference with given parameters
        
        Note: Ensure default_params is populated before calling this method.
        Use batch_runner.default_params.update({...}) to set base parameters.
        
        Args:
            test_name: Name of the test (e.g., "test1_guidance")
            config_name: Configuration name (e.g., "guide_scale_5.0")
            override_params: Parameters to override defaults
        """
        # Merge parameters
        params = self.default_params.copy()
        if override_params:
            params.update(override_params)
        
        # Create output directory
        output_dir = self.base_save_dir / test_name / config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "out_video.mp4"
        
        # Create W&B run name
        run_name = f"lora-{test_name}-{config_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize W&B
        wandb.init(
            project=self.wandb_project,
            name=run_name,
            config={
                "test_name": test_name,
                "config_name": config_name,
                "lora_path": self.lora_path,
                "lora_alpha": self.lora_alpha,
                **params,
            },
            reinit=True,
        )
        
        print(f"\n{'='*60}")
        print(f"Running: {test_name}/{config_name}")
        print(f"Output: {output_path}")
        print(f"{'='*60}\n")
        
        try:
            # Load video and reference images
            vace_video = self.load_video_frames(
                params["src_video"],
                params["num_frames"],
                params["height"],
                params["width"]
            )
            
            vace_reference_images = self.load_reference_images(
                params["src_ref_images"],
                params["height"],
                params["width"]
            )
            
            # Run inference
            video = self.pipe(
                prompt=params["prompt"],
                negative_prompt=params["negative_prompt"],
                vace_video=vace_video,
                vace_reference_image=vace_reference_images,
                num_frames=params["num_frames"],
                height=params["height"],
                width=params["width"],
                seed=params["seed"],
                cfg_scale=params["cfg_scale"],
                num_inference_steps=params["num_inference_steps"],
                sigma_shift=params["sigma_shift"],
                tiled=True,
            )
            
            # Save video locally
            save_video(video, str(output_path), fps=params["fps"], quality=params["quality"])
            print(f"Video saved to: {output_path}")
            
            # Log to W&B
            wandb.log({
                "video_output": wandb.Video(
                    str(output_path),
                    format="mp4",
                    caption=f"{test_name}/{config_name}"
                ),
                "cfg_scale": params["cfg_scale"],
                "num_inference_steps": params["num_inference_steps"],
                "sigma_shift": params["sigma_shift"],
            })
            
            print(f"✓ Successfully completed: {test_name}/{config_name}")
            
        except Exception as e:
            print(f"✗ Error in {test_name}/{config_name}: {str(e)}")
            wandb.log({"error": str(e)})
        finally:
            wandb.finish()
    
    def run_guidance_scale_tests(self, guide_scales: list, fixed_steps: int, fixed_shift: float):
        """
        Test with different guidance scale values
        
        Args:
            guide_scales: List of guidance scale values to test
            fixed_steps: Fixed number of inference steps for all runs
            fixed_shift: Fixed sigma shift value for all runs
        """
        print("\n" + "="*70)
        print("TEST 1: Guidance Scale Variations")
        print("="*70)
        
        for scale in guide_scales:
            self.run_inference(
                test_name="test1_guidance",
                config_name=f"guide_scale_{scale}",
                override_params={
                    "cfg_scale": scale,
                    "num_inference_steps": fixed_steps,
                    "sigma_shift": fixed_shift,
                }
            )
    
    def run_sampling_steps_tests(self, sampling_steps: list, fixed_cfg_scale: float, fixed_shift: float):
        """
        Test with different sampling steps
        
        Args:
            sampling_steps: List of sampling step values to test
            fixed_cfg_scale: Fixed guidance scale for all runs
            fixed_shift: Fixed sigma shift value for all runs
        """
        print("\n" + "="*70)
        print("TEST 2: Sampling Steps Variations")
        print("="*70)
        
        for steps in sampling_steps:
            self.run_inference(
                test_name="test2_steps",
                config_name=f"sample_steps_{steps}",
                override_params={
                    "cfg_scale": fixed_cfg_scale,
                    "num_inference_steps": steps,
                    "sigma_shift": fixed_shift,
                }
            )
    
    def run_sample_shift_tests(self, sample_shifts: list, fixed_cfg_scale: float, fixed_steps: int):
        """
        Test with different sample shift values
        
        Args:
            sample_shifts: List of sigma shift values to test
            fixed_cfg_scale: Fixed guidance scale for all runs
            fixed_steps: Fixed number of inference steps for all runs
        """
        print("\n" + "="*70)
        print("TEST 3: Sample Shift Variations")
        print("="*70)
        
        for shift in sample_shifts:
            self.run_inference(
                test_name="test3_shift",
                config_name=f"sample_shift_{int(shift)}",
                override_params={
                    "cfg_scale": fixed_cfg_scale,
                    "num_inference_steps": fixed_steps,
                    "sigma_shift": shift,
                }
            )
    
    def run_prompt_variations_tests(self, prompts: list, fixed_cfg_scale: float, fixed_steps: int, fixed_shift: float):
        """
        Test with different prompt variations
        
        Args:
            prompts: List of prompt strings to test
            fixed_cfg_scale: Fixed guidance scale for all runs
            fixed_steps: Fixed number of inference steps for all runs
            fixed_shift: Fixed sigma shift value for all runs
        """
        print("\n" + "="*70)
        print("TEST 4: Prompt Variations")
        print("="*70)
        
        for i, prompt in enumerate(prompts, 1):
            self.run_inference(
                test_name="test4_prompt_variations",
                config_name=f"prompt_{i}",
                override_params={
                    "prompt": prompt,
                    "cfg_scale": fixed_cfg_scale,
                    "num_inference_steps": fixed_steps,
                    "sigma_shift": fixed_shift,
                }
            )
    
    def print_summary(self, total_runs: int):
        """
        Print batch testing summary
        
        Args:
            total_runs: Total number of inference runs completed
        """
        print("\n" + "="*70)
        print("BATCH TESTING COMPLETED")
        print("="*70)
        print(f"Results saved to: {self.base_save_dir}")
        print(f"Total runs: {total_runs}")
        print("\nTo view results:")
        print(f"  ls -lh {self.base_save_dir}/*/*/out_video.mp4")
        print("="*70 + "\n")
