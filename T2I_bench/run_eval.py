#!/usr/bin/env python3
"""
T2I Evaluation Script
Generates images using multiple models (Gemini 2.5-flash, Flux.1, Qwen) 
and saves them in organized timestamped folders.
"""

import argparse
import os
import json
import pandas as pd
import torch
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

# Gemini imports
from google import genai
from PIL import Image
from io import BytesIO

# Diffusers imports  
from diffusers import FluxPipeline, DiffusionPipeline


class ImageGenerator:
    """Base class for image generators"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config['model_name']
        
    def generate(self, prompt: str, num_generations: int = 4) -> List[Image.Image]:
        raise NotImplementedError
    
    def generate_specific(self, prompt: str, indices: List[int]) -> Dict[int, Image.Image]:
        """Generate specific images by index - default implementation"""
        all_images = self.generate(prompt, max(indices) + 1 if indices else 0)
        return {idx: all_images[idx] for idx in indices if idx < len(all_images)}
        
    def setup(self):
        """Setup model and dependencies"""
        raise NotImplementedError


class GeminiImageGenerator(ImageGenerator):
    """Gemini 2.5-flash image generator"""
    
    def setup(self):
        # Setup Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.client = genai.Client()
    
    def generate(self, prompt: str, num_generations: int = 4) -> List[Image.Image]:
        images = []
        for i in range(num_generations):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=[prompt],
                )
                
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image = Image.open(BytesIO(part.inline_data.data))
                        images.append(image)
                        break
                        
            except Exception as e:
                print(f"Error generating image {i} with Gemini: {e}")
                # Create placeholder image on error
                placeholder = Image.new('RGB', (1024, 1024), color='gray')
                images.append(placeholder)
                
        return images
    
    def generate_specific(self, prompt: str, indices: List[int]) -> Dict[int, Image.Image]:
        """Generate specific images by index"""
        result = {}
        for idx in indices:
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=[prompt],
                )
                
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        image = Image.open(BytesIO(part.inline_data.data))
                        result[idx] = image
                        break
                        
            except Exception as e:
                print(f"Error generating image {idx} with Gemini: {e}")
                placeholder = Image.new('RGB', (1024, 1024), color='gray')
                result[idx] = placeholder
        
        return result


class FluxImageGenerator(ImageGenerator):
    """Flux.1 image generator using diffusers"""
    
    def setup(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Load Flux pipeline
        model_path = self.config.get('model_path', 'black-forest-labs/FLUX.1-dev')
        self.pipe = FluxPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.pipe.to(device)
        print(f"Flux model loaded on {device}")
    
    def generate(self, prompt: str, num_generations: int = 4) -> List[Image.Image]:
        images = []
        for i in range(num_generations):
            try:
                seed = 3407 + i  # Match UniGenBench seed pattern
                generator = torch.Generator(device=self.device).manual_seed(seed)
                
                image = self.pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=3.5,
                    num_inference_steps=30,
                    max_sequence_length=512,
                    generator=generator
                ).images[0]
                
                images.append(image)
                
            except Exception as e:
                print(f"Error generating image {i} with Flux: {e}")
                placeholder = Image.new('RGB', (1024, 1024), color='gray')
                images.append(placeholder)
                
        return images


class QwenImageGenerator(ImageGenerator):
    """Qwen-Image generator using diffusers"""
    
    def setup(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Load Qwen-Image model
        model_path = self.config.get('model_path', 'Qwen/Qwen-Image')
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.pipe.to(device)
        print(f"Qwen-Image model loaded on {device}")
    
    def generate(self, prompt: str, num_generations: int = 4) -> List[Image.Image]:
        images = []
        for i in range(num_generations):
            try:
                seed = 3407 + i
                generator = torch.Generator(device=self.device).manual_seed(seed)
                
                image = self.pipe(
                    prompt,
                    height=1024,
                    width=1024,
                    guidance_scale=7.5,
                    num_inference_steps=30,
                    generator=generator
                ).images[0]
                
                images.append(image)
                
            except Exception as e:
                print(f"Error generating image {i} with Qwen-Image: {e}")
                placeholder = Image.new('RGB', (1024, 1024), color='gray')
                images.append(placeholder)
                
        return images


def load_test_prompts(csv_path: str) -> pd.DataFrame:
    """Load test prompts from CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test prompts file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} test prompts from {csv_path}")
    
    if len(df) != 600:
        print(f"Warning: Expected 600 prompts, got {len(df)}")
    
    return df


def create_output_directory(model_name: str, num_generations: int, start_index: int = 0, 
                           end_index: int = None, total_prompts: int = 600, 
                           base_dir: str = "outputs") -> str:
    """Create output directory with fixed naming scheme"""
    # Base name: <model_name>_g<num_generations>
    dir_name = f"{model_name}_g{num_generations}"
    
    # Add range suffix if not default (0 to total)
    if start_index != 0 or (end_index is not None and end_index != total_prompts):
        end_idx = end_index if end_index is not None else total_prompts
        dir_name += f"[{start_index}:{end_idx}]"
    
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def check_existing_images(output_dir: str, prompt_index: int, num_generations: int) -> List[int]:
    """Check which generation indices are missing or incomplete for a prompt"""
    missing_indices = []
    
    for gen_idx in range(num_generations):
        image_path = os.path.join(output_dir, f"{prompt_index}_{gen_idx}.png")
        if not os.path.exists(image_path):
            missing_indices.append(gen_idx)
        else:
            # Validate image file is not corrupted/empty
            try:
                with Image.open(image_path) as img:
                    img.verify()
            except Exception:
                # Image is corrupted, mark for regeneration
                missing_indices.append(gen_idx)
    
    return missing_indices


def should_skip_prompt(output_dir: str, prompt_index: int, num_generations: int) -> bool:
    """Check if prompt should be skipped (all images exist and are valid)"""
    missing = check_existing_images(output_dir, prompt_index, num_generations)
    return len(missing) == 0


def get_model_generator(model_name: str, config: Dict[str, Any]) -> ImageGenerator:
    """Get appropriate image generator based on model name"""
    config['model_name'] = model_name
    
    if model_name == "gemini-2.5-flash":
        return GeminiImageGenerator(config)
    elif model_name == "flux.1":
        return FluxImageGenerator(config)
    elif model_name == "qwen":
        return QwenImageGenerator(config)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="T2I Evaluation Script")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["gemini-2.5-flash", "flux.1", "qwen"],
        required=True,
        help="Model to use for image generation"
    )
    parser.add_argument(
        "--num_generation_per_prompt",
        type=int,
        default=4,
        help="Number of images to generate per prompt"
    )
    parser.add_argument(
        "--prompts_csv",
        type=str,
        default="UniGenBench/data/test_prompts_en.csv",
        help="Path to test prompts CSV file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Base directory for outputs"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model (for diffusers models)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start from specific prompt index (for resuming)"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End at specific prompt index"
    )
    
    args = parser.parse_args()
    
    # Load test prompts
    df = load_test_prompts(args.prompts_csv)
    
    # Create output directory with fixed naming
    output_dir = create_output_directory(
        args.model, 
        args.num_generation_per_prompt, 
        start_idx, 
        end_idx, 
        len(df), 
        args.output_dir
    )
    print(f"Output directory: {output_dir}")
    
    # Setup model configuration
    config = {
        'model_path': args.model_path,
        'output_dir': output_dir
    }
    
    # Initialize generator
    generator = get_model_generator(args.model, config)
    generator.setup()
    
    # Determine range
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else len(df)
    
    print(f"Generating images for prompts {start_idx} to {end_idx-1}")
    
    # Save configuration
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump({
            'model': args.model,
            'num_generation_per_prompt': args.num_generation_per_prompt,
            'prompts_csv': args.prompts_csv,
            'start_index': start_idx,
            'end_index': end_idx,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Generate images with auto-resume
    skipped_count = 0
    generated_count = 0
    
    for idx in tqdm(range(start_idx, end_idx), desc="Generating images"):
        try:
            row = df.iloc[idx]
            prompt_index = row['index']
            prompt = row['prompt_en']
            
            # Check if we should skip this prompt (all images already exist)
            if should_skip_prompt(output_dir, prompt_index, args.num_generation_per_prompt):
                skipped_count += 1
                continue
            
            print(f"\nProcessing prompt {prompt_index}: {prompt[:100]}...")
            
            # Check which specific images need to be generated
            missing_indices = check_existing_images(output_dir, prompt_index, args.num_generation_per_prompt)
            
            if missing_indices:
                print(f"Generating {len(missing_indices)} missing images: {missing_indices}")
                
                # Generate only the missing images
                images_dict = generator.generate_specific(prompt, missing_indices)
                
                # Save the generated images
                for idx, image in images_dict.items():
                    image_path = os.path.join(output_dir, f"{prompt_index}_{idx}.png")
                    image.save(image_path)
                    generated_count += 1
                    print(f"Saved image {prompt_index}_{idx}.png")
            
        except Exception as e:
            print(f"Error processing prompt {idx}: {e}")
            continue
    
    print(f"\nGeneration complete!")
    print(f"Skipped prompts (already complete): {skipped_count}")
    print(f"Generated new images: {generated_count}")
    
    print(f"\nCompleted! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()