from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path
from PIL import Image
import os
import time
import math

def sanitize_filename(text):
    return text.replace("'", "").replace(" ", "_").replace(",", "").replace(".", "").replace("+", "").lower()[:50]

def generate_images(
    prompt_file="./datasets/cc12m_prompt_counts.txt",
    output_dir="./results/sd21/cc12m_new",
    model_id="stabilityai/stable-diffusion-2-1",
    num_samples=8,
    guidance_scale=5.0, # similar to previous baseline SPELL ICML25
    num_inference_steps=50, # this is default setting
    image_size=(768, 768),
    seed=42,
    max_batch_size=8,
):
    # Load prompts
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Load SD2.1 model
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    # Replace sampler with DDIM for consistency
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Disable NSFW checker for research
    pipe.safety_checker = None # Disable NSFW filter for research
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_time = 0.0

    for i, line in enumerate(lines):

        prompt, num_images = line.rsplit("+", 1)
        prompt = prompt.strip()
        num_images = int(num_images)
        subdir = output_root / f"{i:03d}_{sanitize_filename(prompt)}"
        subdir.mkdir(parents=True, exist_ok=True)

        if all((subdir / f"sample_{j:02d}.png").exists() for j in range(num_samples)):
            print(f"[{i+1}/{len(prompts)}] Skipped (already exists): {prompt}")
            continue

        print(f"[{i+1}/{len(lines)}] Generating: {prompt} → {num_images} images")
        start = time.time()

        image_idx = 0
        n_batches = math.ceil(num_images / max_batch_size)
        for _ in range(n_batches):
            n_to_generate = min(max_batch_size, num_images - image_idx)
            images = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=n_to_generate,
                generator=generator
            ).images

            for image in images:
                image.save(subdir / f"sample_{image_idx:02d}.png")
                image_idx += 1

        duration = time.time() - start
        total_time += duration
        total_images += num_images

        print(f"Prompt took {duration:.2f} sec → {duration/num_images:.2f} sec/image")
        print("Total num count:", i, num_images, total_images)

if __name__ == "__main__":
    generate_images()