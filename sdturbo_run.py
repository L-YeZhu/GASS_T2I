from diffusers import AutoPipelineForText2Image
import torch
from pathlib import Path
from PIL import Image
import time
import math

def sanitize_filename(text):
    return text.replace(" ", "_").replace(",", "").replace(".", "").lower()[:50]

def generate_images_turbo(
    prompt_file="./datasets/drawbench_prompts.txt", 
    output_dir="./results/sdturbo/drawbench",
    model_id="stabilityai/sd-turbo",
    image_size=(512, 512),  # SD-Turbo 默认分辨率
    num_inference_steps=1,  # Turbo 通常一步
    guidance_scale=0.0,
    seed=42,
    max_batch_size=8,
):
    # For CC12M: prompt + count
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    generator = torch.Generator(device="cuda").manual_seed(seed)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_time = 0.0

    for i, line in enumerate(lines):

    	## for CC12M dataste only
        # prompt, num_images = line.rsplit("+", 1)
        # prompt = prompt.strip()
        # num_images = int(num_images)

        ## for drawbench
        prompt = line
        num_images = 8



        subdir = output_root / f"{i:03d}_{sanitize_filename(prompt)}"
        subdir.mkdir(parents=True, exist_ok=True)

        if all((subdir / f"sample_{j:02d}.png").exists() for j in range(num_images)):
            print(f"[{i+1}/{len(lines)}] Skipped (already exists): {prompt}")
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
        # duration = time.time() - start
        # print(f"→ {duration:.2f} sec total, {duration/num_images:.2f} sec/image")

if __name__ == "__main__":
    generate_images_turbo()