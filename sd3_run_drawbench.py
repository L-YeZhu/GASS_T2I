import torch
from pathlib import Path
from PIL import Image
import time
from diffusers import StableDiffusion3Pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

def sanitize_filename(text):
    return text.replace("'", "").replace(" ", "_").replace(",", "").replace(".", "").lower()[:50]

def generate_images_sd3(
    prompt_file="./datasets/drawbench_prompts.txt",
    output_dir="./results/sd3medium/drawbench",
    model_id="stabilityai/stable-diffusion-3-medium-diffusers",
    guidance_scale=7.0,
    num_inference_steps=28,
    negative_prompt="",
    seed=42
):
    # Load prompts with counts
    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        # prompt_counts = [(line.rsplit("+", 1)[0], int(line.rsplit(" ", 1)[1])) for line in lines]

    # Load SD3 pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to("cuda")

    # pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    # print("check prompt and count", len(prompt_counts), prompt_counts[0])
    # exit()

    # total_images = 0
    # total_time = 0.0
    count = 8
    for i, prompt in enumerate(lines):
        subdir = output_root / f"{i:03d}_{sanitize_filename(prompt)}"
        subdir.mkdir(parents=True, exist_ok=True)

        if all((subdir / f"sample_{j:02d}.png").exists() for j in range(count)):
            print(f"[{i+1}/{len(lines)}] Skipped (already exists): {prompt}")
            continue

        print(f"[{i+1}/{len(lines)}] Generating: {prompt}")
        start = time.time()
        for j in range(count):
            image = pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=768,
                width=768,
                generator=generator,
            ).images[0]
            image.save(subdir / f"sample_{j:02d}.png")

        duration = time.time() - start
        print(f"→ {count} images in {duration:.2f}s → {duration/count:.2f}s/image")

if __name__ == "__main__":
    generate_images_sd3()