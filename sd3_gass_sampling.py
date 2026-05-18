from diffusers import StableDiffusion3Pipeline
import torch
from pathlib import Path
from PIL import Image
import os
import time
import torch.nn.functional as F
import math
from torchvision.utils import save_image
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
from tqdm import tqdm
import json
import torch.nn.functional as F
from typing import Optional


device = "cuda"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


## manual define the clip_transform
clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


def sanitize_filename(text):
    return text.replace(" ", "_").replace(",", "").replace(".", "").lower()[:50]



def create_orthogonal_bases(
    e_text: torch.Tensor,
    dim: int = 512,
    n_bases: int = 10,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Create orthonormal bases orthogonal to e_text on the unit sphere.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if e_text.dim() == 1:
        e_text = e_text.unsqueeze(0)
    e_text_norm = F.normalize(e_text, p=2, dim=1)  # (1, D)
    
    bases = []
    for _ in range(n_bases):
        random_vec = torch.randn(1, dim, device=e_text.device)
        proj_on_text = (random_vec @ e_text_norm.T) * e_text_norm
        orthogonal_vec = random_vec - proj_on_text
        orthogonal_vec = F.normalize(orthogonal_vec, p=2, dim=1)
        bases.append(orthogonal_vec)
    bases = torch.cat(bases, dim=0)
    Q, _ = torch.linalg.qr(bases.T)
    bases = Q[:, :n_bases].T
    return bases



def build_e_target_gass(
    image_embeds: torch.Tensor, 
    text_embed: torch.Tensor, 
    w1: float = 0.0,
    w2: float = 0.0,
    r1_perturb_strength: float = 0.02,
    r2_perturb_strength: float = 0.02,
    n_candidate_bases: int = 10,
    seed: Optional[int] = 42
):
    device = image_embeds.device
    B, D = image_embeds.shape
    
    e_text = text_embed / text_embed.norm()
    
    proj_on_text = (image_embeds @ e_text.unsqueeze(1)).squeeze(1)  # [B]
    
    if seed is not None:
        torch.manual_seed(seed)

    
    perturb_text = torch.empty(B, device=device).uniform_(-r1_perturb_strength, r1_perturb_strength)
    perturbed_proj = (proj_on_text + w1 * perturb_text).clamp(-1.0, 1.0)  

    # Compute residual orthogonal components
    alignment_component_perturbed = perturbed_proj[:, None] * e_text[None, :]  # [B, D]
    residual = image_embeds - proj_on_text[:, None] * e_text[None, :]          
    
    
    bases = create_orthogonal_bases(e_text, dim=D, n_bases=n_candidate_bases, seed=seed)  # (n, D)

    # find the dominat base
    proj_scores = []
    for i in range(n_candidate_bases):
        base = bases[i].unsqueeze(0)
        proj = (residual @ base.T).squeeze(1)
        mean_abs_proj = proj.abs().mean().item()
        proj_scores.append(mean_abs_proj)
    proj_scores = torch.tensor(proj_scores, device=device)
    main_idx = proj_scores.argmax().item()
    main_base = bases[main_idx].unsqueeze(0)  

    
    perturb_tangent = torch.empty(B, device=device).uniform_(-r2_perturb_strength, r2_perturb_strength)
    delta_tangent = (w2 * perturb_tangent)[:, None] * main_base  # [B, D]

    
    e_target = alignment_component_perturbed + residual + delta_tangent

    
    e_target = e_target / e_target.norm(dim=1, keepdim=True)

    return e_target



def create_orthogonal_component(x_perp: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Create a component orthogonal to both x_perp and axis for rotation.
    In high dimensions, we use Gram-Schmidt on a random vector.
    """
    B, D = x_perp.shape
    device = x_perp.device
    
    # Generate random vector
    random_vec = torch.randn(B, D, device=device)
    
    # Orthogonalize w.r.t axis
    random_vec = random_vec - (random_vec @ axis).unsqueeze(1) * axis.unsqueeze(0)
    
    # Orthogonalize w.r.t x_perp
    x_perp_norm = x_perp / (x_perp.norm(dim=1, keepdim=True) + 1e-8)
    random_vec = random_vec - (random_vec * x_perp_norm).sum(dim=1, keepdim=True) * x_perp_norm
    
    # Normalize
    random_vec = F.normalize(random_vec, p=2, dim=1)
    
    # Scale to match x_perp magnitude
    random_vec = random_vec * x_perp.norm(dim=1, keepdim=True)
    
    return random_vec




@torch.enable_grad()
def compute_delta_x0_from_e_target(x_0, e_target, 
                                   lr=8e-3, max_steps=60, tol=5e-5, patience=4, device="cuda"):
    """
    Given a batch of pixel-space images x_0 and a target embedding e_target,
    estimate delta_x0 via gradient ascent to match e_target in CLIP space.

    Args:
        x_0: [B, 3, H, W], pixel image in [0, 1], float32 or float16
        e_target: [B, D], target CLIP embeddings (unit norm)
        clip_model: CLIPModel or similar, with .get_image_features()
        processor: CLIPProcessor or transform function for preprocessing
        lr: learning rate for gradient update
        steps: number of ascent steps

    Returns:
        x_0_updated: [B, 3, H, W], perturbed pixel image (clamped to [0, 1])
        delta_x0: [B, 3, H, W], the actual perturbation applied
    """

    x_0 = x_0.to(device=device, dtype=torch.float32).requires_grad_(True)
    x_0_start = x_0.detach().clone()
    e_target = e_target.to(device=device, dtype=torch.float32)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    
    optimizer = torch.optim.Adam([x_0], lr=lr)

    best_loss = torch.tensor(10.0, dtype=torch.float32, device="cuda")
    patience_counter = 0


    for i in range(max_steps):
        optimizer.zero_grad()

        # Apply CLIP preprocessing and encode

        x_resized = F.interpolate(x_0, size=(224, 224), mode="bicubic", align_corners=False)
        x_norm = (x_resized - mean) / std

        e_pred = clip_model.get_image_features(pixel_values=x_norm)
        e_pred = e_pred / e_pred.norm(dim=1, keepdim=True)

        # Cosine similarity loss to maximize
        cos_sim = (e_pred * e_target).sum(dim=1)  # [B]
        loss = - cos_sim.mean()  # maximize similarity → minimize negative
        # print("check loss:", i, cos_sim, loss)

        loss.backward()
        optimizer.step()
        
        # Optional clamp to valid pixel range
        with torch.no_grad():
            x_0.clamp_(0, 1)

        # Convergence check -> early stop if necessary
        if abs(best_loss - loss.item()) < tol:
            patience_counter += 1
            if patience_counter >= patience:
                break
        else:
            best_loss = loss.item()
            patience_counter = 0

    delta_x0 = (x_0.detach() - x_0_start).to(dtype=x_0_start.dtype)
    return x_0, delta_x0



@torch.no_grad()
def step_from_x0_hat_sd3(scheduler, x_0_hat, t_idx, generator, noise=None):
    """
    Approximate x_{t-1} from x_0_hat using SD3's FlowMatchEulerDiscreteScheduler.
    Assumes t_idx is an integer index into scheduler.timesteps.
    """
    if noise is None:
        noise = torch.randn_like(x_0_hat)

    # Get sigmas
    sigmas = scheduler.sigmas  # shape: [num_inference_steps]
    sigma_t = sigmas[t_idx]

    # Euler step for reverse SDE: x_{t-1} = x_0 + sigma_t * noise
    x_t_minus_1 = x_0_hat + sigma_t * noise
    return x_t_minus_1
 


def gass_backward_step(
    x_t, t, model, scheduler,
    guidance_scale=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    pooled_prompt_embeds=None,
    negative_pooled_prompt_embeds=None,
    pipe=None,
    t_idx=None,
    generator=None,
    prompt_batch=None,
):


    # 1. Predict noise: this requires manual implementation for the CFG in SD3
    with torch.no_grad():
        noise_pred_cond = model(
            hidden_states=x_t,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
        ).sample

        noise_pred_uncond = model(
            hidden_states=x_t,
            timestep=t,
            encoder_hidden_states=negative_prompt_embeds,
            pooled_projections=negative_pooled_prompt_embeds,
        ).sample

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    # 2. Predict x0 from noise: this also requires manual implementation for SD3
        ### manual computation for SD3
        sigma_t = pipe.scheduler.sigmas[t_idx]
        alpha_t = (1.0 - sigma_t**2) ** 0.5
        x_0_hat_latent = (x_t - sigma_t * noise_pred) / alpha_t

        ### Decode latent to pixel space
        x_0_hat_img = pipe.vae.decode(x_0_hat_latent / pipe.vae.config.scaling_factor).sample
        x_0_hat_img = (x_0_hat_img.clamp(-1, 1) + 1) / 2  # [B, 3, H, W], pixel values in [0,1]

    # 3. Get the text_prompt and predicted x_0 image embedding in CLIP space
        if t_idx > 0 and t_idx <= 10: # SPP intervention steps, could be spare
            inputs = clip_processor(images=list(x_0_hat_img), text=prompt_batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds # [B, 512], norm = 1
            text_embeds = outputs.text_embeds # [B, 512], with the same embedding, norm = 1 

            ##### Get the target img_embeds
            e_target = build_e_target_gass(image_embeds, text_embeds[0])
            x_0_hat_new_img, delta_x0 = compute_delta_x0_from_e_target(x_0_hat_img, e_target)

            #####
            x_0_hat_new_img = x_0_hat_new_img.clamp(0, 1)
            x_0_hat_new_img = x_0_hat_new_img * 2.0 - 1.0
            x_0_hat_new_img = x_0_hat_new_img.half()

            x_0_hat_new_latent = pipe.vae.encode(x_0_hat_new_img).latent_dist.sample() * pipe.vae.config.scaling_factor            
            ##### Construct target image_embeds
            noise_pred_new = (x_t - x_0_hat_new_latent) / sigma_t

            x_t_minus_1 = scheduler.step(noise_pred_new, t[0], x_t).prev_sample


        else:
            x_t_minus_1 = scheduler.step(noise_pred, t[0], x_t).prev_sample

    return x_t_minus_1





def generate_images_sd3(
    prompt_count_file="./drawbench_prompts.txt",
    output_dir="./results/sd3m/drawbench_gass",
    model_id="stabilityai/stable-diffusion-3-medium-diffusers",
    max_batch_size=10,
    guidance_scale=7.0,
    num_inference_steps=28,
    image_size=(512, 512),
    seed=42,
):
    device = "cuda"
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16"
    ).to(device)
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    # Output path
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load prompts
    with open(prompt_count_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]


    for i, line in enumerate(lines):

        prompt = line
        count = 10 ## total number of generated images required for each prompt 
        print("check prompt:", i, prompt, count)

        subdir = output_root / f"{i:03d}_{sanitize_filename(prompt)}"
        subdir.mkdir(parents=True, exist_ok=True)

        if all((subdir / f"sample_{j:02d}.png").exists() for j in range(count)):
            print(f"[{i+1}/{len(lines)}] Skipped: {prompt}")
            continue

        print(f"[{i+1}/{len(lines)}] Generating: {prompt} ({count} images)")

        image_idx = 0
        n_batches = math.ceil(count / max_batch_size)
        with torch.no_grad():
            for b in range(n_batches):
                bsz = min(max_batch_size, count - image_idx)
                prompt_batch = [prompt] * bsz

                # Encode prompt
                prompt_embeds, neg_embeds, pooled_embeds, pooled_neg_embeds = pipe.encode_prompt(
                    prompt=prompt_batch,
                    prompt_2=None,
                    prompt_3=None,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=["blurry, low quality, bad anatomy, deformed, extra limbs"] * len(prompt_batch),
                )

                
                model = pipe.transformer
                pipe.scheduler.set_timesteps(num_inference_steps)

                # Init latents
                H, W = image_size  # e.g., (768, 768)
                latent_h, latent_w = H // 8, W // 8 ## downsampling rate in SD3 is 8
                latents = torch.randn(
                    (bsz, pipe.transformer.config.in_channels, latent_h, latent_w),
                    device=device,
                    generator=generator,
                    dtype=torch.float16
                ) * pipe.scheduler.sigmas[0]


                ### Intervention in the generation sampling trajectory
                start = time.time()
                for idx, t in enumerate(pipe.scheduler.timesteps):
                    t_bs = torch.tensor([t] * bsz, dtype=torch.float32, device=device)
                    x_t_minus_1 = gass_backward_step(
                        x_t=latents,
                        t=t_bs,
                        model=pipe.transformer,
                        scheduler=pipe.scheduler,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_embeds,
                        negative_prompt_embeds=neg_embeds,
                        negative_pooled_prompt_embeds=pooled_neg_embeds,
                        guidance_scale=guidance_scale,
                        pipe = pipe,
                        t_idx=idx,
                        generator=generator,
                        prompt_batch=prompt_batch,
                    )

                    latents = x_t_minus_1

                # 4. decode
                images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
                images = (images.clamp(-1, 1) + 1) / 2

                duration = time.time() - start
                print(f"→ {count} images in {duration:.2f}s → {duration/count:.2f}s/image")



if __name__ == "__main__":
    generate_images_sd3()