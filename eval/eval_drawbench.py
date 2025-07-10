import os
from pathlib import Path
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision.models import inception_v3
from tqdm import tqdm
from vendi_score import image_utils, vendi, data_utils
import clip
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
import ImageReward as RM
# from transformers import AutoProcessor, AutoModel
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForImageClassification


device = "cuda" if torch.cuda.is_available() else "cpu"
sscd_model = torch.jit.load("sscd_disc_mixup.torchscript.pt")#.to(device).eval()
reward_model = RM.load("ImageReward-v1.0")
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
pick_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pick_model = CLIPModel.from_pretrained("yuvalkirstain/PickScore_v1").to("cuda").eval()
with open("/home/ec2-user/pareto_ft/datasets/cc12m_ordered_prompts.txt", "r") as f:
    prompts = [line.strip() for line in f if line.strip()]



### some pre-processing required for calculating the sscd-based MSS score
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)

small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])


def load_images_resized(folder, size=(256, 256)):
    images = []
    for file in sorted(Path(folder).glob("*.png")):
        img = Image.open(file).convert("RGB").resize(size, Image.LANCZOS)
        # images.append(np.array(img))
        images.append(img)
    return images

def load_images_path(folder):
    paths = []
    for file in sorted(Path(folder).glob("*.png")):
        paths.append(str(file))
    return paths


def compute_mean_msssim(images):
    n = len(images)
    scores = []
    for i in range(n):
        for j in range(i+1, n):
            sim, _ = ssim(images[i], images[j], channel_axis=2, full=True)
            scores.append(sim)
    return np.mean(scores)


def compute_mss_k(images):
    n = len(images)
    features = []
    for i in range(n):
        batch = small_288(images[i]).unsqueeze(0)
        embedding = sscd_model(batch)[0, :]
        features.append(embedding.detach().numpy())
    features = np.array(features)
    sim_matrix = cosine_similarity(features)
    return sim_matrix

def compute_pickscore(images, prompt, model, processor):
    scores = []

    for idx, img in enumerate(images):
        inputs = processor(text=prompt, images=img, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            # image_features = model.get_image_features(**inputs)
            # text_features = model.get_text_features(**inputs)
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            score = model.logit_scale.exp() * (text_features @ image_features.T)
        scores.append(score.item())
    return scores


def evaluate_all(folder_root, image_resize=(256, 256), save_csv="diversity_metrics.csv"):
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")

    results = []
    total_ssim = 0
    total_vendi_pix = 0
    total_vendi_em = 0
    total_vendi_k = 0
    total_sscd_mss = 0
    total_reward = 0
    total_pickscore = 0
    count = 0
    max_img = 36

    all_dirs = [f for f in sorted(Path(folder_root).glob("*")) if f.is_dir()]
    for subdir in tqdm(all_dirs):
        images = load_images_resized(subdir, size=image_resize)
        image_paths = load_images_path(subdir)
        prompt = prompts[count]

        if len(images) < 2:
            continue

        if len(images) > max_img:
            images = images[:max_img]
            image_paths = image_paths[:max_img]
            # print("check len:", len(images))

        np_images = [np.array(img) for img in images]
        ssim = compute_mean_msssim(np_images)
        sim_matrix = compute_mss_k(images)
        vendi_pix = image_utils.pixel_vendi_score(images)
        vendi_em = image_utils.embedding_vendi_score(images, device="cuda")
        vendi_k = vendi.score_K(sim_matrix)
        sscd_mss = np.mean(sim_matrix)
        rewards = np.mean(reward_model.score(prompt, image_paths))
        pick_score = np.mean(compute_pickscore(images, prompt,  pick_model, pick_processor))
        # print("check pick_score:", pick_score)
        # exit()


        results.append({
            "prompt_id": subdir.name,
            "prompts:": prompt,
            "num_images": len(images),
            "mean_ssim": ssim,
            "vendi_pixel_score": vendi_pix,
            "vendi_emb_score": vendi_em,
            "vendi_emb_k": vendi_k,
            "mss_sscd": sscd_mss,
            "image_reward": rewards,
            "pick_score": pick_score
        })

        total_ssim += ssim
        total_vendi_pix += vendi_pix
        total_vendi_em += vendi_em
        total_vendi_k += vendi_k
        total_sscd_mss += sscd_mss
        total_reward += rewards
        total_pickscore += pick_score
        count += 1

        print("check results:", count, total_ssim/count, total_vendi_pix/count, total_vendi_em/count, total_vendi_k/count, total_sscd_mss/count, total_reward/count, total_pickscore/count)

    df = pd.DataFrame(results)
    df.to_csv(save_csv, index=False)
    print(f"\n Saved results to {save_csv}")

if __name__ == "__main__":
    # Example usage
    evaluate_all("/home/ec2-user/pareto_ft/results/sd3medium/cc12m/", save_csv="sd3_cc12m_new_metrics.csv")

