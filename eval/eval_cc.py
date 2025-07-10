import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import clip
from prdc import compute_prdc
from tqdm import tqdm
import pandas as pd

clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
clip_model.eval()

def extract_clip_features(image_paths, batch_size=8, max_images=128):
    features = []
    # image_paths = image_paths[:max_images]
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_imgs = [clip_preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in batch_paths]
        batch_tensor = torch.cat(batch_imgs).to("cuda")

        with torch.no_grad():
            feats = clip_model.encode_image(batch_tensor)
            features.append(feats.cpu().numpy())

    return np.concatenate(features, axis=0)

def evaluate_all_pairs(ref_root, gen_root, output_csv="prdc_metrics.csv", max_images=128):
    ref_root = Path(ref_root)
    gen_root = Path(gen_root)
    ref_ids = sorted([d.name for d in ref_root.iterdir() if d.is_dir()], key=int)
    gen_ids = sorted([d.name for d in gen_root.iterdir() if d.is_dir()])
    # print("check ref_ids:", len(ref_ids), ref_ids[0], ref_ids[1])
    # print("check gen_ids:", len(gen_ids), gen_ids[0], gen_ids[1])
    results = []
    total_precision = 0
    total_reall = 0
    total_density = 0
    total_coverage = 0
    count = 0


    for c, pid in enumerate(ref_ids):
        ref_dir = ref_root / ref_ids[c]
        gen_dir = gen_root / gen_ids[c]
        # ref_dir = ref_ids[c]
        # gen_dir = gen_ids[c]

        # if not gen_dirs:
        #     print(f"[Warning] No gen folder matched {pid}")
        #     continue
        # gen_dir = gen_dirs[0]
        ref_images = sorted([p for p in ref_dir.glob("*.png")] + [p for p in ref_dir.glob("*.jpg")])
        gen_images = sorted([p for p in gen_dir.glob("*.png")] + [p for p in gen_dir.glob("*.jpg")])

        if len(ref_images) < 5 or len(gen_images) < 5:
            print(f"[Skip] {pid} has too few images")
            continue

        # try:
        if len(ref_images) != len(gen_images):
            print("check path:", ref_ids[c], gen_ids[c])
            print("check ref_images and gen_images:", ref_images, gen_images)
            exit()

        real_feats = extract_clip_features(ref_images, max_images=max_images)
        gen_feats = extract_clip_features(gen_images, max_images=max_images)
        metrics = compute_prdc(real_feats, gen_feats, nearest_k=5)
        precision = metrics['precision']
        recall = metrics['recall']
        density = metrics['density']
        coverage = metrics['coverage']
        count += 1
        # except Exception as e:
        #     print(f"[Error] Skipping {pid} due to error: {e}")
        #     continue

        results.append({
            "prompt_id": pid,
            "num_ref": len(ref_images),
            "num_gen": len(gen_images),
            **metrics
        })

        total_precision += precision
        total_reall += recall
        total_density += density
        total_coverage += coverage
        print("scores at", c+1, precision, recall, density, coverage)
        print("two folders:", ref_ids[c], gen_ids[c])

    print("check scores:", total_precision/count, total_reall/count, total_density/count, total_coverage/count)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved PRDC results to {output_csv}")
    return df

# Example usage
if __name__ == "__main__":
    evaluate_all_pairs(
        ref_root="/home/ec2-user/pareto_ft/datasets/cc12m_filtered_imgs_small",
        gen_root="/home/ec2-user/pareto_ft/results/sdturbo/cc12m",
        output_csv="prdc_metrics_sdturbo.csv",
        # max_images=32
    )