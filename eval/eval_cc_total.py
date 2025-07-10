from pathlib import Path
from PIL import Image
import torch
import clip
from prdc import compute_prdc
import numpy as np

clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
clip_model.eval()

def extract_features_all(image_paths, batch_size=8):
    features = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        imgs = [clip_preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in batch]
        tensor = torch.cat(imgs).to("cuda")
        with torch.no_grad():
            feats = clip_model.encode_image(tensor)
            features.append(feats.cpu().numpy())
    return np.concatenate(features, axis=0)

# 设置路径
ref_root = Path("/home/ec2-user/pareto_ft/datasets/cc12m_filtered_imgs_small")
gen_root = Path("/home/ec2-user/pareto_ft/results/sd3medium/cc12m")

# 聚合所有图像路径
ref_images = sorted(list(ref_root.glob("*/*.jpg")) + list(ref_root.glob("*/*.png")), key=lambda x: int(x.parent.name))
gen_images = sorted(list(gen_root.glob("*/*.jpg")) + list(gen_root.glob("*/*.png")))

print(f"Total ref: {len(ref_images)}, gen: {len(gen_images)}")

# 特征提取
real_feats = extract_features_all(ref_images)
gen_feats = extract_features_all(gen_images)

# PRDC
metrics = compute_prdc(real_feats, gen_feats, nearest_k=5)
print("Global PRDC:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")