import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

from ddpm import Diffusion
from ddpm_train1 import set_seed1
from model1 import UNet1
from ddpm_train1 import prepare_dataloader1

# -------------------- PATH SETUP --------------------
project_root = Path(__file__).resolve().parents[2]

vgg_weights_path = project_root / "src" / "task_3" / "models" / "vgg" / "vgg.pth"
ddpm_weights_path_1 = project_root / "src" / "task_3" / "models" / "ddpm1" / "weights-396.pt"
results_root = project_root / "results"
png_generated_dir_model2 = project_root / "src" / "task_3" / "rendered_levels_batch"
real_images_dir_model2 = project_root / "src" / "task_3" / "real_images"

# -----------------------------------------------------

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features[:10]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

def get_features(model, images):
    model.eval()
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()

def feature_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = np.sum((mu1 - mu2) ** 2)
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
SEED = 1

def inference(device='cpu', T=500, img_size=224):
    ddpm = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)
    vgg_model = VGG().to(device)
    vgg_model.load_state_dict(torch.load(vgg_weights_path, map_location=device, weights_only=True), strict=False)

    # ------- Model 1 -------
    result_dir_1 = results_root / "inference_1"
    real_dir_1 = result_dir_1 / "real_images"
    gen_dir_1 = result_dir_1 / "generated_images"
    result_dir_1.mkdir(parents=True, exist_ok=True)
    real_dir_1.mkdir(parents=True, exist_ok=True)
    gen_dir_1.mkdir(parents=True, exist_ok=True)

    unet_img = UNet1(img_size=img_size, c_in=3, c_out=3, time_dim=256, channels=32, device=device).to(device)
    unet_img.load_state_dict(torch.load(ddpm_weights_path_1, map_location=device))
    test_loader1 = prepare_dataloader1(batch_size=32)
    dims = 256
    dataset_len1 = len(test_loader1.dataset)
    original_feat1 = np.empty((dataset_len1, dims))
    generated_feat1 = np.empty((dataset_len1, dims))
    real_images_list = []
    generated_images_list = []
    start_idx = 0

    for i, batch in enumerate(tqdm(test_loader1, desc="Model 1")):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device)
        original = get_features(vgg_model, images)
        ddpm_images = ddpm.p_sample_loop(unet_img, batch_size=images.shape[0])
        norm_images = (ddpm_images / 255.0 - 0.5) / 0.5
        generated = get_features(vgg_model, norm_images)

        original_feat1[start_idx:start_idx + original.shape[0]] = original
        generated_feat1[start_idx:start_idx + original.shape[0]] = generated
        start_idx += original.shape[0]

        if i < 5:
            real_images_list.append(images[0].detach().cpu())
            generated_images_list.append(ddpm_images[0].detach().cpu())

    mu_original, sigma_original = feature_statistics(original_feat1)
    mu, sigma = feature_statistics(generated_feat1)
    fid = frechet_distance(mu_original, sigma_original, mu, sigma)
    print(f'[Model 1 - FID score:] {fid:.3f}')
    with open(result_dir_1 / "fid_score.txt", "w") as f:
        f.write(f"FID score: {fid:.3f}\n")

    for i in range(len(real_images_list)):
        real_img = real_images_list[i].permute(1, 2, 0).numpy()
        gen_img = generated_images_list[i].permute(1, 2, 0).numpy()
        real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min())
        gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())

        def save_with_label(image, path):
            plt.figure(figsize=(2, 2))
            plt.imshow(image)
            plt.axis("off")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        save_with_label(real_img, real_dir_1 / f"real_{i}.png")
        save_with_label(gen_img, gen_dir_1 / f"generated_{i}.png")

    print(f"Saved FID score and {len(real_images_list)} image pairs for Model 1 in: {result_dir_1}")

     # ------- Model 2 (Pre-Generated + Real Images) -------
    result_dir_2 = results_root / "inference_2"
    real_dir_2 = result_dir_2 / "real_images"
    gen_dir_2 = result_dir_2 / "generated_images"
    result_dir_2.mkdir(parents=True, exist_ok=True)
    real_dir_2.mkdir(parents=True, exist_ok=True)
    gen_dir_2.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    print("[Model 2] Loading images...")
    gen_images = sorted(list(png_generated_dir_model2.glob("*.png")))
    real_images = sorted(list(real_images_dir_model2.glob("*.png")))
    assert len(gen_images) == len(real_images), "Mismatch between number of real and generated images"

    all_gen = [transform(read_image(str(p)).float() / 255.0) for p in tqdm(gen_images, desc="Loading generated")]
    all_real = [transform(read_image(str(p)).float() / 255.0) for p in tqdm(real_images, desc="Loading real")]

    batch_size = 32
    gen_feat2, real_feat2 = [], []

    print("[Model 2] Extracting features in batches...")
    for i in tqdm(range(0, len(all_gen), batch_size), desc="Batches"):
        batch_gen = torch.stack(all_gen[i:i+batch_size]).to(device)
        batch_real = torch.stack(all_real[i:i+batch_size]).to(device)
        gen_feat2.append(get_features(vgg_model, batch_gen))
        real_feat2.append(get_features(vgg_model, batch_real))

    gen_feat2 = np.concatenate(gen_feat2, axis=0)
    real_feat2 = np.concatenate(real_feat2, axis=0)

    mu_real, sigma_real = feature_statistics(real_feat2)
    mu_gen, sigma_gen = feature_statistics(gen_feat2)
    fid2 = frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    print(f'[Model 2 - FID score:] {fid2:.3f}')
    with open(result_dir_2 / "fid_score.txt", "w") as f:
        f.write(f"FID score: {fid2:.3f}\n")

    print("[Model 2] Saving 10 sample image pairs...")
    for i in range(min(10, len(all_gen))):
        real_img = torchvision.transforms.functional.to_pil_image((all_real[i] * 0.5 + 0.5).cpu().clamp(0, 1))
        gen_img = torchvision.transforms.functional.to_pil_image((all_gen[i] * 0.5 + 0.5).cpu().clamp(0, 1))
        real_img.save(real_dir_2 / f"real_{i}.png")
        gen_img.save(gen_dir_2 / f"generated_{i}.png")

    print(f"Saved FID score and 10 image pairs for Model 2 in: {result_dir_2}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed1(seed=SEED)
    inference(device=device)

if __name__ == '__main__':
    main()
