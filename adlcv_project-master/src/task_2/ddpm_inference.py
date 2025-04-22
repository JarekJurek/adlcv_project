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

from ddpm import Diffusion
from ddpm_train import set_seed
from model import UNet
from ddpm_train import prepare_dataloader

# -------------------- PATH SETUP --------------------
project_root = Path(__file__).resolve().parents[2]

vgg_weights_path = project_root / "src" / "task_1" / "models" / "vgg" / "vgg.pth"
ddpm_weights_path = project_root / "src" / "task_2" / "models" / "ddpm" / "weights-999.pt"
results_root = project_root / "results"
# -----------------------------------------------------

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT).features[:10]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x)           # (B, 256, H, W)
        x = self.avg_pool(x)           # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)      # (B, 256)
        return x

def get_features(model, images):
    print(">> get_features input shape:", images.shape)  # Add here

    model.eval()
    with torch.no_grad():
        features = model(images)

    print(">> get_features output shape (before numpy):", features.shape)  # Add here
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
DATASET_SIZE = None

def inference(device='cpu', T=500, img_size=224, num="1"):
    # --- Create result folders ---
    result_dir = results_root / f"inference_{num}"
    real_dir = result_dir / "real_images"
    gen_dir = result_dir / "generated_images"
    result_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    # --- Load models ---
    unet = UNet(img_size=img_size, c_in=3, c_out=3, time_dim=256, channels=32, device=device).to(device)
    unet.load_state_dict(torch.load(ddpm_weights_path, map_location=device))

    ddpm = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    vgg_model = VGG().to(device)
    vgg_model.load_state_dict(torch.load(vgg_weights_path, map_location=device, weights_only=True), strict=False)
    dims = 256
    
    test_loader = prepare_dataloader(batch_size=1)

    original_feat = np.empty((len(test_loader.dataset), dims))
    generated_feat = np.empty((len(test_loader.dataset), dims))
    start_idx = 0

    for images in tqdm(test_loader):
        images = images.to(device)
        
        print("\n--- ORIGINAL IMAGE DEBUG ---")
        print("Original image batch shape (before get_features):", images.shape)

        original = get_features(vgg_model, images)

        y = torch.randint(0, 5, (images.shape[0],), device=device)
        ddpm_images = ddpm.p_sample_loop(unet, batch_size=images.shape[0])
        
        images = (ddpm_images / 255.0 - 0.5) / 0.5  

        print("\n--- GENERATED IMAGE DEBUG ---")
        print("Generated image batch shape (before get_features):", images.shape)

        features = get_features(vgg_model, images)

        print("Storing original features of shape:", original.shape)

        original_feat[start_idx:start_idx + original.shape[0]] = original

        print("Storing generated features of shape:", features.shape)

        generated_feat[start_idx:start_idx + original.shape[0]] = features
        start_idx += original.shape[0]

    mu_original, sigma_original = feature_statistics(original_feat)
    mu, sigma = feature_statistics(generated_feat)

    fid = frechet_distance(mu_original, sigma_original, mu, sigma)
    print(f'[FID score:] {fid:.3f}')

    # Save FID to file
    with open(result_dir / "fid_score.txt", "w") as f:
        f.write(f"FID score: {fid:.3f}\n")

    # Save 5 image pairs
    num_samples = 5
    for i in range(num_samples):
        real_img = images[i].cpu().permute(1, 2, 0).numpy()
        ddpm_img = ddpm_images[i].cpu().permute(1, 2, 0).numpy()

        real_img = (real_img - real_img.min()) / (real_img.max() - real_img.min())
        ddpm_img = (ddpm_img - ddpm_img.min()) / (ddpm_img.max() - ddpm_img.min())

        def save_with_label(image, path):
            plt.figure(figsize=(2, 2))
            plt.imshow(image)
            plt.axis("off")
            plt.savefig(path, bbox_inches="tight")
            plt.close()

        save_with_label(real_img, real_dir / f"real_{i}.png")
        save_with_label(ddpm_img, gen_dir / f"generated_{i}.png")

    print(f"Saved FID score and image samples in: {result_dir}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed(seed=SEED)
    inference(device=device, num="1")

if __name__ == '__main__':
    main()
