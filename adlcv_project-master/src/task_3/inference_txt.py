import os
import torch
import numpy as np
from pathlib import Path
from tqdm import trange

from ddpm import Diffusion
from model2 import UNet2
from ddpm_train2 import prepare_dataloader2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Paths
project_root = Path(__file__).resolve().parents[2]
output_dir = project_root / "src" / "task_3" / "generated_txt"
weights_path = project_root / "src" / "task_3" / "models" / "ddpm2" / "weights-999.pt"

output_dir.mkdir(parents=True, exist_ok=True)

# Load model
model = UNet2(img_size=14, c_in=13, c_out=13, time_dim=256, device=device, channels=32).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# Setup diffusion
ddpm = Diffusion(img_size=14, T=500, beta_start=1e-4, beta_end=0.02, device=device)

# Determine how many samples to generate
real_loader = prepare_dataloader2(batch_size=32)
num_samples = 785
print(f"Generating {num_samples} samples...")

# Generate and save as .npy files (in [H, W, C] format)
batch_size = 32
with torch.no_grad():
    sample_idx = 0
    for _ in trange(0, num_samples, batch_size, desc="Generating samples"):
        curr_batch_size = min(batch_size, num_samples - sample_idx)
        samples = ddpm.p_sample_loop(model, batch_size=curr_batch_size)  # [B, 13, 14, 14]
        samples_np = samples.cpu().numpy()  # [B, 13, 14, 14]

        for j in range(curr_batch_size):
            sample_np = np.transpose(samples_np[j], (1, 2, 0))  # [14, 14, 13]
            file_path = output_dir / f"sample_{sample_idx:03}.npy"
            np.save(file_path, sample_np)
            sample_idx += 1

