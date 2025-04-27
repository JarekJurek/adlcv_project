import numpy as np
import os
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import optim
import logging
from ddpm import Diffusion
from model import UNet
from pathlib import Path

SEED = 1
DATASET_SIZE = None

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_onehot_batch(images, out_dir, epoch):
    out_dir = os.path.join(out_dir, f"epoch_{epoch}", "onehot")
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        onehot_np = img.cpu().detach().numpy()  # [C, H, W]
        onehot_np = np.transpose(onehot_np, (1, 2, 0))  # [H, W, C]
        out_path = os.path.join(out_dir, f"sample_{i:02d}.npy")
        np.save(out_path, onehot_np)

def prepare_dataloader(batch_size):
    from torch.utils.data import DataLoader
    from dataset import MarioSequentialDataset

    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / "processed_text" / "stride_224_text" / "scenarios_npy"
    dataset = MarioSequentialDataset(npy_dir=dataset_dir, seed=SEED)
    print(f"Loaded dataset with shape: {dataset.data.shape}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def setup_logger(log_dir):
    logger = logging.getLogger(log_dir.name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')

    file_handler = logging.FileHandler(log_dir / 'training.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def train(device='cpu', T=500, img_size=14, input_channels=26, channels=32, time_dim=256,  # 26 = 13 (x_t) + 13 (current_frame)
          batch_size=32, lr=1e-3, num_epochs=2500, experiment_name="run_2"):
    
    output_dir = Path("runs_task4")
    models_dir = output_dir / experiment_name / "models"
    log_dir = output_dir / experiment_name
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = setup_logger(log_dir)
    dataloader = prepare_dataloader(batch_size)

    model = UNet(img_size=img_size, c_in=input_channels, c_out=13,  # c_out = 13 - predict noise for next_frame
                 time_dim=time_dim, channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(dataloader)

        for next_frame, current_frame in pbar:
            next_frame = next_frame.to(device)
            current_frame = current_frame.to(device)

            t = diffusion.sample_timesteps(next_frame.shape[0]).to(device)

            # Add noise ONLY to next_frame
            x_t, noise = diffusion.q_sample(next_frame, t)

            # Final model input
            model_input = torch.cat([x_t, current_frame], dim=1)  # Shape: [batch, 26, 14, 14]

            predicted_noise = model(model_input, t)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

        logger.info(f"Epoch {epoch}, MSE Loss: {loss.item():.6f}")

        # save model weights every 100 epochs and at the end
        if epoch % 100 == 0 or epoch == num_epochs:
            save_dir = models_dir / f"weights-{epoch}.pt"
            torch.save(model.state_dict(), save_dir)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed(seed=SEED)
    train(device=device)

if __name__ == '__main__':
    main()
