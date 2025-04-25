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

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")

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

def create_result_folders(experiment_name):
    os.makedirs(os.path.join("/work3/s242911/models_task4", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("/work3/s242911/results_task4", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("/work3/s242911/runs_task4", experiment_name), exist_ok=True)

def train(device='cpu', T=500, img_size=14, input_channels=30, channels=32, time_dim=256,  # 30 = 13 (x_t) + 13 (current_frame) + 4 (action_map)
          batch_size=32, lr=1e-3, num_epochs=1000, experiment_name="contitional"):
    
    create_result_folders(experiment_name)
    dataloader = prepare_dataloader(batch_size)

    model = UNet(img_size=img_size, c_in=input_channels, c_out=13,  # c_out = 13 - predict noise for next_frame
                 time_dim=time_dim, channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs_task4", experiment_name))
    l = len(dataloader)

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)

        for i, (next_frame, current_frame, action_map) in enumerate(pbar):
            next_frame = next_frame.to(device)
            current_frame = current_frame.to(device)
            action_map = action_map.to(device)

            t = diffusion.sample_timesteps(next_frame.shape[0]).to(device)

            # Add noise ONLY to next_frame
            x_t, noise = diffusion.q_sample(next_frame, t)

            # Prepare condition: current_frame + action_map
            condition = torch.cat([current_frame, action_map], dim=1)   # Shape: [batch, 17, 14, 14]

            # Final model input
            model_input = torch.cat([x_t, condition], dim=1)            # Shape: [batch, 30, 14, 14]

            predicted_noise = model(model_input, t)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # Sample at every epoch and save as .npy
        # samples = diffusion.p_sample_loop(model, batch_size=sampling_batch_size)
        # save_onehot_batch(samples,
        #                   out_dir=os.path.join("results_task2_txt", experiment_name),
        #                   epoch=epoch)

        # Final epoch → generate 10 samples
        if epoch == num_epochs:
            logging.info(f"Sampling 10 final images...")
            final_samples = diffusion.p_sample_loop(model, batch_size=10)
            save_onehot_batch(final_samples,
                              out_dir=os.path.join("results_task4", experiment_name),
                              epoch="final")

        # Save model weights
        torch.save(model.state_dict(),
                   os.path.join("/work3/s242911/models_task4", experiment_name, f"weights-{epoch}.pt"))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed(seed=SEED)
    train(device=device)

if __name__ == '__main__':
    main()
