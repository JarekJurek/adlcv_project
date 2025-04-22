import numpy as np
import os
import random
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import optim
import logging
from ddpm import Diffusion
from model2 import UNet2

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")

SEED = 1
DATASET_SIZE = None

def set_seed2(seed=SEED):
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

def prepare_dataloader2(batch_size):
    from torch.utils.data import DataLoader
    from dataset2.mario_dataset2 import MarioDataset
    dataset = MarioDataset(num_samples=DATASET_SIZE, seed=SEED)
    print(f"Loaded dataset with shape: {dataset.data.shape}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
def create_result_folders(experiment_name):
    os.makedirs(os.path.join("models_task2_txt", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("results_task2_txt", experiment_name), exist_ok=True)
    os.makedirs(os.path.join("runs_task2_txt", experiment_name), exist_ok=True)

def train(device='cpu', T=500, img_size=14, input_channels=13, channels=32, time_dim=256,
          batch_size=32, lr=1e-3, num_epochs=1000, experiment_name="ddpm"):
    
    #sampling_batch_size = 0

    create_result_folders(experiment_name)
    dataloader = prepare_dataloader2(batch_size)

    model = UNet2(img_size=img_size, c_in=input_channels, c_out=input_channels,
                 time_dim=time_dim, channels=channels, device=device).to(device)
    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs_task2_txt", experiment_name))
    l = len(dataloader)

    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)

        for i, images in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.q_sample(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # Sample at every epoch and save as .npy
        #samples = diffusion.p_sample_loop(model, batch_size=sampling_batch_size)
        #save_onehot_batch(samples,
                          #out_dir=os.path.join("results_task2_txt", experiment_name),
                          #epoch=epoch)

        # Final epoch → generate 10 samples
        if epoch == num_epochs:
            logging.info(f"Sampling 10 final images...")
            final_samples = diffusion.p_sample_loop(model, batch_size=10)
            save_onehot_batch(final_samples,
                              out_dir=os.path.join("results_task2_txt", experiment_name),
                              epoch="final")

        # Save model weights
        torch.save(model.state_dict(),
                   os.path.join("models_task2_txt", experiment_name, f"weights-{epoch}.pt"))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model will run on {device}")
    set_seed2(seed=SEED)
    train(device=device)

if __name__ == '__main__':
    main()
