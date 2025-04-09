import os
import torch
import logging

from ddpm import Diffusion
from ddpm_train import save_images, set_seed
from model import UNet

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

SEED = 1
DATASET_SIZE = 580


def create_result_folders(experiment_name):
    os.makedirs("inference", exist_ok=True)
    os.makedirs(os.path.join("results", experiment_name), exist_ok=True)

def inference(device='cpu', T=500, model_path='models/ddpm/weights-1.pt', img_size=224, num="111", samples_number=1):
    create_result_folders(f"inference_{num}")

    model = UNet(img_size=img_size, c_in=3, c_out=3, time_dim=256, channels=32, device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    diffusion = Diffusion(img_size=img_size, T=T, beta_start=1e-4, beta_end=0.02, device=device)

    for img_num in range(0, samples_number):
        sampled_images = diffusion.p_sample_loop(model, batch_size=1)
        save_images(images=sampled_images, path=os.path.join("results", f"inference_{num}", f"{num}_{img_num+1}.jpg"), show=False)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")

    model_path = '/zhome/a2/c/213547/DLCV/adlcv_project/src/task_1/models/ddpm/weights-396.pt'

    set_seed(seed=SEED)
    inference(device=device, model_path=model_path, num=1, samples_number=3)

if __name__ == '__main__':
    main()
