import torch
import numpy as np
from ddpm import Diffusion
from model import UNet
from pathlib import Path


def load_initial_frame(npy_path, device):
    frame = np.load(npy_path)   # Shape: [14,14,13]
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,13,14,14]
    return frame


def generate_sequence(model, diffusion, initial_frame, action_map, steps=5):
    current_frame = initial_frame.clone()
    generated_frames = []

    for i in range(steps):
        condition = torch.cat([current_frame, action_map], dim=1)
        x_t = torch.randn((1, 13, 14, 14)).to(current_frame.device)

        for t in reversed(range(1, diffusion.T)):
            t_batch = torch.ones(1).long().to(current_frame.device) * t
            x_t = diffusion.p_sample(model, x_t, condition, t_batch)

        next_frame = x_t.squeeze(0).detach().cpu().numpy()  # [13, 14, 14]
        next_frame = np.transpose(next_frame, (1, 2, 0))  # [14, 14, 13]
        generated_frames.append(next_frame)

    return generated_frames


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load model ---
    model = UNet(img_size=14, c_in=30, c_out=13, time_dim=256,
                 channels=32, device=device).to(device)
    model.load_state_dict(torch.load("/zhome/a2/c/213547/work3/models_task4/contitional/weights-950.pt"))
    model.eval()

    diffusion = Diffusion(T=500, img_size=14, device=device)

    # --- Load initial frame ---
    initial_frame_path = Path("/zhome/a2/c/213547/DLCV/adlcv_project/adlcv_project-master/data/processed_text/stride_224_text/testing/mario-1-1_13.npy")
    initial_frame = load_initial_frame(initial_frame_path, device)

    # --- Prepare action map (hardcoded "right") ---
    action = torch.tensor([0, 1, 0, 0], dtype=torch.float32).to(device)
    action_map = action.view(1, 4, 1, 1).repeat(1, 1, 14, 14)  # [1,4,14,14]

    # --- Generate sequence ---
    generated_frames = generate_sequence(model, diffusion, initial_frame, action_map, steps=7)

    # --- Save generated frames ---
    output_dir = Path("generated_sequences")
    output_dir.mkdir(exist_ok=True)

    for idx, frame in enumerate(generated_frames):
        np.save(output_dir / f"generated_frame_{idx+1}.npy", frame)

    print(f"Generated {len(generated_frames)} frames and saved to {output_dir}")


if __name__ == "__main__":
    main()
