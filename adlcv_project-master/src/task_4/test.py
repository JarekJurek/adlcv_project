import torch
import numpy as np
from ddpm import Diffusion
from model import UNet
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.decoder.onehot_to_ascii import decode_onehot_to_ascii, load_idx_to_char
from src.decoder.mario_decoder import render_level


def main():
    print("Running inference...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load model ---
    model = UNet(img_size=14, c_in=26, c_out=13, time_dim=256, channels=32, device=device).to(device)
    model.load_state_dict(torch.load("runs_task4/models/run_1/weights-1200.pt"))
    model.eval()

    diffusion = Diffusion(img_size=14, T=500, beta_start=1e-4, beta_end=0.02, device=device)

    # --- Load initial frame ---
    initial_frame_path = Path("/zhome/a2/c/213547/DLCV/adlcv_project/adlcv_project-master/data/processed_text/stride_224_text/testing/mario-1-1_13.npy")
    frame = np.load(initial_frame_path)  # [14,14,13]
    initial_frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)  # [1,13,14,14]

    # --- Generate sequence ---
    generated_frames = diffusion.p_sample_loop_seq(model, initial_frame, steps=7)

    output_dir = Path("generated_sequences")
    output_dir.mkdir(exist_ok=True)

    for idx, frame in enumerate(generated_frames):
        frame = np.transpose(frame, (1, 2, 0))  # [14, 14, 13]

        ascii_level = decode_onehot_to_ascii(frame)
        decoded_level = render_level(ascii_level)

        output_path = output_dir / f"generated_frame_{idx+1}.png"
        decoded_level.save(output_path)

    print(f"Generated {len(generated_frames)} frames and saved to {output_dir}")


if __name__ == "__main__":
    main()
