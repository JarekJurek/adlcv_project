import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class MarioSequentialDataset(Dataset):
    def __init__(self, npy_dir, seed=1):
        """
        Args:
            npy_dir: Directory containing per-scenario .npy files.
        """
        self.seed = seed
        self.data = []
        self.sequence_limits = []

        npy_files = sorted(Path(npy_dir).glob("*.npy"))
        cumulative_index = 0

        for npy_file in npy_files:
            scenario_data = np.load(npy_file)
            num_scenes = len(scenario_data)
            self.data.append(scenario_data)

            cumulative_index += num_scenes
            self.sequence_limits.append(cumulative_index)

        self.data = np.concatenate(self.data, axis=0)   # Shape: [Total_Scenes, 14, 14, 13]
        self.num_classes = self.data.shape[-1]

        # Build valid indices for sequential pairing
        self.valid_indices = self.get_valid_indices()

        print(f"Loaded dataset: {self.data.shape}, Valid pairs: {len(self.valid_indices)}")

    def get_valid_indices(self):
        valid = set(range(len(self.data)))
        for limit in self.sequence_limits:
            valid.discard(limit - 1)  # Remove last frame of each scenario
        return sorted(list(valid))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        current_frame = self.data[real_idx].astype(np.float32)
        next_frame = self.data[real_idx + 1].astype(np.float32)

        current_frame = torch.from_numpy(current_frame).permute(2, 0, 1)  # [13,14,14]
        next_frame = torch.from_numpy(next_frame).permute(2, 0, 1)

        return next_frame, current_frame

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "data" / "processed_text" / "stride_224_text" / "scenarios_npy"
    dataset = MarioSequentialDataset(npy_dir=dataset_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    for i, batch in enumerate(loader):
        next_frame, current_frame = batch
        print(f"Batch {i}: next_frame {next_frame.shape}, current_frame {current_frame.shape}")
        if i == 1:
            break # todo: check if this is really sequentional
