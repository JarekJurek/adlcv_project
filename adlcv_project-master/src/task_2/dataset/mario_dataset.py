import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class MarioDataset(Dataset):
    def __init__(self, 
                 npy_path=None,
                 num_samples=None,
                 seed=1
    ):
        # Resolve path
        if npy_path is None:
            project_root = Path(__file__).resolve().parents[3]
            npy_path = project_root / "data" / "processed_text" / "stride_32_onehot.npy"

        self.data = np.load(npy_path)
        self.seed = seed

        if num_samples is None:
            self.num_samples = len(self.data)
        else:
            self.num_samples = min(num_samples, len(self.data))

        # Reduce dataset size if needed
        if self.num_samples < len(self.data):
            set_seed(seed=self.seed)
            indices = random.sample(range(len(self.data)), self.num_samples)
            self.data = self.data[indices]

        print(f"Dataset shape: {self.data.shape}")  # [N, 14, 14, 13]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [13, 14, 14]
        return x


# -------------------------- #
#           TEST            #
# -------------------------- #
if __name__ == "__main__":
    dataset = MarioDataset(num_samples=10)
    loader = DataLoader(dataset, batch_size=2)

    for i, batch in enumerate(loader):
        print(f"Batch {i} shape: {batch.shape}")  # Should be [2, 13, 14, 14]
        if i == 1:
            break
