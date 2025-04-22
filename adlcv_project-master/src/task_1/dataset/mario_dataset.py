import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from .helpers import set_seed

class MarioDataset(Dataset):
    def __init__(self, 
                 transform, 
                 img_file=None,
                 num_samples=40000,
                 seed=1
    ):
        # Resolve path to dataset file
        if img_file is None:
            project_root = Path(__file__).resolve().parents[3]  # from src/task_1/dataset/
            img_file = project_root / "data" / "processed" / "stride_64_224.npy"

        self.images = np.load(img_file)
        self.num_samples = num_samples
        self.seed = seed

        # Reduce dataset size
        if num_samples:
            set_seed(seed=self.seed)
            sampled_indeces = random.sample(range(len(self.images)), self.num_samples)
            self.images = self.images[sampled_indeces]

        print(f"Dataset shape: {self.images.shape}")
        
        self.transform = transform
        self.images_shape = self.images.shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return image
