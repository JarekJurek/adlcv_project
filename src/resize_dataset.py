import numpy as np
from PIL import Image
import os

# === Config ===
input_path = '/zhome/a2/c/213547/DLCV/adlcv_project/src/task_1/data/custom_208_dataset.npy'  # Change this to your input file
output_path = '/zhome/a2/c/213547/DLCV/adlcv_project/src/task_1/data/128_dataset.npy'  # Path to save the resized output
target_size = (128, 128)  # New image size

# === Load original dataset ===
print("Loading dataset...")
data = np.load(input_path)  # shape: (929, 208, 208, 3)
print(f"Original shape: {data.shape}")

# === Resize images ===
resized_data = np.zeros((data.shape[0], *target_size, 3), dtype=np.uint8)

print("Resizing images...")
for i in range(data.shape[0]):
    img = Image.fromarray(data[i])
    img_resized = img.resize(target_size, Image.BICUBIC)  # You can use Image.LANCZOS too
    resized_data[i] = np.array(img_resized)

print(f"Saving resized dataset to {output_path}...")
np.save(output_path, resized_data)
print("Done! Resized shape:", resized_data.shape)
