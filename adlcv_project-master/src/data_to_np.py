import os
import numpy as np
from PIL import Image

def convert_folder_to_npy(image_folder, output_file, image_size=(224, 224)):
    data = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(image_folder, filename)
            img = Image.open(path).convert('RGB').resize(image_size)
            arr = np.array(img)
            data.append(arr)

    data = np.stack(data)
    print(f"Saved dataset shape: {data.shape}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, data)

image_folder = 'C:/Users/filip/adlcv_project-master (1)/adlcv_project-master/data/processed/stride_64'   # <-- path to your image folder
output_file = 'C:/Users/filip/adlcv_project-master (1)/adlcv_project-master/data/processed/stride_64_224.npy'  # <-- path to save the .npy file
convert_folder_to_npy(image_folder, output_file)