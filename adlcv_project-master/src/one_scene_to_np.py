import json
import numpy as np
from pathlib import Path

def load_tile_mapping(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    chars = sorted(data['tiles'].keys())  # consistent order
    char_to_idx = {char: i for i, char in enumerate(chars)}
    return char_to_idx, len(chars)

def one_hot_encode_tile_grid(tile_grid, char_to_idx, num_classes):
    h, w = tile_grid.shape
    one_hot = np.zeros((h, w, num_classes), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            char = tile_grid[y, x]
            if char in char_to_idx:
                idx = char_to_idx[char]
                one_hot[y, x, idx] = 1.0
            else:
                print(f"[Warning] Unknown character '{char}' at ({y}, {x}) — skipped.")
    return one_hot

def process_single_scene(txt_file_path, json_mapping_path, output_npy_path):
    txt_file = Path(txt_file_path)
    char_to_idx, num_classes = load_tile_mapping(json_mapping_path)

    with open(txt_file, 'r') as f:
        lines = f.readlines()
    grid = np.array([list(line.strip()) for line in lines if line.strip()])

    if grid.shape != (14, 14):
        print(f"[Error] {txt_file.name} has incorrect shape {grid.shape}")
        return

    encoded = one_hot_encode_tile_grid(grid, char_to_idx, num_classes)

    np.save(output_npy_path, encoded)
    print(f"Saved one-hot encoded scene to {output_npy_path}. Shape: {encoded.shape}")

if __name__ == "__main__":
    txt_file_path = "/zhome/a2/c/213547/DLCV/adlcv_project/adlcv_project-master/data/processed_text/stride_224_text/testing/mario-1-1_13.txt"     # Path to your single 14x14 txt scene
    json_mapping_path = "task_2/smb.json"
    output_npy_path = "/zhome/a2/c/213547/DLCV/adlcv_project/adlcv_project-master/data/processed_text/stride_224_text/testing/mario-1-1_13.npy"

    process_single_scene(txt_file_path, json_mapping_path, output_npy_path)
