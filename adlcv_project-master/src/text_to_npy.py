import os
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

def save_onehot_as_txt(onehot_array, output_txt_path):
    """
    Save a [14,14,num_classes] one-hot array as a .txt file where each line
    is a space-separated one-hot vector per tile (flattened across classes).
    """
    h, w, c = onehot_array.shape
    with open(output_txt_path, 'w') as f:
        for y in range(h):
            for x in range(w):
                vec = onehot_array[y, x]
                line = ' '.join(str(int(v)) for v in vec)
                f.write(line + '\n')

def process_txt_patches(txt_dir, json_mapping_path, output_path):
    txt_dir = Path(txt_dir)
    char_to_idx, num_classes = load_tile_mapping(json_mapping_path)

    all_encoded = []

    # New folder for individual one-hot txt slices
    onehot_txt_dir = txt_dir.parent / (txt_dir.name + "_onehot_txt")
    onehot_txt_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(txt_dir.glob("*.txt")):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        grid = np.array([list(line.strip()) for line in lines if line.strip()])
        if grid.shape != (14, 14):
            print(f"[Skip] {txt_file.name} has incorrect shape {grid.shape}")
            continue

        encoded = one_hot_encode_tile_grid(grid, char_to_idx, num_classes)

        # Save to separate one-hot folder
        onehot_txt_output_path = onehot_txt_dir / (txt_file.stem + "_onehot.txt")
        save_onehot_as_txt(encoded, onehot_txt_output_path)

        all_encoded.append(encoded)

    all_encoded = np.stack(all_encoded)  # [N, 14, 14, num_classes]
    np.save(output_path, all_encoded)
    print(f"Saved one-hot encoded array to {output_path}. Shape: {all_encoded.shape}")
    print(f"Saved individual one-hot slices to: {onehot_txt_dir}")

if __name__ == "__main__":
    txt_patch_dir = "data/processed_text/stride_32_text_overworld"
    json_path = "src/task_2/smb.json"
    output_npy_path = "data/processed_text/stride_32_onehot_overworld.npy"

    process_txt_patches(txt_patch_dir, json_path, output_npy_path)
