import numpy as np
import json
from pathlib import Path

def load_idx_to_char(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    chars = sorted(data['tiles'].keys())
    return {i: char for i, char in enumerate(chars)}

def decode_onehot_to_ascii(onehot_array, idx_to_char):
    if onehot_array.shape[0] == 16:
        onehot_array = onehot_array[1:-1, 1:-1, :]
    indices = np.argmax(onehot_array, axis=-1)
    ascii_grid = np.vectorize(idx_to_char.get)(indices)
    return ascii_grid

def save_ascii_grid(ascii_grid, output_path):
    with open(output_path, 'w') as f:
        for row in ascii_grid:
            f.write(''.join(row) + '\n')

def convert_all_npy_to_ascii(npy_dir, output_dir, json_path):
    npy_dir = Path(npy_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idx_to_char = load_idx_to_char(json_path)

    for npy_file in sorted(npy_dir.glob("*.npy")):
        onehot = np.load(npy_file)
        ascii_grid = decode_onehot_to_ascii(onehot, idx_to_char)
        output_path = output_dir / (npy_file.stem + ".txt")
        save_ascii_grid(ascii_grid, output_path)
        print(f"Converted {npy_file.name} → {output_path.name}")

if __name__ == "__main__":
    npy_dir = "src/task_3/generated_txt/"
    output_dir = "src/task_3/decoded_ascii_levels"
    json_path = "src/decoder/smb.json"

    convert_all_npy_to_ascii(npy_dir, output_dir, json_path)
