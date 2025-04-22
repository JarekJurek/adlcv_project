import os
from pathlib import Path
import numpy as np

def split_text_sliding_window(char_array, patch_size=14, stride=4):
    h, w = char_array.shape
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = char_array[y:y + patch_size, x:x + patch_size]
            patches.append(((x, y), patch))

    return patches

def process_text_files(input_dir, output_dir, patch_size=14, stride=4):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for text_file in input_path.glob("*.txt"):
        print(text_file)
        with open(text_file, "r") as f:
            lines = [list(line.rstrip()) for line in f.readlines()]
            char_array = np.array(lines)

        patches = split_text_sliding_window(char_array, patch_size, stride)

        for idx, ((x, y), patch) in enumerate(patches):
            patch_filename = output_path / f"{text_file.stem}_x{x}_y{y}.txt"
            with open(patch_filename, "w") as pf:
                for row in patch:
                    pf.write("".join(row) + "\n")

        print(f"Processed {text_file.name}: {len(patches)} patches")

def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"

    patch_size = 14  # 14 characters = 224 pixels
    stride = 2       # 4 characters = 64 pixels

    input_dir = data_dir / "raw_text_overworld"
    output_dir = data_dir / "processed_text/stride_32_text_overworld"

    process_text_files(input_dir, output_dir, patch_size, stride)

if __name__ == "__main__":
    main()
