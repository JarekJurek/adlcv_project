import os
import cv2
import numpy as np
from pathlib import Path

def split_image_sliding_window(image, patch_size=64, stride=32):
    h, w, _ = image.shape
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(((x, y), patch))

    return patches

def process_images(input_dir, output_dir, patch_size=64, stride=32):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_file in input_path.glob("*.png"):
        print(image_file)
        image = cv2.imread(str(image_file))
        patches = split_image_sliding_window(image, patch_size, stride)

        for idx, ((x, y), patch) in enumerate(patches):
            patch_filename = output_path / f"{image_file.stem}_x{x}_y{y}.png"
            cv2.imwrite(str(patch_filename), patch)

        print(f"Processed {image_file.name}: {len(patches)} patches")


def main():
    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data"


    patch_size = 224  # square size
    stride = 64      # desired stride

    input_dir = data_dir / "raw" / "day_good"
    output_dir = data_dir / "processed" / f"stride_{stride}"

    process_images(input_dir, output_dir, patch_size, stride)


if __name__ == "__main__":
    main()