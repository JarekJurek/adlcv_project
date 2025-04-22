import json
import numpy as np
from PIL import Image
import os
from pathlib import Path

def load_level(file_path):
    with open(file_path, 'r') as f:
        return [line for line in f.read().splitlines()]

def load_tile_mappings(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_tileset(image_path):
    return Image.open(image_path)

def create_ascii_to_tile_mapping():
    mapping = {
        'S': (16, 0),
        '-': (48, 360),
        '?': (348, 0),
        'Q': (348, 0),
        'E': (420, 130),
        '<': (0, 160),
        '>': (16, 160),
        '[': (0, 176),
        ']': (16, 176),
        'o': (348, 16),
        'B': (144, 0),
        'b': (144, 16),
    }
    return mapping

def render_level(ascii_level, mapping, tileset, tile_size=16):
    height = len(ascii_level)
    width = max(len(line) for line in ascii_level)
    level_image = Image.new('RGB', (width * tile_size, height * tile_size), color=(66, 135, 245))

    for y, line in enumerate(ascii_level):
        for x, char in enumerate(line):
            try:
                if char == 'X':
                    is_bottom = (y == len(ascii_level) - 1)
                    tile_coords = (0, 0) if is_bottom else (0, 16)
                    tile_image = tileset.crop((tile_coords[0], tile_coords[1],
                                               tile_coords[0] + tile_size, tile_coords[1] + tile_size))
                    level_image.paste(tile_image, (x * tile_size, y * tile_size))
                elif char in mapping:
                    tile_coords = mapping[char]
                    tile_image = tileset.crop((tile_coords[0], tile_coords[1],
                                               tile_coords[0] + tile_size, tile_coords[1] + tile_size))
                    level_image.paste(tile_image, (x * tile_size, y * tile_size))
            except Exception as e:
                print(f"Error rendering tile '{char}' at ({x},{y}): {e}")
    return level_image

def main():
    base_dir = Path('C:/Users/filip/adlcv_project-master (1)/adlcv_project-master')
    input_dir = base_dir / 'data/processed_text/stride_32_text_overworld'
    output_dir = base_dir / 'src/task_3/real_images'
    json_path = base_dir / 'src/decoder/smb.json'
    tileset_path = base_dir / 'src/decoder/tiles.png'

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        smb_json = load_tile_mappings(json_path)
        tileset = load_tileset(tileset_path)
        mapping = create_ascii_to_tile_mapping()
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    for txt_file in sorted(input_dir.glob("*.txt")):
        try:
            ascii_level = load_level(txt_file)
            rendered = render_level(ascii_level, mapping, tileset)
            output_path = output_dir / (txt_file.stem + ".png")
            rendered.save(output_path)
            print(f"Rendered {txt_file.name} → {output_path.name}")
        except Exception as e:
            print(f"Failed to render {txt_file.name}: {e}")

if __name__ == "__main__":
    main()
