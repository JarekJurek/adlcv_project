import json
import numpy as np
from PIL import Image
from pathlib import Path

def load_level(file_path):
    """Load the ASCII level from a file."""
    with open(file_path, 'r') as f:
        return [line for line in f.read().splitlines()]

def load_tile_mappings(json_path):
    """Load the mapping information from the JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def load_tileset(image_path):
    """Load the tileset image."""
    return Image.open(image_path)

def create_ascii_to_tile_mapping():
    """Create a mapping from ASCII characters to tile coordinates."""
    # Map each ASCII character to its corresponding tile coordinates
    mapping = {
        'S': (16, 0),
        '-': (48, 360),
        '?': (384, 0),
        'Q': (384, 0),
        'E': (420, 130),
        '<': (0, 160),
        '>': (16, 160),
        '[': (0, 176),
        ']': (16, 176),
        'o': (384, 16),
        'B': (144, 0),
        'b': (144, 16),
        # 'X' is handled specially in the render function
    }
    return mapping

def render_level(ascii_level, mapping, tileset, tile_size=16):
    """Render the level using the tileset."""
    height = len(ascii_level)
    width = max(len(line) for line in ascii_level)
    
    # Create a blank image for the rendered level
    level_image = Image.new('RGB', (width * tile_size, height * tile_size), color=(66, 135, 245))  # Sky blue background
    
    # Place each tile on the level image
    for y, line in enumerate(ascii_level):
        for x, char in enumerate(line):
            if char == 'X':
                # Special case for 'X' - check if it's on the bottom row
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
    
    return level_image

def main():
    # File paths
    # base_dir = Path(__name__).resolve().parents[2]
    # input_dir = base_dir / 'src/task_4/decoded_ascii_levels'
    # output_dir = base_dir / 'src/task_4/real_images'
    # json_path = base_dir / 'src/decoder/smb.json'
    # tileset_path = base_dir / 'src/decoder/tiles.png'

    level_path = '/zhome/a2/c/213547/DLCV/adlcv_project/adlcv_project-master/src/task_4/decoded_ascii_levels/generated_frame_1.txt'
    json_path = 'smb.json'
    tileset_path = 'tiles.png'
    output_path = 'rendered_level.png'
    
    # Load resources
    ascii_level = load_level(level_path)
    smb_json = load_tile_mappings(json_path)  # Still loading for potential other uses
    tileset = load_tileset(tileset_path)
    
    # Create mapping and render
    mapping = create_ascii_to_tile_mapping()
    rendered_level = render_level(ascii_level, mapping, tileset)
    
    # Save the result
    rendered_level.save(output_path)
    print(f"Rendered level saved to {output_path}")

if __name__ == "__main__":
    main()
