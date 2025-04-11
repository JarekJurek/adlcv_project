import json
import numpy as np
from PIL import Image
import os

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
    # These are the specific coordinates provided
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
            try:
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
            except Exception as e:
                print(f"Error rendering tile '{char}' at position ({x}, {y}): {e}")
    
    return level_image

def main():
    # File paths
    base_dir = '/Users/rami/Documents/DTU/Semester 2/ADLCV/supermariotest'
    level_path = os.path.join(base_dir, 'mario-1-1.txt')
    json_path = os.path.join(base_dir, 'smb.json')
    tileset_path = os.path.join(base_dir, 'tiles.png')
    output_path = os.path.join(base_dir, 'rendered_level.png')
    
    # Load resources
    try:
        ascii_level = load_level(level_path)
    except Exception as e:
        print(f"Error loading level file: {e}")
        return
        
    try:
        smb_json = load_tile_mappings(json_path)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    try:
        tileset = load_tileset(tileset_path)
    except Exception as e:
        print(f"Error loading tileset image: {e}")
        return
    
    # Create mapping and render
    mapping = create_ascii_to_tile_mapping()
    rendered_level = render_level(ascii_level, mapping, tileset)
    
    # Save the result
    try:
        rendered_level.save(output_path)
        print(f"Rendered level saved to {output_path}")
    except Exception as e:
        print(f"Error saving rendered level: {e}")
    
    # Display debug information
    print(f"Level dimensions: {rendered_level.width}x{rendered_level.height}")
    print(f"Characters mapped: {sorted(list(mapping.keys()) + ['X'])}")
    print("Note: 'X' is specially handled based on vertical position")

if __name__ == "__main__":
    main()
