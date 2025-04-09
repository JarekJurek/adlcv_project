import os
from PIL import Image

def add_top_stripe(input_dir, output_dir, stripe_height):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue  # skip non-image files

        path = os.path.join(input_dir, filename)
        img = Image.open(path).convert('RGB')

        width, height = img.size

        if stripe_height > height:
            print(f"Skipping {filename}: stripe height exceeds image height.")
            continue

        # Crop the top stripe
        stripe = img.crop((0, 0, width, stripe_height))

        # Create new image with extra height
        new_img = Image.new('RGB', (width, height + stripe_height))
        new_img.paste(stripe, (0, 0))             # paste stripe at top
        new_img.paste(img, (0, stripe_height))    # paste original image below it

        # Save the result
        save_path = os.path.join(output_dir, filename)
        new_img.save(save_path)
        print(f"Processed {filename} → saved to {save_path}")

def remove_bottom_stripe(input_dir, output_dir, stripe_height):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue  # skip non-image files

        path = os.path.join(input_dir, filename)
        img = Image.open(path).convert('RGB')

        width, height = img.size

        if stripe_height >= height:
            print(f"Skipping {filename}: stripe height exceeds or matches image height.")
            continue

        # Crop everything except the bottom stripe
        cropped = img.crop((0, 0, width, height - stripe_height))

        # Save the result
        save_path = os.path.join(output_dir, filename)
        cropped.save(save_path)
        print(f"Cropped {filename} → saved to {save_path}")

# ==== CONFIGURATION ====
input_directory = "/home/gregory/adlcv_project/data/raw/sth"        # Folder with input images
output_directory = "/home/gregory/adlcv_project/data/raw/sth2"      # Folder to save output images
stripe_height = 16                # Number of pixels to remove from bottom


if __name__ == "__main__":
    add_top_stripe(input_directory, output_directory, stripe_height)
    remove_bottom_stripe(output_directory, output_directory, stripe_height)

