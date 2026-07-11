from PIL import Image
import math

def combine_images_grid(image_paths, output_path="combined_grid.png", cols=2):
    """
    Combine multiple images into a grid (default: 2x2)
    """
    images = [Image.open(path) for path in image_paths]
    count = len(images)
    rows = math.ceil(count / cols)

    # Resize all images to the same width and height (optional, to align perfectly)
    widths, heights = zip(*(img.size for img in images))
    max_width = max(widths)
    max_height = max(heights)

    # Create a blank canvas for the grid
    combined = Image.new("RGB", (cols * max_width, rows * max_height), color=(255, 255, 255))

    # Paste images in grid
    for idx, img in enumerate(images):
        x = (idx % cols) * max_width
        y = (idx // cols) * max_height
        combined.paste(img, (x, y))

    combined.save(output_path)
    print(f"✅ Saved {output_path} ({cols}x{rows} grid)")

# Example usage:
image_files = ["1.png", "2.png", "3.png", "4.png"]
combine_images_grid(image_files, "grid_image.png")
