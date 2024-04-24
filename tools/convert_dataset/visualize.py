import h5py
import numpy as np
from PIL import Image, ImageDraw
import os

def load_and_visualize_h5(file_path, output_dir):
    with h5py.File(file_path, 'r') as h5_file:
        for i, group_key in enumerate(h5_file.keys()):
            group = h5_file[group_key]
            images = group['image_frames'][:]  # Shape: (3, 2, 256, 512)
            heatmap = group['heatmap'][:]  # Shape: (1, 256, 512)
            bbox = group['bbox'][:]  # Shape: (1, 4)
            heatmap = heatmap.squeeze()  # Ensuring shape is (256, 512)

            # Normalize heatmap for visualization
            heatmap_normalized = np.uint8(255 * heatmap / np.max(heatmap))
            heatmap_image = Image.fromarray(heatmap_normalized)
            heatmap_image = heatmap_image.convert("RGBA")

            # Process each image in the pair
            for j, image in enumerate(images.transpose(1, 2, 3, 0)):  # Shape: (2, 256, 512, 3)
                image = np.uint8(image * 255)  # Scale back to 0-255
                img = Image.fromarray(image)
                img = img.convert("RGBA")
                
                draw = ImageDraw.Draw(img)
                # Draw the bounding box
                bbox_coords = bbox.squeeze()  # Ensure bbox is a single dimension array
                x_min, y_min, x_max, y_max = bbox_coords[1], bbox_coords[0], bbox_coords[3], bbox_coords[2]
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                x_min, y_min, x_max, y_max = x_min * 512, y_min * 256, x_max * 512, y_max * 256
                
                draw.rectangle([x_min, y_min, x_max, y_max], outline="white", width=5)

                draw.ellipse([center_x - 5, center_y - 5, center_x + 5, center_y + 5], fill="red")

                # Save the image with the bounding box
                img.save(f"{output_dir}/bbox_image_{i}_{j}.png")
                # save heatmap and image
                heatmap_image.save(f"{output_dir}/heatmap_{i}_{j}.png")
                img.save(f"{output_dir}/image_{i}_{j}.png")
                # Blend the heatmap and image
                blended = Image.blend(img, heatmap_image, alpha=0.5)

                # Save the blended image
                blended.save(f"{output_dir}/combined_{i}_{j}.png")

if __name__ == '__main__':
    file_path = '/home/ihua/VLM/tools/convert_dataset/visualize/segment_0_file_0.h5'  # Path to your H5 file
    output_dir = '/home/ihua/VLM/tools/convert_dataset/visualize'  # Directory to save output images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    load_and_visualize_h5(file_path, output_dir)