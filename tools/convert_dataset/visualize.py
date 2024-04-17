import h5py
import numpy as np
from PIL import Image
import os

def load_and_visualize_h5(file_path, output_dir):
    with h5py.File(file_path, 'r') as h5_file:
        for i, group_key in enumerate(h5_file.keys()):
            group = h5_file[group_key]
            images = group['image_frames'][:]  # Shape: (3, 2, 256, 512)
            heatmap = group['heatmap'][:]  # Shape: (1, 256, 512)
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
                
                # save heatmap and image
                heatmap_image.save(f"{output_dir}/heatmap_{i}_{j}.png")
                img.save(f"{output_dir}/image_{i}_{j}.png")
                # Blend the heatmap and image
                blended = Image.blend(img, heatmap_image, alpha=0.5)

                # Save the blended image
                blended.save(f"{output_dir}/combined_{i}_{j}.png")

if __name__ == '__main__':
    file_path = '/data2/peter/aiw/processed_data.h5'  # Path to your H5 file
    output_dir = '/home/ihua/VLM/tools/convert_dataset/visualize'  # Directory to save output images
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    load_and_visualize_h5(file_path, output_dir)