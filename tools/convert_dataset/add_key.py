import h5py
import torch
from tqdm import tqdm

# Define the path to your HDF5 file

def process_heatmaps(file_path):
    with h5py.File(file_path, 'r+') as f:
        # Process every key in the file
        for key in tqdm(f.keys(), desc='Processing datasets'):
            heatmaps = f[key]['heatmap']
            num_items = heatmaps.shape[0]

            # Prepare to store the tap_points for this dataset
            tap_point_dataset_name = f'{key}/tap_point'
            if tap_point_dataset_name in f:
                del f[tap_point_dataset_name]  # Remove existing dataset if it exists
            tap_points = f.create_dataset(tap_point_dataset_name, (2), dtype='float32')

            # Process each heatmap
            for i in range(num_items):
                heatmap = torch.tensor(heatmaps[i])  # Convert to torch tensor

                # Flatten the heatmap and find the max value's flat index
                max_val, flat_index = torch.max(heatmap.flatten(), 0)

                # Calculate the original indices
                dims = heatmap.shape
                max_index = [
                    flat_index // (dims[0] * dims[1]),
                    (flat_index % (dims[0] * dims[1])) // dims[1],
                    flat_index % dims[1]
                ]

                # Normalize the indices
                max_index[1] = max_index[1] / dims[0]  # Normalize y-coordinate
                max_index[2] = max_index[2] / dims[1]  # Normalize x-coordinate

                # Convert indices to float tensor and store only the normalized x, y coordinates
                max_index_float = torch.tensor(max_index, dtype=torch.float)
                tap_points = max_index_float[1:3]

    print("All transformations completed successfully!")

def process_frames(file_path):
    with h5py.File(file_path, 'r') as f:
        # Create a new h5 file to save the updated data
        new_file_path = file_path.replace('.h5', '_updated.h5')
        with h5py.File(new_file_path, 'w') as new_f:
            for key in tqdm(f.keys(), desc='Processing datasets'):
                # Copy all the keys from the original file to the new file
                new_f.create_group(key)
                for subkey in f[key].keys():
                    if subkey != 'image_frames':
                        new_f[key].copy(f[key][subkey], subkey)

                frames = f[key]['image_frames'][:]
                frames = frames / 255.0
                
                # Update the 'image_frames' key with the processed frames
                new_f[key].create_dataset('image_frames', data=frames)

    print("All transformations completed successfully!")


if __name__ == "__main__":
    # find all .h5 files in the directory
    import os
    directory = '/data2/peter/validation_set/manual/origin'
    for filename in os.listdir(directory):
        if filename.endswith(".h5"):
            file_path = os.path.join(directory, filename)
            process_frames(file_path)
        else:
            continue