import h5py
import json
import os
from tqdm import tqdm

def delete_incomplete_entries(h5_file_path, required_keys):
    # Open the HDF5 file
    with h5py.File(h5_file_path, 'a') as file:  # Use mode 'a' to allow editing
        keys_to_delete = []
        # Iterate through all entries in the file
        for entry_key in tqdm(file.keys(), desc="Checking entries"):
            entry = file[entry_key]
            # Check if all required keys are present in this entry
            if not all(key in entry for key in required_keys):
                keys_to_delete.append(entry_key)
        
        # Delete entries that are incomplete
        for key in keys_to_delete:
            del file[key]
            print(f"Deleted incomplete entry: {key}")

# Function to count elements in each key
def count_elements(json_data):
    counts = {}
    for key in json_data:
        # Count the number of elements for each key
        counts[key] = len(json_data[key])
    return counts

def print_keys(h5_file_path):
    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as file:
        # Recursively print all keys
        def print_all_keys(obj, indent=0):
            for key in obj.keys():
                print('    ' * indent + key)  # Print the key with indentation
                # If this key is a group, we need to explore its subgroups/datasets
                if isinstance(obj[key], h5py.Group):
                    print_all_keys(obj[key], indent + 1)
                elif isinstance(obj[key], h5py.Dataset):
                    # If it's a dataset, we could also print its shape, type, etc.
                    print('    ' * (indent + 1) + f"Shape: {obj[key].shape}, Type: {obj[key].dtype}")
        # Start printing keys from the root of the HDF5 file
        print_all_keys(file)
        exit()

if __name__ == "__main__":
    directory_path = '/data2/peter/aiw'  # Path to the directory containing HDF5 files
    required_keys = ['bbox', 'heatmap', 'image_frames', 'tap_point', 'ui_annotations_attention_mask', 'ui_annotations_positions', 'ui_annotations_text_embeddings']
    
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.h5'):
            h5_file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {h5_file_path}")
            delete_incomplete_entries(h5_file_path, required_keys)


    # If you have JSON data in a file, use the following to read it:
    # with open('/home/ihua/UTModel/tools/convert_dataset/android-in-the-wild_splits_standard.json', 'r') as file:
    #     json_data = json.load(file)
    #     element_counts = count_elements(json_data)
    #     for key, count in element_counts.items():
    #         print(f"The number of elements in '{key}': {count}")

