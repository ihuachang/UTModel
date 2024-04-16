import h5py

def print_keys(h5_file_path):
    # Open the HDF5 file
    with h5py.File(h5_file_path, 'r') as file:
        # Recursively print all keys
        def print_all_keys(obj, indent=0):
            print(len(obj.keys()))
            # for key in obj.keys():
            #     print('    ' * indent + key)  # Print the key with indentation
            #     # If this key is a group, we need to explore its subgroups/datasets
            #     if isinstance(obj[key], h5py.Group):
            #         print_all_keys(obj[key], indent + 1)
            #     elif isinstance(obj[key], h5py.Dataset):
            #         # If it's a dataset, we could also print its shape, type, etc.
            #         print('    ' * (indent + 1) + f"Shape: {obj[key].shape}, Type: {obj[key].dtype}")

        # Start printing keys from the root of the HDF5 file
        print_all_keys(file)

# Usage example:
if __name__ == "__main__":
    hdf5_file_path = '/data2/peter/combined_output.h5'  # Change this to the path of your HDF5 file
    print_keys(hdf5_file_path)