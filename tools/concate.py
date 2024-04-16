import h5py
import os

def concatenate_h5_files(input_files, output_file):
    print(input_files)
    """
    Concatenates multiple HDF5 files into a single HDF5 file.

    Parameters:
        input_files (list of str): List of input HDF5 file paths.
        output_file (str): Path to the output HDF5 file.
    """
    with h5py.File(output_file, 'w') as h5out:
        for file_path in input_files:
            with h5py.File(file_path, 'r') as h5in:
                copy_group(h5in, h5out)

def copy_group(h5in, h5out):
    """
    Recursively copies all groups and datasets from one HDF5 file to another.
    Assumes that the output HDF5 file is empty and has no overlapping keys.

    Parameters:
        h5in (h5py.File): Input HDF5 file handle.
        h5out (h5py.File): Output HDF5 file handle where data needs to be copied.
    """
    for name in h5in:
        if isinstance(h5in[name], h5py.Dataset):
            # Copy dataset
            h5out.copy(h5in[name], name)
        elif isinstance(h5in[name], h5py.Group):
            # Create new group and recurse
            grp = h5out.create_group(name)
            copy_group(h5in[name], grp)

if __name__ == "__main__":
    # find all .h5 files in directory
    target_directory = '/data2/peter/'
    input_files = [os.path.join(target_directory, f) for f in os.listdir(target_directory) if f.endswith('.h5')]

    output_file = os.path.join(target_directory, 'combined_output.h5')
    concatenate_h5_files(input_files, output_file)
