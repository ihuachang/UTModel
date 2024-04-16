import h5py
import os
import torch
import pickle
from collections import defaultdict

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv = None, info=False, train=True, demo=False):
        self.hdf5_file = data_dir
        self.file = h5py.File(self.hdf5_file, 'r')
        self.keys = list(self.file.keys())
        split_index = int(len(self.keys) * 0.8) if len(self.keys) != 1 else 1

        if train:
            self.keys = self.keys[:split_index]
        else:
            self.keys = self.keys[split_index:]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.file[self.keys[idx]]
        image_frames = torch.from_numpy(data['image_frames'][:])
        heatmap = torch.from_numpy(data['heatmap'][:])
        text = torch.from_numpy(data['ui_annotations_text_embeddings'][:])
        bound = torch.from_numpy(data['ui_annotations_positions'][:])
        mask = torch.from_numpy(data['ui_annotations_attention_mask'][:])

        return text, bound, mask, image_frames, heatmap

    def __del__(self):
        self.file.close()  # Ensure we close the file when the dataset object is deleted


class MultiFileDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv = None, info=False, train=True, demo=False):
        self.data_dir = data_dir
        # List all .h5 files in the specified directory
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.dataset_keys = []
        # Extract keys from all files
        for file_path in self.files:
            with h5py.File(file_path, 'r') as file:
                self.dataset_keys.extend([(file_path, key) for key in file.keys()])
        
        # Split the dataset into training and testing based on an 80/20 split
        split_index = int(len(self.dataset_keys) * 0.8)
        if train:
            self.dataset_keys = self.dataset_keys[:split_index]
        else:
            self.dataset_keys = self.dataset_keys[split_index:]

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        file_path, key = self.dataset_keys[idx]
        # Return file path and key so that collate_fn can use it to load data
        return file_path, key

def custom_collate_fn(batch):
    grouped_data = defaultdict(list)
    # Group data by the file path to minimize file open/close operations
    for file_path, key in batch:
        grouped_data[file_path].append(key)

    # Lists to store the batched data
    batched_texts = []
    batched_bounds = []
    batched_masks = []
    batched_image_frames = []
    batched_heatmaps = []

    # Process each file's data
    for file_path, keys in grouped_data.items():
        with h5py.File(file_path, 'r') as file:
            for key in keys:
                data = file[key]
                text = torch.from_numpy(data['ui_annotations_text_embeddings'][:])
                bound = torch.from_numpy(data['ui_annotations_positions'][:])
                mask = torch.from_numpy(data['ui_annotations_attention_mask'][:])
                image_frames = torch.from_numpy(data['image_frames'][:])
                heatmap = torch.from_numpy(data['heatmap'][:])

                batched_texts.append(text)
                batched_bounds.append(bound)
                batched_masks.append(mask)
                batched_image_frames.append(image_frames)
                batched_heatmaps.append(heatmap)

    # Return lists of tensors instead of trying to convert them into a single tensor
    batched_texts = torch.stack(batched_texts)
    batched_bounds = torch.stack(batched_bounds)
    batched_masks = torch.stack(batched_masks)
    batched_image_frames = torch.stack(batched_image_frames)
    batched_heatmaps = torch.stack(batched_heatmaps)
    
    return batched_texts, batched_bounds, batched_masks, batched_image_frames, batched_heatmaps
