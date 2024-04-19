import h5py
import os
import torch
import pickle
import csv
from collections import defaultdict

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv_file=None, info=False, type="train", demo=False):
        self.hdf5_file = data_dir
        self.file = h5py.File(self.hdf5_file, 'r')
        all_keys = list(self.file.keys())

        if csv_file is not None:
            csv_keys = set()
            with open(csv_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    app_name, trace_id, _ = row
                    formatted_key = f"{app_name.replace('.', '_')}_{trace_id.replace(',', '_')}"
                    csv_keys.add(formatted_key)

            self.test_keys = [key for key in all_keys if any(csv_key in key for csv_key in csv_keys)]
            other_keys = [key for key in all_keys if key not in self.test_keys].sort()
            self.training_keys = other_keys[:int(0.8 * len(other_keys))]
            self.validation_keys = other_keys[int(0.8 * len(other_keys)):]
        else:
            # sort the keys
            all_keys.sort()
            self.training_keys = all_keys[:int(0.8 * len(all_keys))]
            self.validation_keys = all_keys[int(0.8 * len(all_keys)):int(0.95 * len(all_keys))]
            self.test_keys = all_keys[int(0.95 * len(all_keys)):]

        if type == "train":
            self.keys = self.training_keys
        elif type == "val":
            self.keys = self.validation_keys
        elif type == "test":
            self.keys = self.test_keys

        if demo:
            self.keys = self.keys[:1000]

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
    def __init__(self, data_dir, csv_file=None, info=False, type="train", demo=False):
        self.data_dir = data_dir
        # List all .h5 files in the specified directory
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.val_dataset_keys = []
        self.train_dataset_keys = []
        self.test_dataset_keys = []
        self.dataset_keys = []

        # Extract keys from all files
        if csv_file is not None:
            csv_keys = set()
            with open(csv_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    app_name, trace_id, id = row
                    formatted_key = f"{app_name}_{trace_id}_{id}"
                    csv_keys.add(formatted_key)
            
            other_keys = []
            for file_path in self.files:
                with h5py.File(file_path, 'r') as file:
                    for key in file.keys():
                        if any(csv_key in key for csv_key in csv_keys):
                            self.test_dataset_keys.append((file_path, key))
                        else:
                            other_keys.append((file_path, key))
            
            other_keys = sorted(other_keys, key=lambda x: x[0])
            split_index = int(len(other_keys) * 0.8)
            self.train_dataset_keys = other_keys[:split_index]
            self.val_dataset_keys = other_keys[split_index:]
            
        else:
            for file_path in self.files:
                with h5py.File(file_path, 'r') as file:
                    self.dataset_keys.extend([(file_path, key) for key in file.keys()])
           
            self.dataset_keys.sort(key=lambda x: x[0])
            self.train_dataset_keys = self.dataset_keys[:int(len(self.dataset_keys) * 0.8)]
            self.val_dataset_keys = self.dataset_keys[int(len(self.dataset_keys) * 0.8):int(len(self.dataset_keys) * 0.95)]
            self.test_dataset_keys = self.dataset_keys[int(len(self.dataset_keys) * 0.95):]

        if demo:
            self.train_dataset_keys = self.train_dataset_keys[:1000]
            self.val_dataset_keys = self.val_dataset_keys[:500]
        
        if type == "train":
            self.dataset_keys = self.train_dataset_keys
        elif type == "val":
            self.dataset_keys = self.val_dataset_keys
        elif type == "test":
            self.dataset_keys = self.test_dataset_keys
        

    def __len__(self):
        return len(self.dataset_keys)

    def __getitem__(self, idx):
        file_path, key = self.dataset_keys[idx]
        # Return file path and key so that collate_fn can use it to load data
        return file_path, key

def point_collate_fn(batch):
    grouped_data = defaultdict(list)
    # Group data by the file path to minimize file open/close operations
    for file_path, key in batch:
        grouped_data[file_path].append(key)

    # Lists to store the batched data
    batched_texts = []
    batched_bounds = []
    batched_masks = []
    batched_image_frames = []
    batched_point = []
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
                max_val, flat_index = torch.max(heatmap.flatten(), 0)
                dims = heatmap.shape
                max_index = [flat_index // (dims[1] * dims[2]), (flat_index % (dims[1] * dims[2])) // dims[2], flat_index % dims[2]]
                # normalize the point
                max_index[1] = max_index[1] / dims[1]
                max_index[2] = max_index[2] / dims[2]
                # Converting index to a floating-point tensor
                max_index_float = torch.tensor(max_index, dtype=torch.float)
                batched_point.append(max_index_float[1:3])
                batched_heatmaps.append(heatmap)

    # Return lists of tensors instead of trying to convert them into a single tensor
    batched_texts = torch.stack(batched_texts)
    batched_bounds = torch.stack(batched_bounds)
    batched_masks = torch.stack(batched_masks)
    batched_image_frames = torch.stack(batched_image_frames)
    batched_point = torch.stack(batched_point)
    batched_heatmaps = torch.stack(batched_heatmaps)
    
    return batched_texts, batched_bounds, batched_masks, batched_image_frames, batched_point, batched_heatmaps

def heatmap_collate_fn(batch):
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
    dummy = torch.zeros(batched_heatmaps.shape[0], 1, batched_heatmaps.shape[2], batched_heatmaps.shape[3])
    return batched_texts, batched_bounds, batched_masks, batched_image_frames, batched_heatmaps, dummy