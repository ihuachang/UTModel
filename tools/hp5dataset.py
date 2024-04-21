import h5py
import os
import torch
import pickle
import csv
from collections import defaultdict

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, decode_type ="heatmap", csv_file=None, info=False, type="train", demo=False):
        self.hdf5_file = data_dir
        self.file = h5py.File(self.hdf5_file, 'r')
        all_keys = list(self.file.keys())
        other_keys = []
        if csv_file is not None:
            csv_keys = set()
            with open(csv_file, newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    app_name, trace_id, _ = row
                    formatted_key = f"{app_name.replace('.', '_')}_{trace_id.replace(',', '_')}"
                    csv_keys.add(formatted_key)
            
            for key in all_keys:
                if any(csv_key in key for csv_key in csv_keys):
                    self.test_keys.append(key)
                else:
                    other_keys.append(key)
            
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

        if decode_type == "heatmap":
            self.label = "heatmap"
            self.label2 = "tap_point"
        else:
            self.label = "tap_point"
            self.label2 = "heatmap"

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.file[self.keys[idx]]
        image_frames = torch.from_numpy(data['image_frames'][:])
        text = torch.from_numpy(data['ui_annotations_text_embeddings'][:])
        bound = torch.from_numpy(data['ui_annotations_positions'][:])
        mask = torch.from_numpy(data['ui_annotations_attention_mask'][:])
        label = torch.from_numpy(data[self.label][:])
        label2 = torch.from_numpy(data[self.label2][:])

        return text, bound, mask, image_frames, label, label2

    def __del__(self):
        self.file.close()  # Ensure we close the file when the dataset object is deleted


class MultiFileDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, decode_type ="heatmap", csv_file=None, info=False, type="train", demo=False):
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
            self.val_dataset_keys = self.val_dataset_keys[:100]
        
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

class CachedFileLoader:
    def __init__(self):
        self.open_files = {}

    def get_file(self, file_path):
        if file_path not in self.open_files:
            self.open_files[file_path] = h5py.File(file_path, 'r')
        return self.open_files[file_path]

    def close_all(self):
        for file in self.open_files.values():
            file.close()
        self.open_files = {}

def point_collate_fn(batch):
    loader = CachedFileLoader()
    grouped_data = defaultdict(list)
    
    for file_path, key in batch:
        grouped_data[file_path].append(key)
    
    # Use dictionaries to collect tensors by type
    batched_data = {
        'texts': [], 'bounds': [], 'masks': [], 
        'image_frames': [], 'bbox': [], 'heatmaps': []
    }

    try:
        for file_path, keys in grouped_data.items():
            file = loader.get_file(file_path)
            for key in keys:
                data = file[key]
                batched_data['texts'].append(torch.from_numpy(data['ui_annotations_text_embeddings'][:]))
                batched_data['bounds'].append(torch.from_numpy(data['ui_annotations_positions'][:]))
                batched_data['masks'].append(torch.from_numpy(data['ui_annotations_attention_mask'][:]))
                batched_data['image_frames'].append(torch.from_numpy(data['image_frames'][:]))
                batched_data['heatmaps'].append(torch.from_numpy(data['heatmap'][:]))
                batched_data['bbox'].append(torch.from_numpy(data['bbox'][:]))

        # Ensure all elements are converted to tensors and then stacked
        for key in batched_data:
            batched_data[key] = torch.stack(batched_data[key])
        
        return (
            batched_data['texts'], batched_data['bounds'], batched_data['masks'],
            batched_data['image_frames'], batched_data['bbox'], batched_data['heatmaps']
        )
    finally:
        loader.close_all()

def heatmap_collate_fn(batch):
    loader = CachedFileLoader()
    grouped_data = defaultdict(list)
    
    for file_path, key in batch:
        grouped_data[file_path].append(key)
    
    # Use dictionaries to collect tensors by type
    batched_data = {
        'texts': [], 'bounds': [], 'masks': [], 
        'image_frames': [], 'heatmaps': [], 'tappoints': []
    }

    try:
        for file_path, keys in grouped_data.items():
            file = loader.get_file(file_path)
            for key in keys:
                data = file[key]
                batched_data['texts'].append(torch.from_numpy(data['ui_annotations_text_embeddings'][:]))
                batched_data['bounds'].append(torch.from_numpy(data['ui_annotations_positions'][:]))
                batched_data['masks'].append(torch.from_numpy(data['ui_annotations_attention_mask'][:]))
                batched_data['image_frames'].append(torch.from_numpy(data['image_frames'][:]))
                batched_data['heatmaps'].append(torch.from_numpy(data['heatmap'][:]))
                batched_data['tappoints'].append(torch.from_numpy(data['tap_point'][:]))

        # Ensure all elements are converted to tensors and then stacked
        for key in batched_data:
            batched_data[key] = torch.stack(batched_data[key])
        
        return (
            batched_data['texts'], batched_data['bounds'], batched_data['masks'],
            batched_data['image_frames'], batched_data['heatmaps'], batched_data['tappoints']
        )
    finally:
        loader.close_all()