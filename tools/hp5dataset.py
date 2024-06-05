import h5py
import os
import torch
import pickle
import csv
import random
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
    def __init__(self, data_dir, decode_type ="heatmap", csv_file=None, type="train", demo=False):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.h5')]
        self.val_dataset_keys = []
        self.train_dataset_keys = []
        self.test_dataset_keys = []
        self.dataset_keys = []
        self.file_handles = {}
        self.decode_type = decode_type
        # Extract keys from all files
        if csv_file is not None:
            print("Using CSV file to filter keys")
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
            split_index = int(len(other_keys) * 0.9)
            self.train_dataset_keys = other_keys[:split_index]
            self.val_dataset_keys = other_keys[split_index:]
        
        else:
            for file_path in self.files:
                with h5py.File(file_path, 'r') as file:
                    self.dataset_keys.extend([(file_path, key) for key in file.keys()])
           
            self.dataset_keys.sort(key=lambda x: x[0])
            self.train_dataset_keys = self.dataset_keys[:int(0.9 * len(self.dataset_keys))]
            self.val_dataset_keys = self.dataset_keys[int(0.9 * len(self.dataset_keys)):]
            self.test_dataset_keys = self.dataset_keys

        if demo:
            self.train_dataset_keys = self.train_dataset_keys[:1000]
            self.val_dataset_keys = self.val_dataset_keys[:1000]
        
        if type == "train":
            self.dataset_keys = self.train_dataset_keys
        elif type == "val":
            self.dataset_keys = self.val_dataset_keys
        elif type == "test":
            self.dataset_keys = self.test_dataset_keys
        
        # load all the files
        for file_path in self.files:
            self.open_file(file_path)
    
    def shuffle_blocks(self, block_size=4096):
        # Shuffle each block of 'block_size' within self.dataset_keys
        for start in range(0, len(self.dataset_keys), block_size):
            end = min(start + block_size, len(self.dataset_keys))
            block = self.dataset_keys[start:end]
            random.shuffle(block)
            self.dataset_keys[start:end] = block

    def __len__(self):
        return len(self.dataset_keys)

    def open_file(self, file_path):
        if file_path not in self.file_handles:
            # Open the file and store the handle
            self.file_handles[file_path] = h5py.File(file_path, 'r')
        return self.file_handles[file_path]

    def __getitem__(self, index):
        # Logic to determine which file and key corresponds to this index
        file_path, key = self.dataset_keys[index]
        file = self.file_handles[file_path]
        data = file[key]
        if self.decode_type == "heatmap":
            return data["ui_annotations_text_embeddings"][:], data["ui_annotations_positions"][:], data["ui_annotations_attention_mask"][:], data["image_frames"][:], data["heatmap"][:], data["heatmap"][:]
        else:
            return data["ui_annotations_text_embeddings"][:], data["ui_annotations_positions"][:], data["ui_annotations_attention_mask"][:], data["image_frames"][:], data["bbox"][:], data["heatmap"][:]
    
    def close_files(self):
        for file in self.file_handles.values():
            file.close()
        self.file_handles.clear()
    
    def __del__(self):
        self.close_files()