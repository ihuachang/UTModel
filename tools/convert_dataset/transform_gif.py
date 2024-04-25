import os
import numpy as np
from PIL import Image, ImageSequence
import json
from tqdm import tqdm
from convert_tools import get_gaussian_heatmap, get_ocr_text, get_bert_embeddings_batched, pad_embeddings_and_positions, resize_image
import h5py
import torch
import logging

DATASET_FOLDER = '/data2/peter/auto_dataset/dataset'
# Directory containing all gifs
GIF_FOLDER = os.path.join(DATASET_FOLDER, 'segments')
# JSON file containing actions and bounding boxes
ACTIONS_JSON_FILE = os.path.join(DATASET_FOLDER, 'action.json')

OUTPUT_FOLDER = os.path.join(DATASET_FOLDER, 'output')
ORIGIN_SIZE = (1080, 2400)
TARGET_SIZE = (256, 512)
MAX_PAD = 62
X_SCALE = TARGET_SIZE[0] / ORIGIN_SIZE[0]
Y_SCALE = TARGET_SIZE[1] / ORIGIN_SIZE[1]

# Load the actions JSON file
with open(ACTIONS_JSON_FILE, 'r') as file:
    actions_data = json.load(file)

def prerocess_bbox():
    for gif_name, action in actions_data.items():
        if 'bbox' not in action:
            continue
        x_min, y_min, x_max, y_max = action['bbox']['column_min'], action['bbox']['row_min'], action['bbox']['column_max'], action['bbox']['row_max']
        x_min, x_max = x_min / ORIGIN_SIZE[0], x_max / ORIGIN_SIZE[0]
        y_min, y_max = y_min / ORIGIN_SIZE[1], y_max / ORIGIN_SIZE[1]
        width, height = x_max - x_min, y_max - y_min

        if width < 1/5 :
            offset = (1/5 - width) / 2
            x_min = max(0, x_min - offset)
            x_max = min(1, x_max + offset)
        if height < 1/10 :
            offset = (1/10 - height) / 2
            y_min = max(0, y_min - offset)
            y_max = min(1, y_max + offset)
        
        action['bbox'] = [np.float16(x_min),
            np.float16(y_min),
            np.float16(x_max),
            np.float16(y_max)]
        

# Function to split a GIF into frames and apply Gaussian heatmap
def process_gif(gif_path, action, num_frame):
    gif = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]
    heatmap = get_gaussian_heatmap(action["bbox"], TARGET_SIZE[::-1])
    processed_frames = []
    step = 1
    for i in range(min(num_frame, len(frames))):
        target_frame = frames[i*step]
        target_frame = np.array(target_frame.convert('RGB'))
        processed_frames.append(target_frame)
    
    if len(processed_frames) < num_frame:
        processed_frames += [processed_frames[-1]] * (num_frame - len(processed_frames))

    return processed_frames, heatmap

def process():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    file_count = 0
    current_file = None
    current_size = 0

    def open_new_file():
        nonlocal current_file, file_count, current_size
        if current_file:
            current_file.close()
        file_path = os.path.join(OUTPUT_FOLDER, f"auto_dataset_{file_count}.h5")
        current_file = h5py.File(file_path, 'w')
        file_count += 1
        current_size = 0  # Reset current file size

    open_new_file()  # Start the first file

    for gif_name, action in tqdm(actions_data.items(), desc="Processing actions"):
        if 'type' not in action or action['type'] != 'click':
            continue
        
        gif_path = os.path.join(GIF_FOLDER, gif_name)
        if not os.path.exists(gif_path):
            logging.warning(f"GIF not found: {gif_path}")
            continue
        

        frames, heatmap = process_gif(gif_path, action, 8)
        heatmap = heatmap.reshape(1, 512, 256).transpose(0, 2, 1).astype(np.float16)
        heatmap = heatmap / np.max(heatmap)
        heatmap[heatmap > 0.9] = 1

        bbox = np.array([action['bbox'][0], action['bbox'][1], action['bbox'][2], action['bbox'][3]]).astype(np.float16)

        processed_data = []
        for frame in [frames[0], frames[-1]]:
            texts, positions, scores = get_ocr_text(frame)
            embeddings = get_bert_embeddings_batched(texts)
            embeddings_padded, positions_padded, mask_padded = pad_embeddings_and_positions(embeddings, positions, scores)
            processed_data.append((embeddings_padded, positions_padded, mask_padded))
        
        embeddings_concat = torch.cat([data[0] for data in processed_data], dim=0)
        positions_concat = torch.cat([data[1] for data in processed_data], dim=0)
        masks_concat = torch.cat([data[2] for data in processed_data], dim=0)

        frames_array = []
        for frame in frames:
            frames_array.append(resize_image(frame, TARGET_SIZE[::-1]).transpose(1, 0, 2).astype(np.float16))

        # stack all frames
        image_frames = np.array(frames_array).transpose((3, 0, 1, 2))

        tap_point = np.array([action['location'][1]/ORIGIN_SIZE[1], action['location'][0]/ORIGIN_SIZE[0]]).astype(np.float16)

        grp = current_file.create_group(f'item_{gif_name[:-4]}')
        grp.create_dataset('ui_annotations_text_embeddings', data=embeddings_concat.numpy())
        grp.create_dataset('ui_annotations_positions', data=positions_concat.numpy())
        grp.create_dataset('ui_annotations_attention_mask', data=masks_concat.numpy())
        grp.create_dataset('heatmap', data=heatmap)
        grp.create_dataset('image_frames', data=image_frames)
        grp.create_dataset('tap_point', data=tap_point)
        grp.create_dataset('bbox', data=bbox)

    if current_file:
        current_file.close()

if __name__ == "__main__":
    prerocess_bbox()
    process()
