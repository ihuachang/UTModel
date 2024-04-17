import tensorflow as tf
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import os
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from skimage.transform import resize
import h5py
from PIL import Image
from skimage import img_as_ubyte
import argparse

RAW_PATH = '/data/poyang/android-in-the-wild/General'
# SAVE_PATH = '/data2/peter/aiw'
SAVE_PATH = './'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_PAD = 50  # Maximum padding for batch uniformity
TEST = False
WIDTH, HEIGHT = 256, 512

# Load the BERT model and tokenizer
bert_model = BertModel.from_pretrained('prajjwal1/bert-tiny').to(DEVICE)
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

def is_tap_action(
    normalized_start_yx, normalized_end_yx
):
  distance = np.linalg.norm(
      np.array(normalized_start_yx) - np.array(normalized_end_yx)
  )
  return distance <= 0.04

def find_bounding_box(touch_coordinates, positions):
    """
    Find the bounding box containing the touch point.
    
    Args:
    touch_coordinates (np.array): An array of the touch coordinates [y, x].
    positions (np.array): An array of shape (n, 4, 2) where n is the number of UI elements, 
                          4 for each corner of the bounding box, and 2 for (y, x) coordinates of each corner.

    Returns:
    tuple: The bounding box (x_min, y_min, x_max, y_max) that contains the touch point, or None if no such box exists.
    """
    # Transform touch coordinates to be comparable with positions
    y_touch, x_touch = touch_coordinates
    x_min, x_max, y_min, y_max = x_touch, x_touch, y_touch, y_touch
    for bbox in positions:
        # bbox shape is (4, 2) for (y, x) at each corner [top_left, top_right, down_right, down_left]
        y_coords, x_coords = bbox[:, 0], bbox[:, 1]
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        if y_min <= y_touch <= y_max and x_min <= x_touch <= x_max:
            return (x_min, y_min, x_max, y_max)
    width, height = x_max - x_min, y_max - y_min
    if width < 1/5 :
        offset = (1/5 - width) / 2
        x_min = max(0, x_min - offset)
        x_max = min(1, x_max + offset)
    if height < 1/10 :
        offset = (1/10 - height) / 2
        y_min = max(0, y_min - offset)
        y_max = min(1, y_max + offset)
    
    return (x_min, y_min, x_max, y_max)

def get_bert_embeddings_batched(texts, max_length=512):
    """Compute BERT embeddings for a batch of text."""
    tokenized_texts = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    input_ids = tokenized_texts["input_ids"].to(DEVICE)
    attention_masks = tokenized_texts["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_masks)
    embeddings = outputs['pooler_output'].cpu().numpy()
    return embeddings

def pad_embeddings_and_positions(embeddings, positions, max_pad=MAX_PAD):
    """Pad embeddings and positions to the same length for batch processing."""
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    positions = torch.tensor(positions, dtype=torch.float)

    current_length = embeddings.shape[0]
    pad_length = max_pad - current_length
    attention_mask = torch.ones(max_pad, dtype=torch.float)
    attention_mask[current_length:] = 0

    embeddings_padded = F.pad(embeddings, (0, 0, 0, pad_length), "constant", 0)
    positions_padded = F.pad(positions, (0, 0, 0, 0, 0, pad_length), "constant", 0)

    return embeddings_padded, positions_padded, attention_mask

def convert_positions_to_corners(positions):
    """Convert (y, x, height, width) to corners (top_left, top_right, down_right, down_left)."""
    y, x, h, w = positions.T
    top_left = np.array([y, x]).T
    top_right = np.array([y, x + w]).T
    down_right = np.array([y + h, x + w]).T
    down_left = np.array([y + h, x]).T
    return np.stack([top_left, top_right, down_right, down_left], axis=1)

def get_gaussian_heatmap(bbox, shape=(512, 256)):
    """Generate a Gaussian heatmap for a given bounding box."""
    x_min, y_min, x_max, y_max = bbox
    # x multiply 256, y multiply 512
    x_min, x_max = x_min * shape[1], x_max * shape[1]
    y_min, y_max = y_min * shape[0], y_max * shape[0]
    
    X, Y = np.meshgrid(np.linspace(0, shape[1]-1, shape[1]), np.linspace(0, shape[0]-1, shape[0]))
    x, y = (x_min + x_max) / 2, (y_min + y_max) / 2
    width, height = x_max - x_min, y_max - y_min
    sigma_x, sigma_y = width / 2, height / 5
    heatmap = np.exp(-(((X-x)**2)/(2*sigma_x**2) + ((Y-y)**2)/(2*sigma_y**2)))
    return heatmap

def decode_tfrecord(example):
    feature_description = {
        'episode_id': tf.io.FixedLenFeature([], tf.string),
        'episode_length': tf.io.FixedLenFeature([], tf.int64),
        
        'image/channels': tf.io.FixedLenFeature([], tf.int64),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/ui_annotations_positions': tf.io.VarLenFeature(tf.float32),
        'image/ui_annotations_text': tf.io.VarLenFeature(tf.string),
        'image/ui_annotations_ui_types': tf.io.VarLenFeature(tf.string),
        
        'results/action_type': tf.io.FixedLenFeature([], tf.int64),
        'results/type_action': tf.io.FixedLenFeature([], tf.string),
        'results/yx_touch': tf.io.FixedLenFeature([2], tf.float32),
        'results/yx_lift': tf.io.FixedLenFeature([2], tf.float32),
        
        'step_id': tf.io.FixedLenFeature([], tf.int64)
    }
    
    parsed_example = tf.io.parse_single_example(example, feature_description)
    for varlen_key in ['image/ui_annotations_positions', 'image/ui_annotations_text', 'image/ui_annotations_ui_types']:
        parsed_example[varlen_key] = tf.sparse.to_dense(parsed_example[varlen_key])
    raw_image = tf.io.decode_raw(parsed_example['image/encoded'], tf.uint8)
    parsed_example['image/encoded'] = tf.reshape(raw_image, (parsed_example['image/height'], parsed_example['image/width'], parsed_example['image/channels']))
    return parsed_example

def load_tfrecords_to_h5(filenames, segment_number, num_segments, save_path):
    segment_size = len(filenames) // num_segments
    start_index = segment_number * segment_size
    end_index = start_index + segment_size if segment_number != num_segments - 1 else len(filenames)

    segment_filenames = filenames[start_index:end_index]
    h5_filename = os.path.join(save_path, f'processed_data_segment_{segment_number}.h5')

    raw_dataset = tf.data.TFRecordDataset(segment_filenames, compression_type='GZIP')
    parsed_dataset = raw_dataset.map(decode_tfrecord)

    prev_record = None
    file_index = 0

    with h5py.File(h5_filename, 'w') as h5_file:
        for record in tqdm(parsed_dataset, desc="Processing TFRecords"):
            if prev_record is not None:
                prev_episode_id = prev_record['episode_id'].numpy().decode('utf-8')
                current_episode_id = record['episode_id'].numpy().decode('utf-8')
                
                if prev_episode_id == current_episode_id and is_tap_action(prev_record['results/yx_touch'].numpy(), prev_record['results/yx_lift'].numpy()):
                    # Process both the previous and current record
                    pass_flag = 0
                    combined_data = []
                    for each_record in [prev_record, record]:
                        texts = [text.decode('utf-8') for text in each_record['image/ui_annotations_text'].numpy()]
                        positions = each_record['image/ui_annotations_positions'].numpy().reshape(-1, 4)

                        # Filter out empty texts and corresponding positions
                        filtered_texts = []
                        filtered_positions = []
                        for text, pos in zip(texts, positions):
                            if text.strip() != '':
                                filtered_texts.append(text)
                                filtered_positions.append(pos)

                        # Get embeddings
                        if len(filtered_texts) != 0:
                            embeddings = get_bert_embeddings_batched(filtered_texts)
                        else:
                            pass_flag = 1
                            break

                        # Convert positions from (y, x, h, w) to corners
                        filtered_positions = convert_positions_to_corners(np.array(filtered_positions))

                        # Pad embeddings and positions
                        max_length = MAX_PAD  # Assuming some maximum length for padding
                        embeddings_padded, positions_padded, attention_mask = pad_embeddings_and_positions(embeddings, filtered_positions, max_pad=max_length)

                        combined_data.append((embeddings_padded, positions_padded, attention_mask))

                    # Combine data from previous and current records
                    if all(data[0].shape[0] > 0 for data in combined_data) and pass_flag == 0:  # Ensure there is data to process
                        # Concatenate embeddings, positions, masks from previous and current
                        embeddings_concat = torch.cat([data[0] for data in combined_data], dim=0)
                        positions_concat = torch.cat([data[1] for data in combined_data], dim=0)
                        masks_concat = torch.cat([data[2] for data in combined_data], dim=0)

                        corner_position = convert_positions_to_corners(prev_record['image/ui_annotations_positions'].numpy().reshape(-1, 4))
                        bbox = find_bounding_box(prev_record['results/yx_touch'].numpy(), corner_position)  # Assuming function find_bounding_box() exists
                        heatmap = get_gaussian_heatmap(bbox).reshape(1, 512, 256).transpose(0, 2, 1).astype(np.float32)

                        prev_image = prev_record['image/encoded'].numpy().astype(np.float32)
                        cur_image = record['image/encoded'].numpy().astype(np.float32)

                        # Resize images to a fixed size
                        prev_image = resize(prev_image, (HEIGHT, WIDTH), anti_aliasing=True).transpose(1, 0, 2)
                        cur_image = resize(cur_image, (HEIGHT, WIDTH), anti_aliasing=True).transpose(1, 0, 2)

                        combined_image = np.stack((prev_image, cur_image))  # Shape: (2, WIDTH, HEIGHT, 3)
                        combined_image = np.transpose(combined_image, (3, 0, 1, 2))  # Shape: (3, 2, WIDTH, HEIGHT)

                        # Store in HDF5 (adjust according to actual storage structure)
                        grp = h5_file.create_group(f'entry_{file_index}')
                        grp.create_dataset('ui_annotations_text_embeddings', data=embeddings_concat.numpy())
                        grp.create_dataset('ui_annotations_positions', data=positions_concat.numpy())
                        grp.create_dataset('ui_annotations_attention_mask', data=masks_concat.numpy())
                        grp.create_dataset('heatmap', data=heatmap)
                        grp.create_dataset('image_frames', data=combined_image)

                        file_index += 1
                        if file_index == 10:
                            break

            if file_index % 10000 == 0:
                print(f"Processed {file_index} entries")

            prev_record = record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process part of a dataset.")
    parser.add_argument('segment_number', type=int, help="The segment number to process (0-indexed).")
    parser.add_argument('num_segments', type=int, help="Total number of segments.")
    args = parser.parse_args()

    print(f"Loading from {RAW_PATH}")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    print(f"Saving to {SAVE_PATH}")

    filenames = sorted([os.path.join(RAW_PATH, file) for file in os.listdir(RAW_PATH) if os.path.isfile(os.path.join(RAW_PATH, file))])

    if TEST:
        filenames = filenames[:1]  # Only for testing purposes

    load_tfrecords_to_h5(filenames, args.segment_number, args.num_segments, SAVE_PATH)