import h5py
import torch
import numpy as np
from paddleocr import PaddleOCR
from transformers import BertModel, BertTokenizer
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import argparse

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize OCR and BERT
ocr = PaddleOCR(use_angle_cls=True, show_log=False)  # adjust parameters as necessary
bert_model = BertModel.from_pretrained('prajjwal1/bert-tiny').to(DEVICE)
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

def get_bert_embeddings_batched(texts, max_length=512):
    tokenized_texts = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    input_ids = tokenized_texts["input_ids"].to(DEVICE)
    attention_masks = tokenized_texts["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_masks)
    embeddings = outputs['pooler_output'].cpu().numpy()
    return embeddings

def ensure_correct_format(image_frames):
    # Convert to uint8 if not already and ensure BGR format
    if image_frames.dtype != np.uint8:
        image_frames = np.clip(image_frames * 255, 0, 255).astype(np.uint8)  # assuming input might be float
    return image_frames

def pad_embeddings_and_positions(embeddings, positions, max_pad, score):
    """
    Pad embeddings and positions to ensure they are all the same length for batch processing.
    """
    # Convert to tensors
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    positions = torch.tensor(positions, dtype=torch.float)
    score = torch.tensor(score, dtype=torch.float)
    
    # Calculate current number of detections and the necessary padding amount
    current_length = embeddings.shape[0]
    pad_length = max_pad - current_length

    # Initialize the attention mask with ones for actual data and zeros for padding
    attention_mask = torch.ones(max_pad, dtype=torch.float)
    attention_mask[current_length:] = 0  # Set padding area to 0

    # Pad embeddings and positions if necessary
    if pad_length >= 0:
        # Padding embeddings
        embeddings_padded = F.pad(embeddings, (0, 0, 0, pad_length), "constant", 0)
        # Padding positions. Since positions are 3D, we need to pad the first dimension
        positions_padded = F.pad(positions, (0, 0, 0, 0, 0, pad_length), "constant", 0)
    else:
        # Only select top max_pad embeddings and positions with the highest score
        _, indices = torch.topk(score, max_pad)
        embeddings_padded = embeddings[indices]
        positions_padded = positions[indices]

    return embeddings_padded, positions_padded, attention_mask

def process_dataset(input_hdf5_file, output_hdf5_file, segment_number, num_segments, test=False, max_pad=50):
    with h5py.File(input_hdf5_file, 'r') as infile:
        keys = list(infile.keys())
        if test:
            keys = keys[:10]
        total_keys = len(keys)
        segment_size = total_keys // num_segments
        start_index = segment_number * segment_size
        end_index = start_index + segment_size if segment_number != num_segments - 1 else total_keys

        segment_keys = keys[start_index:end_index]

        with h5py.File(output_hdf5_file + f"_part_{segment_number}.h5", 'w') as outfile:
            for key in tqdm(segment_keys, desc=f"Processing segment {segment_number}"):
                pass_flag = 0
                group = infile[key]
                concatenated_frames = group['concatenated_frames'][:]
                
                # Extract text and bounding boxes using OCR
                image_frames = torch.tensor(group['image_frames']).permute(1, 3, 2, 0)
                # convert torch tensor to numpy array
                image_frames = image_frames.numpy()
                image_frames = ensure_correct_format(image_frames)

                processed_data = []
                for frame in [image_frames[0], image_frames[-1]]:
                    frame_text = ocr.ocr(frame, cls=True)[0]
                    if frame_text is None:
                        pass_flag = 1
                    else:
                        texts = [item[1][0] for item in frame_text]
                        positions = [item[0] for item in frame_text]
                        scores = [item[1][1] for item in frame_text]

                    for pos in positions:
                        for point in pos:
                            point[0] = point[0] / frame.shape[1]
                            point[1] = point[1] / frame.shape[0]

                    if len(texts) == 0:
                        embeddings = np.zeros((1, 128))
                    else:
                        embeddings = get_bert_embeddings_batched(texts)
                    embeddings_padded, positions_padded, mask_padded = pad_embeddings_and_positions(embeddings, positions, max_pad, scores)

                    processed_data.append((embeddings_padded, positions_padded, mask_padded))

                if pass_flag == 1:
                    continue

                out_group = outfile.create_group(key)
                for dset in group:
                    out_group.create_dataset(dset, data=group[dset][:])

                concatenated_embeddings = np.concatenate([data[0] for data in processed_data], axis=0)
                concatenated_positions = np.concatenate([data[1] for data in processed_data], axis=0)
                concatenated_masks = np.concatenate([data[2] for data in processed_data], axis=0)

                out_group.create_dataset('ui_annotations_text_embeddings', data=concatenated_embeddings, dtype=np.float32)
                out_group.create_dataset('ui_annotations_positions', data=concatenated_positions, dtype=np.float32)
                out_group.create_dataset('ui_annotations_attention_mask', data=concatenated_masks, dtype=np.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process part of a dataset.")
    parser.add_argument('segment_number', type=int, help="The segment number to process (0-indexed).")
    parser.add_argument('num_segments', type=int, help="Total number of segments.")
    args = parser.parse_args()

    input_file = '/home/ihua/replay/rico_dataset.h5'
    output_file = '/data2/peter/rico_vlm_dataset'
    process_dataset(input_file, output_file, args.segment_number, args.num_segments, test=False)
