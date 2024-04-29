import numpy as np
import torch
import torch.nn.functional as F
import os
from transformers import BertModel, BertTokenizer
from paddleocr import PaddleOCR
from skimage.transform import resize
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Load the BERT model and tokenizer
DEVICE = torch.device('cpu')
MAX_PAD = 62  # Maximum padding for batch uniformity

ocr = PaddleOCR(use_angle_cls=True, show_log=False)  # adjust parameters as necessary
bert_model = BertModel.from_pretrained('prajjwal1/bert-tiny').to(DEVICE)
tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-tiny')

def get_bert_embeddings_batched(texts, max_length=512):
    """Compute BERT embeddings for a batch of text."""
    if len(texts) == 0:
        return np.zeros((1, 128), dtype=np.float16)
    tokenized_texts = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)
    input_ids = tokenized_texts["input_ids"].to(DEVICE)
    attention_masks = tokenized_texts["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_masks)
    embeddings = outputs['pooler_output'].cpu().numpy()
    return embeddings

def pad_embeddings_and_positions(embeddings, positions, score, max_pad=MAX_PAD):
    """
    Pad embeddings and positions to ensure they are all the same length for batch processing.
    """
    # Convert to tensors
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    positions = torch.tensor(positions, dtype=torch.float)
    if len(positions) == 0:
        positions = torch.zeros((1, 4, 2), dtype=torch.float)
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

    # convert to float16
    embeddings_padded = embeddings_padded.to(torch.float16)
    positions_padded = positions_padded.to(torch.float16)
    attention_mask = attention_mask.to(torch.float16)

    return embeddings_padded, positions_padded, attention_mask

def resize_image(input_image, output_size):
    # Ensure input is in the correct format
    if input_image.dtype != np.uint8:
        input_image = np.clip(input_image * 255, 0, 255)  # Scale to 0-255 and clip if not already
        input_image = input_image.astype(np.uint8)  # Ensure it's uint8 for PIL compatibility

    # Convert the numpy array to a PIL Image
    img = Image.fromarray(input_image)

    # Resize using BICUBIC interpolation (you can change this as needed)
    resized_img = img.resize((output_size[1], output_size[0]), Image.BICUBIC)  # Width, Height for PIL
    
    # Convert back to numpy array
    resized_np = np.array(resized_img)

    return resized_np

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

def get_ocr_text(frame):
    frame_text = ocr.ocr(frame, cls=True)[0]
    if frame_text is None:
        return [], [], []
    texts = [item[1][0] for item in frame_text]
    positions = [item[0] for item in frame_text]
    scores = [item[1][1] for item in frame_text]
    for pos in positions:
        for point in pos:
            point[0] = point[0] / frame.shape[1]
            point[1] = point[1] / frame.shape[0]
    return texts, positions, scores