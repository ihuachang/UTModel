import numpy as np
import torch
from torch.autograd import Variable
from torch.cuda.amp import autocast
from tqdm import tqdm

class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.00001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_top_k_coordinates_with_masking(outputs, k, offset):
    batch_size, channels, height, width = outputs.shape
    outputs = outputs.clone()  # Clone to avoid modifying the original tensor

    # Store selected coordinates
    selected_coords = []

    for b in range(batch_size):
        current_output = outputs[b][0]
        coords = []

        for _ in range(k):
            # Find the max index in the tensor
            max_index = torch.argmax(current_output)
            max_y, max_x = max_index // width, max_index % width

            # Append the coordinate of the max element
            coords.append((max_y.item(), max_x.item()))

            # Set surrounding area to zero within the specified offset
            y_min = max(0, max_y - offset)
            y_max = min(height, max_y + offset + 1)
            x_min = max(0, max_x - offset)
            x_max = min(width, max_x + offset + 1)

            current_output[y_min:y_max, x_min:x_max] = 0  # Mask the region around the maximum

        selected_coords.append(coords)

    return selected_coords

def check_clicks_topk(outputs, labels, k, offset, threshold=0.0002):
    # Retrieve the top k coordinates with masking
    points = get_top_k_coordinates_with_masking(outputs, k, offset)
    batch_size = outputs.shape[0]
    
    # This will store the total number of correct detections per batch item
    correct_counts = torch.zeros(batch_size, dtype=torch.int)

    # Iterate over each item in the batch
    for b in range(batch_size):
        # Get the coordinates for the current batch item
        batch_points = points[b]
        
        # Convert list of tuples to a tensor for indexing
        if batch_points:
            coords = torch.tensor(batch_points, dtype=torch.long)
            y_coords = coords[:, 0]
            x_coords = coords[:, 1]
        
            # Extract the corresponding label values
            corresponding_label_values = labels[b, 0, y_coords, x_coords]

            # Determine correct detections based on the threshold
            correct = corresponding_label_values > threshold
            if correct.any():
                correct_counts[b] = 1
            else:
                correct_counts[b] = 0
            # correct_counts[b] = correct.sum()

    # Compute precision for the entire batch
    precision = correct_counts.float().sum() / (batch_size)
    return precision.item(), correct_counts

def check_clicks_topk_bbox(outputs, labels, k, offset, threshold=0.0002):
    # Retrieve the top k coordinates with masking
    points = get_top_k_coordinates_with_masking(outputs, k, offset)
    batch_size = outputs.shape[0]
    
    # This will store the total number of correct detections per batch item
    correct_counts = torch.zeros(batch_size, dtype=torch.int)

    # Iterate over each item in the batch
    for b in range(batch_size):
        # Get the coordinates for the current batch item
        batch_points = points[b]
        
        # Convert list of tuples to a tensor for indexing
        if batch_points:
            coords = torch.tensor(batch_points, dtype=torch.long)
            x_coords = (coords[:, 0]/outputs.shape[2]).to(outputs.device)
            y_coords = (coords[:, 1]/outputs.shape[3]).to(outputs.device)
        
            
            # if the point is inside the bbox, it is considered as correct
            if len(labels[b]) != 4:
                bbox = labels[b][0]
            else:
                bbox = labels[b]
            # print(bbox)
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
            inside_x = (x_coords >= x_min) & (x_coords <= x_max)
            inside_y = (y_coords >= y_min) & (y_coords <= y_max)
            inside_bbox = inside_x & inside_y
            if inside_bbox.any():
                correct_counts[b] = 1
            else:
                correct_counts[b] = 0

    # Compute precision for the entire batch
    precision = correct_counts.float().sum() / (batch_size)
    return precision.item(), correct_counts


    
def check_clicks_heatmap(outputs, labels, output_type='heatmap'):
    if output_type == 'point':
        scale = torch.tensor([512, 256]).to(outputs.device)

        # Scale the output tensor
        scaled_output = outputs * scale[None, :]  # Applying the scaling factors
        scaled_output = scaled_output.round()

        # Prepare min and max tensors for clamping
        min_vals = torch.tensor([0, 0], dtype=torch.uint8, device=outputs.device)
        max_vals = torch.tensor([511, 255], dtype=torch.uint8, device=outputs.device)

        # Clamp values to ensure they fall within the specified ranges
        clamped_output = torch.clamp(scaled_output, min=min_vals, max=max_vals)

        # Convert to integers
        max_points = clamped_output.int()  # Converts to 32-bit integers
    else:
        # Get the indices of the max points in the outputs
        max_indices = torch.argmax(outputs.view(outputs.shape[0], -1), dim=1)

        # Compute the coordinates from the indices
        max_points = torch.stack((max_indices // outputs.shape[3], max_indices % outputs.shape[3]), dim=1)

    # Get the corresponding values from the labels
    corresponding_label_values = labels[torch.arange(labels.shape[0]), 0, max_points[:, 0], max_points[:, 1]]

    # Since heatmap is build in the bbox, if the heatmap value is less than 0.0002, it is considered as 0
    correct = torch.sum(corresponding_label_values > 0.0002).item()

    # Compute the precision
    precision = correct / outputs.shape[0]

    return precision


# Assuming you are loading a pre-trained LAModel
def load_blockla_parameters(model, load_path, freeze=True):
    missing_keys, unexpected_keys = model.laModel.load_state_dict(torch.load(load_path), strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    if freeze:
        for param in model.laModel.parameters():
            param.requires_grad = False

# Assuming you are loading a 3d LAModel
def load_block3d_parameters(model, load_path, freeze=True):
    missing_keys, unexpected_keys = model.block3d.load_state_dict(torch.load(load_path), strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    if freeze:
        for param in model.block3d.parameters():
            param.requires_grad = False
            
# Assuming you are loading a 2d LAModel
def load_block2d_parameters(model, load_path, freeze=True):
    missing_keys, unexpected_keys = model.block2d.load_state_dict(torch.load(load_path), strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    if freeze:
        for param in model.block2d.parameters():
            param.requires_grad = False

def validate(model, data_loader, criterion, device, decoder_name):
    model.eval()
    validation_loss = 0.0
    heat_total_precision = 0.0

    with torch.no_grad(), autocast():
        for text, bound, mask, input2, bbox, heats in tqdm(data_loader, total=len(data_loader), desc="Validating"):
            text, bound, mask, input2, bbox, heats = (tensor.to(device) for tensor in [text, bound, mask, input2, bbox, heats])

            if decoder_name == "point":
                labels = bbox
            else:
                labels = heats

            outputs = model(text, bound, mask, input2)
            loss = criterion(outputs, labels)
            # heat_precision = check_clicks_heatmap(outputs, heats, decoder_name)
            heat_precision, _ = check_clicks_topk(outputs, heats, 1, 20, threshold=0.0002)
            # heat_precision, _ = check_clicks_topk_bbox(outputs, bbox, 1, 20, threshold=0.0002)

            validation_loss += loss.item()
            heat_total_precision += heat_precision

    validation_loss /= len(data_loader)
    heat_validation_precision = heat_total_precision / len(data_loader)
    return validation_loss, heat_validation_precision
