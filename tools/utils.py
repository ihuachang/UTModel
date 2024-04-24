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

def check_clicks(outputs, labels, output_type='heatmap'):
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
    count = labels.shape
    corresponding_label_values = labels[torch.arange(labels.shape[0]), 0, max_points[:, 0], max_points[:, 1]]

    # Count the number of t?imes the corresponding label value is greater than 0.95
    correct = torch.sum(corresponding_label_values > 0.0001).item()

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

def validate(model, data_loader, criterion, device, decoder_name):
    model.eval()
    validation_loss = 0.0
    total_precision = 0.0

    with torch.no_grad(), autocast():
        for text, bound, mask, input2, labels, heats in tqdm(data_loader, total=len(data_loader), desc="Validating"):
            text, bound, mask, input2, labels, heats = (tensor.to(device) for tensor in [text, bound, mask, input2, labels, heats])

            outputs = model(text, bound, mask, input2)
            loss = criterion(outputs, labels)
            if decoder_name == "point":
                labels = heats
            precision = check_clicks(outputs, labels, decoder_name)

            validation_loss += loss.item()
            total_precision += precision

    validation_loss /= len(data_loader)
    validation_precision = total_precision / len(data_loader)
    return validation_loss, validation_precision
