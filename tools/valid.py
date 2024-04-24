import torch

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

