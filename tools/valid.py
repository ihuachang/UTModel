import torch

def check_clicks(outputs, labels, output_type='heatmap'):
    if output_type == 'point':
        # convert outputs to integer
        max_points = outputs.round().long()

    else:
        # Get the indices of the max points in the outputs
        max_indices = torch.argmax(outputs.view(outputs.shape[0], -1), dim=1)

        # Compute the coordinates from the indices
        max_points = torch.stack((max_indices // outputs.shape[3], max_indices % outputs.shape[3]), dim=1)

    # Get the corresponding values from the labels
    corresponding_label_values = labels[torch.arange(labels.shape[0]), 0, max_points[:, 0], max_points[:, 1]]

    # Count the number of t?imes the corresponding label value is greater than 0.95
    correct = torch.sum(corresponding_label_values > 0.0001).item()

    # Compute the precision
    precision = correct / outputs.shape[0]

    return precision

