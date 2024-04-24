import torch
import torch.nn.functional as F

def softmax_overimage(x):
        input_reshaped = x.reshape(x.size(0), -1)  # shape: [Batch, 1, 256, 512] -> [Batch,256*512]

        # Apply softmax along the last dimension
        softmax_output = F.softmax(input_reshaped, dim=1)

        # Reshape back to the original shape
        softmax_output_reshaped = softmax_output.view(x.size())

        return softmax_output_reshaped

def softmax_last_two_dims(x):
        # Save the original shape for restoring after softmax
        original_shape = x.shape

        # Reshape to combine the last two dimensions
        combined_shape = x.size(0), x.size(1), -1
        x_reshaped = x.view(combined_shape)

        # Apply softmax along the last dimension
        softmax_output = F.softmax(x_reshaped, dim=-1)

        # Restore to the original shape
        return softmax_output.view(original_shape)

def softmax_overpoint(x):
        return x