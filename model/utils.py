import torch
import torch.nn.functional as F

def softmax_overimage(x):
        input_reshaped = x.view(x.size(0), -1)  # shape: [Batch, 1, 256, 512] -> [Batch,256*512]

        # Apply softmax along the last dimension
        softmax_output = F.softmax(input_reshaped, dim=1)

        # Reshape back to the original shape
        softmax_output_reshaped = softmax_output.view(x.size())

        return softmax_output_reshaped

def softmax_overpoint(x):
        return x