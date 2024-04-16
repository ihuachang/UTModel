import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, beta=2.0, epsilon=1e-5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        B, _, H, W = inputs.size()

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Add epsilon to inputs and subtract epsilon from 1 - inputs
        inputs = torch.clamp(inputs, self.epsilon, 1 - self.epsilon)
        log_inputs = torch.log(inputs)
        log_1_minus_inputs = torch.log(1 - inputs)
        
        # Calculate the two parts of your condition
        loss_1 = (1 - inputs) * log_inputs
        loss_2 = (1 - targets)**self.beta * inputs**self.alpha * log_1_minus_inputs

        # Apply the conditions
        loss = torch.where(targets == 1, loss_1, loss_2)

        # Sum and average the loss
        loss = torch.sum(loss)

        return -loss / (B * H * W)  # As per definition loss should be negative of calculated value

