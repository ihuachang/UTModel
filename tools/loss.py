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

        inputs = inputs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        # normalize target
        
        # Add epsilon to inputs and subtract epsilon from 1 - inputs
        inputs = torch.clamp(inputs, self.epsilon, 1 - self.epsilon)

        log_inputs = torch.log(inputs)  # log(p)
        log_1_minus_inputs = torch.log1p(-inputs)
        
        # Calculate the two parts of your condition
        loss_1 = (1 - inputs) * log_inputs
        loss_2 = (1 - targets).pow(self.beta) * inputs.pow(self.alpha) * log_1_minus_inputs

        # Apply the conditions
        loss = torch.where(targets == 1, loss_1, loss_2)

        # Sum and average the loss
        loss = loss.mean()

        return -loss  # As per definition loss should be negative of calculated value

