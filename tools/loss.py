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

        return loss  

class BBoxLoss(nn.Module):
    def __init__(self, inside_penalty_weight=0.1):
        """
        Initializes the CustomLoss instance.
        
        Parameters:
            inside_penalty_weight (float): The weighting factor for the penalty
                                           for points inside the bounding box but away from the center.
        """
        super(BBoxLoss, self).__init__()
        self.inside_penalty_weight = inside_penalty_weight

    def forward(self, y_pred, y_true):
        """
        Forward pass of the loss module.

        Parameters:
            y_pred (torch.Tensor): Predictions with shape (batch_size, 2), each entry is (pred_x, pred_y).
            y_true (torch.Tensor): Ground truth with shape (batch_size, 4), each entry is (x_min, y_min, x_max, y_max).
        
        Returns:
            torch.Tensor: Computed mean loss for the batch.
        """
        # Unpack coordinates
        pred_x, pred_y = y_pred[:, 0], y_pred[:, 1]
        x_min, y_min, x_max, y_max = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]

        # Check if points are inside the bbox
        inside_x = (pred_x >= x_min) & (pred_x <= x_max)
        inside_y = (pred_y >= y_min) & (pred_y <= y_max)
        inside_bbox = inside_x & inside_y

        # Calculate distance to the nearest bbox edge
        dist_x = torch.min(torch.abs(pred_x - x_min), torch.abs(pred_x - x_max))
        dist_y = torch.min(torch.abs(pred_y - y_min), torch.abs(pred_y - y_max))
        dist_to_edge = torch.where(inside_bbox, torch.zeros_like(dist_x), torch.sqrt(dist_x**2 + dist_y**2))

        # Calculate distance inside the box to the center of the box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        dist_to_center = torch.sqrt((pred_x - center_x)**2 + (pred_y - center_y)**2)
        loss_inside = dist_to_center * self.inside_penalty_weight  # Apply weight to the inside penalty

        # Combine loss: zero if inside (with a small penalty), distance to edge if outside
        loss = torch.where(inside_bbox, loss_inside, dist_to_edge)

        return loss.mean()