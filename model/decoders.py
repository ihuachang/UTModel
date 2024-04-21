import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvLayer, Conv3DLayer

class Decoder_heatmap(nn.Module):
    def __init__(self):
        super(Decoder_heatmap, self).__init__()
        self.layer1 = ConvLayer(256, 256, pool=False, upsample=True)
        self.layer2 = ConvLayer(256, 128, pool=False, upsample=True)
        self.layer3 = ConvLayer(256, 128, pool=False, upsample=True)
        self.layer4 = ConvLayer(128, 64, pool=False, upsample=True)
        self.layer5 = ConvLayer(64, 1, pool=False, upsample=True)
    
    def forward(self, x, short_cut):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x2_shortcut_squeezed = torch.squeeze(short_cut, 2) # since shortcut dimension is [-1, 128, 1, 32, 64]
        x2_shortcut = torch.cat((x2_shortcut_squeezed, x2), dim=1)
        x3 = self.layer3(x2_shortcut)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x5
    
class Decoder_heatmap_nocut(nn.Module):
    def __init__(self):
        super(Decoder_heatmap_nocut, self).__init__()
        self.layer1 = ConvLayer(256, 256, pool=False, upsample=True)
        self.layer2 = ConvLayer(256, 128, pool=False, upsample=True)
        self.layer3 = ConvLayer(128, 128, pool=False, upsample=True)
        self.layer4 = ConvLayer(128, 64, pool=False, upsample=True)
        self.layer5 = ConvLayer(64, 1, pool=False, upsample=True)
    
    def forward(self, x, short_cut):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x5

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)  # Reduces size to (4, 8)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 4 * 8, 100)
        self.fc2 = nn.Linear(100, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, short_cut):
        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers with a ReLU activation between them
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x