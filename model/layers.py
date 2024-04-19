import torch
import torch.nn as nn
import torch.nn.functional as F
    
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=True, upsample=False):
        super(ConvLayer, self).__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = pool
        if pool:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        if self.upsample:
            x = self.upsample_layer(x)
        return x

class Conv3DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=True, max_pool_kernel=2, max_pool_stride=2):
        super(Conv3DLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = pool
        if pool:
            self.maxpool = nn.MaxPool3d(kernel_size=max_pool_kernel, stride=max_pool_stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.maxpool(x)
        return x