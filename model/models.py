import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvLayer, Conv3DLayer
from .decoders import Decoder, Decoder_heatmap, Decoder_heatmap_nocut
from .blocks import Block2D, Block3D, BlockLA
from .utils import softmax_overimage, softmax_overpoint, softmax_last_two_dims
    
class UNet(nn.Module):
    def __init__(self, decoder_type='heatmap'):
        super(UNet, self).__init__()
        self.block2d = Block2D()
        self.block3d = Block3D()
        self.comblayer = ConvLayer(512, 256, pool=False, upsample=False)
        self.decoder = Decoder_heatmap() if decoder_type == 'heatmap' else Decoder()
        self.softmax = softmax_overimage if decoder_type == 'heatmap' else softmax_overpoint

    def forward(self, text, bound, mask, x3d):
        x2d = torch.cat((x3d[:, :, 0], x3d[:, :, -1]), dim=1)
        _, x2d, _ = self.block2d(x2d)

        x3d, short_cut = self.block3d(x3d)
        x2d = softmax_last_two_dims(x2d)
        x3d = softmax_last_two_dims(x3d)
        x3d_squeeze = torch.squeeze(x3d, 2)
        x = torch.cat((x2d, x3d_squeeze), dim=1)
        x = self.comblayer(x)
        x = self.decoder(x, short_cut)
        x = self.softmax(x)
        return x

class UNet2D(nn.Module):
    def __init__(self, decoder_type='heatmap'):
        super(UNet2D, self).__init__()
        self.block2d = Block2D()
        self.decoder = Decoder()
        self.decoder_type = decoder_type
        
    def forward(self, text, bound, mask, x3d):
        x2d = torch.cat((x3d[:, :, 0], x3d[:, :, -1]), dim=1)
        x2d_output, x2d, _ = self.block2d(x2d)
        x2d = x2d_output if self.decoder_type == 'heatmap' else x2d
        x = softmax_overimage(x2d)
        return x if self.decoder_type == 'heatmap' else self.decoder(x, x2d_output)

class UNet3D(nn.Module):
    def __init__(self, decoder_type='heatmap'):
        super(UNet3D, self).__init__()
        self.block3d = Block3D()
        self.decoder = Decoder_heatmap() if decoder_type == 'heatmap' else Decoder()
        self.softmax = softmax_overimage if decoder_type == 'heatmap' else softmax_overpoint

    def forward(self, text, bound, mask, x3d):
        x3d, short_cut = self.block3d(x3d)
        x = softmax_overimage(x3d)
        x_squeeze = torch.squeeze(x, 2)
        x = self.decoder(x_squeeze, short_cut)
        return self.softmax(x)
    
class VLModel(nn.Module):
    def __init__(self, decoder_type='heatmap'):
        super(VLModel, self).__init__()
        self.laModel = BlockLA()  # Make sure BlockLA is defined correctly in blocks.py
        self.block3d = Block3D()
        self.comblayer = ConvLayer(512, 256, pool=False, upsample=False)
        self.decoder = Decoder_heatmap() if decoder_type == 'heatmap' else Decoder()
        self.softmax = softmax_overimage if decoder_type == 'heatmap' else softmax_overpoint

    def forward(self, text, bound, mask, x3d):
        x2d_output = self.laModel(text, bound, mask)
        x3d_output, short_cut = self.block3d(x3d)
        x2d_output = softmax_last_two_dims(x2d_output)
        x3d_output = softmax_last_two_dims(x3d_output)
        x_combined = torch.cat((x2d_output, torch.squeeze(x3d_output, 2)), dim=1)
        x_combined = self.comblayer(x_combined)
        decoded_output = self.decoder(x_combined, short_cut)
        return self.softmax(decoded_output)
    
class VL2DModel(nn.Module):
    def __init__(self, decoder_type='heatmap'):
        super(VL2DModel, self).__init__()
        self.laModel = BlockLA()  # Make sure BlockLA is defined
        self.block2d = Block2D()
        self.comblayer = ConvLayer(512, 256, pool=False, upsample=False)
        self.decoder = Decoder_heatmap() if decoder_type == 'heatmap' else Decoder()
        self.softmax = softmax_overimage if decoder_type == 'heatmap' else softmax_overpoint

    def forward(self, text, bound, mask, x3d):
        language_embed = self.laModel(text, bound, mask)
        x2d = torch.cat((x3d[:, :, 0], x3d[:, :, -1]), dim=1)
        _, x2d_output, shortcut = self.block2d(x2d)
        language_embed = softmax_overimage(language_embed)
        x2d_output = softmax_last_two_dims(x2d_output)
        x_combined = torch.cat((language_embed, torch.squeeze(x2d_output, 2)), dim=1)
        x_combined = self.comblayer(x_combined)
        decoded_output = self.decoder(x_combined, shortcut)
        return self.softmax(decoded_output)

class LModel(nn.Module):
    def __init__(self, decoder_type='heatmap'):
        super(LModel, self).__init__()
        self.laModel = BlockLA()  # Make sure BlockLA is defined
        self.decoder = Decoder_heatmap_nocut() if decoder_type == 'heatmap' else Decoder()
        self.softmax = softmax_overimage if decoder_type == 'heatmap' else softmax_overpoint

    def forward(self, text, bound, mask, x3d=None):
        language_embed = self.laModel(text, bound, mask)
        language_embed = softmax_last_two_dims(language_embed)
        decoded_output = self.decoder(language_embed, None)
        return self.softmax(decoded_output)