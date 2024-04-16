import torch
import torch.nn as nn
import torch.nn.functional as F

class LAModel(nn.Module):
    def __init__(self, layers=2, dropout_rate=0.1, vertical_bins=200, horizontal_bins=96):
        super(LAModel, self).__init__()

        # BERT model for OCR Text
        self.proj = nn.Linear(128, 512)

        # Spatial embedding
        self.spatial_embedding = nn.Embedding(vertical_bins * horizontal_bins, 512)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # Dummy embedding that will be treated as the model output
        self.empty_embedding = nn.Parameter(torch.zeros(512), requires_grad=True)

        # Upsample and expand channels
        self.upsample = nn.Linear(512, 32768)

        self.vertical_bins = vertical_bins
        self.horizontal_bins = horizontal_bins 

    def embed_step(self, ocr_emb, bboxes):
        # Reshape bboxes
        bboxes = bboxes.reshape(ocr_emb.shape[0], ocr_emb.shape[1], 4, 2)

        vertical_bins = self.vertical_bins
        horizontal_bins = self.horizontal_bins

        # BERT embedding
        bert_embed = self.proj(ocr_emb)

        # BBox embedding
        # Convert [batch, seq_len, 4, 2] and it's  top-left, top-right, bottom-right, and bottom-left

        # Bottom-left
        bl = bboxes[:, :, 3, :]
        # Top-left
        tl = bboxes[:, :, 0, :]
        # Bottom-right
        br = bboxes[:, :, 2, :]
        # Top-right
        tr = bboxes[:, :, 1, :]

        #convert to torch tensors
        tl = tl.clone().detach()
        tr = tr.clone().detach()
        bl = bl.clone().detach()
        br = br.clone().detach()

        # bin
        tl[:, :, 0] = (tl[:, :, 0] * (vertical_bins - 1)).long()
        tl[:, :, 1] = (tl[:, :, 1] * (horizontal_bins - 1)).long()
        tr[:, :, 0] = (tr[:, :, 0] * (vertical_bins - 1)).long()
        tr[:, :, 1] = (tr[:, :, 1] * (horizontal_bins - 1)).long()
        bl[:, :, 0] = (bl[:, :, 0] * (vertical_bins - 1)).long()
        bl[:, :, 1] = (bl[:, :, 1] * (horizontal_bins - 1)).long()
        br[:, :, 0] = (br[:, :, 0] * (vertical_bins - 1)).long()
        br[:, :, 1] = (br[:, :, 1] * (horizontal_bins - 1)).long()

        # Convert to single ID for embedding
        tl_id = tl[:, :, 0] * horizontal_bins + tl[:, :, 1]
        tr_id = tr[:, :, 0] * horizontal_bins + tr[:, :, 1]
        bl_id = bl[:, :, 0] * horizontal_bins + bl[:, :, 1]
        br_id = br[:, :, 0] * horizontal_bins + br[:, :, 1]
        
        # Get embeddings
        tl_embed = self.spatial_embedding(tl_id.long())
        tr_embed = self.spatial_embedding(tr_id.long())
        bl_embed = self.spatial_embedding(bl_id.long())
        br_embed = self.spatial_embedding(br_id.long())

        # Combine embeddings
        # combined = bert_embed + icon_embed + tl_embed + tr_embed + bl_embed + br_embed

        # Combine embeddings with torch.sum
        combined = torch.sum(torch.stack([bert_embed, tl_embed, tr_embed, bl_embed, br_embed]), dim=0)

        return combined

    def forward(self, text, bound, mask):

        combined_embed = self.embed_step(text, bound)
        # Concatenate the empty embedding to the start of the sequence
        batch_size = text.size(0)
        empty_input = self.empty_embedding.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)

        combined_embed = torch.cat((empty_input,
                                    combined_embed), dim=1)  # Adding empty embeddings at the start of the sequence
        
        # Create a combined attention mask, with no mask for the empty embedding
        combined_attention_mask = torch.cat((torch.ones(batch_size, 1, dtype=torch.bool, device=combined_embed.device), 
                                            mask), dim=1)
        
        combined_attention_mask = combined_attention_mask.bool()
        # Pass through transformer
        transformer_out = self.transformer(combined_embed.permute(1, 0, 2), src_key_padding_mask=~combined_attention_mask)
        # Extract the output corresponding to the empty embedding
        empty_embed_out = transformer_out[0]
        
        upsampled_output = self.upsample(empty_embed_out)
        upsampled_output = upsampled_output.view(-1, 256, 8, 16)

        return upsampled_output
    
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


class Block2D(nn.Module):
    def __init__(self):
        super(Block2D, self).__init__()
        self.layer1 = ConvLayer(6, 64, pool=True)
        self.layer2 = ConvLayer(64, 128, pool=True)
        self.layer3 = ConvLayer(128, 128, pool=True)
        self.layer4 = ConvLayer(128, 256, pool=True)
        self.layer5 = ConvLayer(256, 256, pool=True)
        self.layer6 = ConvLayer(256, 256, pool=False, upsample=True)
        self.layer7 = ConvLayer(256, 128, pool=False, upsample=True)
        self.layer8 = ConvLayer(256, 128, pool=False, upsample=True)
        self.layer9 = ConvLayer(128, 64, pool=False, upsample=True)
        self.layer10 = ConvLayer(64, 1, pool=False, upsample=True)

    def forward(self, x):
        # x = torch.cat((x1, x2), dim=1)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        x6 = self.layer6(x5)
        x7 = self.layer7(x6)
        x7_shortcut = torch.cat((x3, x7), dim=1)
        x8 = self.layer8(x7_shortcut)
        x9 = self.layer9(x8)
        x10 = self.layer10(x9)
        return x10, x5


class Block3D(nn.Module):
    def __init__(self):
        super(Block3D, self).__init__()
        self.layer1 = nn.Sequential(
            Conv3DLayer(3, 1, pool=False),
            Conv3DLayer(1, 16, pool=True, max_pool_kernel=(1,2,2), max_pool_stride=(1,2,2))
        )
        self.layer2 = Conv3DLayer(16, 64, pool=True, max_pool_kernel=(1,2,2), max_pool_stride=(1,2,2))
        self.layer3 = nn.Sequential(
            Conv3DLayer(64, 128, pool=False),
            Conv3DLayer(128, 128, pool=True, max_pool_kernel=(2,2,2), max_pool_stride=(2,2,2))
        )
        self.layer_shortcut = nn.MaxPool3d(kernel_size=(4,1,1), stride=(4,1,1))

        self.layer4 = nn.Sequential(
            Conv3DLayer(128, 256, pool=False),
            Conv3DLayer(256, 256, pool=True, max_pool_kernel=(2,2,2), max_pool_stride=(2,2,2))
        )
        self.layer5 = nn.Sequential(
            Conv3DLayer(256, 256, pool=False),
            Conv3DLayer(256, 256, pool=True, max_pool_kernel=(2,2,2), max_pool_stride=(2,2,2))
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        short_cut = self.layer_shortcut(x3)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        
        return x5, short_cut
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
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

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.block2d = Block2D()
        self.block3d = Block3D()
        self.comblayer = ConvLayer(512, 256, pool=False, upsample=False)
        self.decoder = Decoder()
    
    def softmax_overimage(self, x):
        input_reshaped = x.view(x.size(0), -1)  # shape: [Batch, 1, 256, 512] -> [Batch,256*512]

        # Apply softmax along the last dimension
        softmax_output = F.softmax(input_reshaped, dim=1)

        # Reshape back to the original shape
        softmax_output_reshaped = softmax_output.view(x.size())

        return softmax_output_reshaped
    
    def forward(self, x2d, x3d):
        # get block2d downsampled feature maps (block2d returns 2 feature maps)
        _, x2d = self.block2d(x2d)

        # get block3d downsampled feature maps and shortcut (block3d returns 2 feature maps)
        x3d, short_cut = self.block3d(x3d)

        # concate the 2d and 3d feature maps
        x2d = self.softmax_overimage(x2d)
        x3d_squeeze = torch.squeeze(x3d, 2)
        x = torch.cat((x2d, x3d_squeeze), dim=1)
        x = self.comblayer(x)

        # send combined feature maps to decoder
        x = self.decoder(x, short_cut)

        # softmax
        x = self.softmax_overimage(x)

        return x

class UNet2D(nn.Module):
    def __init__(self):
        super(UNet2D, self).__init__()
        self.block2d = Block2D()
    
    def softmax_overimage(self, x):
        input_reshaped = x.view(x.size(0), -1)  # shape: [Batch, 1, 256, 512] -> [Batch,256*512]

        # Apply softmax along the last dimension
        softmax_output = F.softmax(input_reshaped, dim=1)

        # Reshape back to the original shape
        softmax_output_reshaped = softmax_output.view(x.size())

        return softmax_output_reshaped
    
    def forward(self, x2d):
        # get block2d downsampled feature maps (block2d returns 2 feature maps)
        x2doutput, x2d = self.block2d(x2d)

        # softmax
        x = self.softmax_overimage(x2doutput)

        return x
    
class VLModel(nn.Module):
    def __init__(self):
        super(VLModel, self).__init__()
        self.laModel = LAModel()  # adjust parameters accordingly
        self.block3d = Block3D()  # Block3D from the UNet model in Model 1
        self.comblayer = ConvLayer(512, 256, pool=False, upsample=False)  # Combining layer from UNet
        self.decoder = Decoder()  # Decoder from UNet

    def softmax_overimage(self, x):
        input_reshaped = x.view(x.size(0), -1)
        softmax_output = F.softmax(input_reshaped, dim=1)
        softmax_output_reshaped = softmax_output.view(x.size())
        return softmax_output_reshaped
    
    def forward(self, text, bound, mask, x3d):
        # Process x2d through LAModel
        x2d_output = self.laModel(text, bound, mask)
        x2d_output = self.softmax_overimage(x2d_output)
        # Process x3d through Block3D
        x3d_output, short_cut = self.block3d(x3d)
        x3d_squeeze = torch.squeeze(x3d_output, 2)
        # Concatenate outputs from LAModel and Block3D
        x_combined = torch.cat((x2d_output, x3d_squeeze), dim=1)
        x_combined = self.comblayer(x_combined)
        
        # Process combined features through Decoder
        decoded_output = self.decoder(x_combined, short_cut)
        
        # Softmax applied to the final output
        final_output = self.softmax_overimage(decoded_output)
        return final_output