import torch
from .layers import ConvLayer, Conv3DLayer
import torch.nn as nn

class BlockLA(nn.Module):
    def __init__(self, layers=12, dropout_rate=0.1, vertical_bins=200, horizontal_bins=96):
        super(BlockLA, self).__init__()

        # BERT model for OCR Text
        self.proj = nn.Linear(128, 512)

        # Spatial embedding
        self.spatial_embedding = nn.Embedding(vertical_bins * horizontal_bins, 512)

        # More complex Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dropout=dropout_rate, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # Upsampling layers
        self.upsample1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)  # from (2, 2) to (4, 4)
        self.upsample2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # from (4, 4) to (8, 8)
        self.final_upsample = nn.ConvTranspose2d(256, 256, kernel_size=(1, 2), stride=(1, 2))  # adjust to (8, 16)

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
        # asser if any of the id is greater than the embedding size
        
        # Get embeddings
        # Correct clamping
        # convert to long
        tl_id, tr_id, bl_id, br_id = tl_id.long(), tr_id.long(), bl_id.long(), br_id.long()

        max_id = self.vertical_bins * self.horizontal_bins - 1
        tl_id.clamp_(0, max_id)
        tr_id.clamp_(0, max_id)
        bl_id.clamp_(0, max_id)
        br_id.clamp_(0, max_id)

        tl_embed = self.spatial_embedding(tl_id)
        tr_embed = self.spatial_embedding(tr_id)
        bl_embed = self.spatial_embedding(bl_id)
        br_embed = self.spatial_embedding(br_id)
        
        # Combine embeddings
        # combined = bert_embed + icon_embed + tl_embed + tr_embed + bl_embed + br_embed

        # Combine embeddings with torch.sum
        combined = torch.sum(torch.stack([bert_embed, tl_embed, tr_embed, bl_embed, br_embed]), dim=0)

        return combined

    def forward(self, text, bound, mask):
        combined_embed = self.embed_step(text, bound)
        batch_size = text.size(0)

        empty_input = torch.zeros(batch_size, 4, 512, dtype=torch.float16).to(text.device)
        
        combined_embed = torch.cat((empty_input, combined_embed), dim=1)
        
        combined_attention_mask = torch.cat((torch.ones(batch_size, 4, dtype=torch.bool, device=combined_embed.device), mask), dim=1)
        combined_attention_mask = combined_attention_mask.bool()
        
        # Transformer processing
        transformer_out = self.transformer(combined_embed, src_key_padding_mask=~combined_attention_mask)

        # Select only the outputs corresponding to the empty inputs and reshape
        empty_outputs = transformer_out[:4].permute(1, 2, 0).view(batch_size, 512, 2, 2)  # Reshape to (batch, 512, 2, 2)

        # Sequential upsampling
        upsampled = self.upsample1(empty_outputs)
        upsampled = self.upsample2(upsampled)
        final_output = self.final_upsample(upsampled).view(batch_size, 256, 8, 16)

        return final_output

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
        return x10, x5, x3

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