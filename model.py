import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import SegformerModel


class CrossTemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key   = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = (channels // 8) ** -0.5

    def forward(self, pre_feat, post_feat):
        B, C, H, W = pre_feat.size()

        diff = torch.abs(pre_feat - post_feat)

        proj_query = self.query(diff).view(B, -1, H * W).permute(0, 2, 1)
        proj_key   = self.key(pre_feat).view(B, -1, H * W)
        proj_query = proj_query * self.scale

        energy    = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value(pre_feat).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + post_feat


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class GLCrossNet(nn.Module):
    def __init__(self, backbone='swin_tiny_patch4_window7_224', num_classes=5):
        super().__init__()
        self.backbone_name = backbone
        self.is_hf_model   = '/' in backbone  # e.g. 'nvidia/mit-b2'

        if self.is_hf_model:
            # --- Hugging Face path (SegFormer / MiT) ---
            self.local_encoder  = SegformerModel.from_pretrained(backbone)
            self.global_encoder = SegformerModel.from_pretrained(backbone)

            # MiT-B2 hidden_sizes = [64, 128, 320, 512]
            # These are the 4 hierarchical stage channel dims, same ordering as timm
            ch = list(self.local_encoder.config.hidden_sizes)

        else:
            # --- Timm path (Swin, PVT-v2, ConvNeXt, etc.) ---
            encoder_kwargs = {'pretrained': True, 'features_only': True}
            if 'swin' in backbone:
                encoder_kwargs['img_size'] = 512

            self.local_encoder  = timm.create_model(backbone, **encoder_kwargs)
            self.global_encoder = timm.create_model(backbone, **encoder_kwargs)

            self.local_encoder.set_grad_checkpointing(enable=True)
            self.global_encoder.set_grad_checkpointing(enable=True)

            ch = self.local_encoder.feature_info.channels()

        # Decoder — identical regardless of backbone API
        self.norm        = nn.LayerNorm(ch[-1])
        self.fusion_conv = nn.Conv2d(ch[-1] * 2, ch[-1], 1)

        self.co_attention = CrossTemporalAttention(channels=ch[-1])

        self.skip1_reduce = nn.Conv2d(ch[-2] * 2, ch[-2], 1)
        self.skip2_reduce = nn.Conv2d(ch[-3] * 2, ch[-3], 1)
        self.skip3_reduce = nn.Conv2d(ch[-4] * 2, ch[-4], 1)

        self.dec1 = DecoderBlock(ch[-1], ch[-2], ch[-2])
        self.dec2 = DecoderBlock(ch[-2], ch[-3], ch[-3])
        self.dec3 = DecoderBlock(ch[-3], ch[-4], ch[-4])

        self.final_up = nn.ConvTranspose2d(ch[-4], ch[-4], 4, 4)

        self.mask_head = nn.Sequential(
            nn.Conv2d(ch[-4], ch[-4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[-4], num_classes, 1)
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(ch[-4], ch[-4], 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[-4], 1, 1)
        )

    def extract_features_hf(self, encoder, x):
        """
        HF SegformerEncoder stores each of the 4 stage outputs in hidden_states
        when output_hidden_states=True. The encoder source confirms it reshapes
        every stage to NCHW before storing, so no permute is needed here.

        We force return_dict=True to guarantee .hidden_states attribute access
        works regardless of the model's default config setting.

        Returns a list of 4 NCHW tensors:
          [stage0: B x 64  x H/4  x W/4,
           stage1: B x 128 x H/8  x W/8,
           stage2: B x 320 x H/16 x W/16,
           stage3: B x 512 x H/32 x W/32]
        """
        outputs = encoder(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )
        # hidden_states is a tuple of 4 NCHW tensors, one per encoder stage
        return list(outputs.hidden_states)

    def extract_features_timm(self, encoder, x):
        """
        Handles three timm output formats:
          - NHWC (Swin): 4D with last dim == channel count → permute
          - NLC  (some ViTs): 3D → reshape using stride info
          - NCHW (ConvNeXt, PVT-v2): pass through
        """
        feats    = encoder(x)
        channels = encoder.feature_info.channels()
        strides  = encoder.feature_info.reduction()

        processed = []
        for i, f in enumerate(feats):
            if f.dim() == 4:
                if f.shape[-1] == channels[i]:
                    # NHWC → NCHW (Swin)
                    processed.append(f.permute(0, 3, 1, 2).contiguous())
                else:
                    # Already NCHW
                    processed.append(f)
            elif f.dim() == 3:
                # NLC → NCHW
                B, L, C = f.shape
                H = x.shape[2] // strides[i]
                W = x.shape[3] // strides[i]
                processed.append(
                    f.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
                )
            else:
                raise ValueError(f"Unexpected feature tensor shape: {f.shape}")
        return processed

    def extract_features(self, encoder, x):
        if self.is_hf_model:
            return self.extract_features_hf(encoder, x)
        else:
            return self.extract_features_timm(encoder, x)

    def forward(self, pre, post, g_pre, g_post):
        l_pre_feats  = self.extract_features(self.local_encoder,  pre)
        l_post_feats = self.extract_features(self.local_encoder,  post)

        g_pre_feats  = self.extract_features(self.global_encoder, g_pre)[-1]
        g_post_feats = self.extract_features(self.global_encoder, g_post)[-1]

        g_pre_up  = F.interpolate(g_pre_feats,  size=l_pre_feats[-1].shape[-2:],
                                  mode='bilinear', align_corners=False)
        g_post_up = F.interpolate(g_post_feats, size=l_post_feats[-1].shape[-2:],
                                  mode='bilinear', align_corners=False)

        pre_fused  = self.fusion_conv(torch.cat([l_pre_feats[-1],  g_pre_up],  dim=1))
        post_fused = self.fusion_conv(torch.cat([l_post_feats[-1], g_post_up], dim=1))

        attn_out = self.co_attention(pre_fused, post_fused)
        x        = 0.5 * attn_out + post_fused

        skip1 = self.skip1_reduce(torch.cat([l_pre_feats[-2], l_post_feats[-2]], dim=1))
        skip2 = self.skip2_reduce(torch.cat([l_pre_feats[-3], l_post_feats[-3]], dim=1))
        skip3 = self.skip3_reduce(torch.cat([l_pre_feats[-4], l_post_feats[-4]], dim=1))

        x  = self.dec1(x,  skip1)
        x  = self.dec2(x,  skip2)
        x  = self.dec3(x,  skip3)
        up = self.final_up(x)

        mask_out = self.mask_head(up)
        edge_out = self.edge_head(up)

        mask_out = F.interpolate(mask_out, size=pre.shape[-2:], mode='bilinear', align_corners=False)
        edge_out = F.interpolate(edge_out, size=pre.shape[-2:], mode='bilinear', align_corners=False)

        return mask_out, edge_out