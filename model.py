import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class CrossTemporalAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, pre_feat, post_feat):
        B, C, H, W = pre_feat.size()

        # INJECTION: Isolate the structural changes
        diff = torch.abs(pre_feat - post_feat)

        # Query uses the difference tensor to focus attention
        proj_query = self.query(diff).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(pre_feat).view(B, -1, H * W)

        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value(pre_feat).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + post_feat


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

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
    def __init__(self, num_classes=5):
        super().__init__()

        # Gradient checkpointing must be enabled via set_grad_checkpointing()
        # after model creation — passing it as a kwarg to create_model is not
        # supported in all timm versions and causes a TypeError.
        self.local_encoder = timm.create_model(
            'convnext_tiny', pretrained=True, features_only=True
        )
        self.global_encoder = timm.create_model(
            'convnext_tiny', pretrained=True, features_only=True
        )
        self.local_encoder.set_grad_checkpointing(enable=True)
        self.global_encoder.set_grad_checkpointing(enable=True)

        ch = self.local_encoder.feature_info.channels()

        self.norm = nn.LayerNorm(ch[-1])
        self.fusion_conv = nn.Conv2d(ch[-1] * 2, ch[-1], 1)

        # CrossTemporalAttention replacing the generic MultiheadAttention
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

    def forward(self, pre, post, g_pre, g_post):
        l_pre_feats = self.local_encoder(pre)
        l_post_feats = self.local_encoder(post)

        g_pre_feats = self.global_encoder(g_pre)[-1]
        g_post_feats = self.global_encoder(g_post)[-1]

        g_pre_up = F.interpolate(
            g_pre_feats, size=l_pre_feats[-1].shape[-2:], mode='bilinear', align_corners=False
        )
        g_post_up = F.interpolate(
            g_post_feats, size=l_post_feats[-1].shape[-2:], mode='bilinear', align_corners=False
        )

        pre_fused = self.fusion_conv(torch.cat([l_pre_feats[-1], g_pre_up], dim=1))
        post_fused = self.fusion_conv(torch.cat([l_post_feats[-1], g_post_up], dim=1))

        # Replaced MultiheadAttention with the dedicated difference-tensor attention
        attn_out = self.co_attention(pre_fused, post_fused)

        x = 0.5 * attn_out + post_fused

        skip1 = self.skip1_reduce(torch.cat([l_pre_feats[-2], l_post_feats[-2]], dim=1))
        skip2 = self.skip2_reduce(torch.cat([l_pre_feats[-3], l_post_feats[-3]], dim=1))
        skip3 = self.skip3_reduce(torch.cat([l_pre_feats[-4], l_post_feats[-4]], dim=1))

        x = self.dec1(x, skip1)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip3)

        up = self.final_up(x)

        mask_out = self.mask_head(up)
        edge_out = self.edge_head(up)

        mask_out = F.interpolate(mask_out, size=pre.shape[-2:], mode='bilinear', align_corners=False)
        edge_out = F.interpolate(edge_out, size=pre.shape[-2:], mode='bilinear', align_corners=False)

        return mask_out, edge_out