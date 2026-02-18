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
        proj_query = self.query(post_feat).view(B, -1, W*H).permute(0, 2, 1)
        proj_key = self.key(pre_feat).view(B, -1, W*H)

        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value(pre_feat).view(B, -1, W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + post_feat

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
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
        # features_only=True returns intermediate layers for skip connections
        self.local_encoder = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        self.global_encoder = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)

        # EfficientNet-B0 feature channels: [16, 24, 40, 112, 320]
        self.fusion_conv = nn.Conv2d(320 * 2, 320, 1)
        self.co_attention = CrossTemporalAttention(320)

        # U-Net style Decoder with Skip Connections
        self.dec1 = DecoderBlock(in_channels=320, skip_channels=112, out_channels=112)
        self.dec2 = DecoderBlock(in_channels=112, skip_channels=40, out_channels=40)
        self.dec3 = DecoderBlock(in_channels=40, skip_channels=24, out_channels=24)

        # Final upsampling to original resolution (Stride 4 total from here)
        self.final_up = nn.ConvTranspose2d(24, 24, kernel_size=4, stride=4)

        self.mask_head = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, num_classes, 1)
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 1, 1)
        )

    def forward(self, pre, post, g_pre, g_post):
        # Local Stream (Extract list of multi-scale features)
        l_pre_feats = self.local_encoder(pre)
        l_post_feats = self.local_encoder(post)

        # Global Stream (Only need the deepest features)
        g_pre_feats = self.global_encoder(g_pre)[-1]
        g_post_feats = self.global_encoder(g_post)[-1]

        # Global-Local Fusion at the bottleneck
        g_pre_pool = F.adaptive_avg_pool2d(g_pre_feats, 1).expand_as(l_pre_feats[-1])
        g_post_pool = F.adaptive_avg_pool2d(g_post_feats, 1).expand_as(l_post_feats[-1])

        pre_fused = self.fusion_conv(torch.cat([l_pre_feats[-1], g_pre_pool], dim=1))
        post_fused = self.fusion_conv(torch.cat([l_post_feats[-1], g_post_pool], dim=1))

        # Cross-Temporal Attention at the bottleneck
        x = self.co_attention(pre_fused, post_fused)

        # Decoding with Skip Connections from the Post-Disaster Image
        x = self.dec1(x, l_post_feats[-2]) # 16x16 -> 32x32
        x = self.dec2(x, l_post_feats[-3]) # 32x32 -> 64x64
        x = self.dec3(x, l_post_feats[-4]) # 64x64 -> 128x128

        # Final Upsample and Head Projection
        up = self.final_up(x) # 128x128 -> 256x256

        mask_out = self.mask_head(up)
        edge_out = self.edge_head(up).squeeze(1)

        # Safety resize to guarantee exact output dimensions
        mask_out = F.interpolate(mask_out, size=pre.shape[-2:], mode='bilinear', align_corners=False)
        edge_out = F.interpolate(edge_out.unsqueeze(1), size=pre.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        return mask_out, edge_out