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
        
        # Post dictates the Query (What happened here?)
        proj_query = self.query(post_feat).view(B, -1, W*H).permute(0, 2, 1)
        # Pre dictates Key/Value (What used to be here?)
        proj_key = self.key(pre_feat).view(B, -1, W*H)
        
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(pre_feat).view(B, -1, W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        # Residual connection
        return self.gamma * out + post_feat

class GLCrossNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Lightweight encoders for 8GB VRAM
        self.local_encoder = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        self.global_encoder = timm.create_model('efficientnet_b0', pretrained=True, features_only=True)
        
        # Local features channel size (EfficientNet-B0 outputs 320 at the final feature map)
        self.fusion_conv = nn.Conv2d(320 * 2, 320, 1)
        self.co_attention = CrossTemporalAttention(320)
        
        # Decoders
        self.upconv = nn.ConvTranspose2d(320, 64, kernel_size=4, stride=4)
        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, num_classes, kernel_size=8, stride=8)
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=8, stride=8)
        )

    def forward(self, pre, post, g_pre, g_post):
        # Local Stream
        l_pre_feats = self.local_encoder(pre)[-1]
        l_post_feats = self.local_encoder(post)[-1]
        
        # Global Stream
        g_pre_feats = self.global_encoder(g_pre)[-1]
        g_post_feats = self.global_encoder(g_post)[-1]
        
        # Fusion (Global context pooled and expanded to local spatial dimensions)
        g_pre_pool = F.adaptive_avg_pool2d(g_pre_feats, 1).expand_as(l_pre_feats)
        g_post_pool = F.adaptive_avg_pool2d(g_post_feats, 1).expand_as(l_post_feats)
        
        pre_fused = self.fusion_conv(torch.cat([l_pre_feats, g_pre_pool], dim=1))
        post_fused = self.fusion_conv(torch.cat([l_post_feats, g_post_pool], dim=1))
        
        # Cross-Temporal Attention
        attended_features = self.co_attention(pre_fused, post_fused)
        
        # Decode
        up = self.upconv(attended_features)
        mask_out = self.mask_head(up)
        edge_out = self.edge_head(up).squeeze(1)
        
        # Resize to original tile size just in case
        mask_out = F.interpolate(mask_out, size=pre.shape[-2:], mode='bilinear', align_corners=False)
        edge_out = F.interpolate(edge_out.unsqueeze(1), size=pre.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)
        
        return mask_out, edge_out