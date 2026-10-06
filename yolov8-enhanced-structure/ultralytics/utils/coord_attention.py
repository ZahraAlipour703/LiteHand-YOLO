# # coord_attention.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CoordAtt(nn.Module):
#     """Coordinate Attention (lightweight).
#     Usage from YAML: CoordAtt, [in_channels]  (out channels assumed = in_channels)
#     """
#     def __init__(self, in_channels: int, out_channels: int = None, reduction: int = 32):
#         super().__init__()
#         out_channels = out_channels or in_channels
#         mid = max(8, in_channels // reduction)

#         # 1x1 conv to encode concatenated pooled features
#         self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn = nn.BatchNorm2d(mid)
#         self.act = nn.ReLU(inplace=True)
#         # projectors for height and width attention
#         self.conv_h = nn.Conv2d(mid, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#         self.conv_w = nn.Conv2d(mid, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

#     def forward(self, x):
#         # x: B,C,H,W
#         b, c, h, w = x.size()
#         # pooled along width -> shape (B, C, H, 1)
#         x_h = x.mean(dim=-1, keepdim=True)
#         # pooled along height -> shape (B, C, 1, W) then transpose to (B, C, W, 1)
#         x_w = x.mean(dim=-2, keepdim=True).permute(0, 1, 3, 2)

#         # concat along the spatial dimension -> (B, C, H+W, 1)
#         y = torch.cat([x_h, x_w], dim=2)

#         y = self.conv1(y)
#         y = self.bn(y)
#         y = self.act(y)

#         # split and restore spatial shapes
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)  # back to (B, C, 1, W)

#         a_h = torch.sigmoid(self.conv_h(x_h))
#         a_w = torch.sigmoid(self.conv_w(x_w))
#         return x * a_h * a_w

# Backward-compatible import.
from ultralytics.nn.modules.attention import CoordAtt

__all__ = ["CoordAtt"]