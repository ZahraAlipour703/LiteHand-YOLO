# # eca.py
# import torch
# import torch.nn as nn

# class ECA(nn.Module):
#     """Efficient Channel Attention (ECA) simple implementation.
#     Usage from YAML: ECA, [channels]  OR ECA, [channels, k_size]
#     """
#     def __init__(self, channels: int, k_size: int = 3):
#         super().__init__()
#         self.channels = channels
#         # keep kernel odd
#         if k_size is None:
#             k_size = 3
#         k_size = k_size if k_size % 2 == 1 else k_size + 1
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # conv1d across channels
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # x: (B, C, H, W)
#         y = self.avg_pool(x)                 # B, C, 1, 1
#         y = y.squeeze(-1).transpose(1, 2)    # B, 1, C
#         y = self.conv(y)                     # B, 1, C
#         y = y.transpose(1, 2).unsqueeze(-1)  # B, C, 1, 1
#         y = self.sigmoid(y)
#         return x * y.expand_as(x)

# Backward-compatible import.
from ultralytics.nn.modules.attention import ECA

__all__ = ["ECA"]