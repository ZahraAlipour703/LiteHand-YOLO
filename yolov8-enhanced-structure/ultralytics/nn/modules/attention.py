# Ultralytics 🚀 AGPL-3.0 License
"""Attention modules used by LiteHand-YOLO."""

from __future__ import annotations

import torch
import torch.nn as nn


class ECA(nn.Module):
    """
    Efficient Channel Attention.

    The module preserves the number of channels. The parser passes
    both input and output channels, so c1 and c2 must match.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        k_size: int = 3,
    ):
        super().__init__()

        c2 = c1 if c2 is None else c2

        if c1 != c2:
            raise ValueError(
                f"ECA must preserve the channel count, but received "
                f"c1={c1}, c2={c2}."
            )

        if k_size is None:
            k_size = 3

        if k_size % 2 == 0:
            k_size += 1

        self.channels = c1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.

        Args:
            x: Tensor of shape (B, C, H, W).

        Returns:
            Attention-refined tensor of the same shape.
        """
        y = self.avg_pool(x)                 # B, C, 1, 1
        y = y.squeeze(-1).transpose(1, 2)    # B, 1, C
        y = self.conv(y)                     # B, 1, C
        y = y.transpose(1, 2).unsqueeze(-1)  # B, C, 1, 1
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class CoordAtt(nn.Module):
    """
    Lightweight Coordinate Attention.

    The module preserves the input channel count.
    """

    def __init__(
        self,
        c1: int,
        c2: int | None = None,
        reduction: int = 32,
    ):
        super().__init__()

        c2 = c1 if c2 is None else c2

        if c1 != c2:
            raise ValueError(
                f"CoordAtt must preserve the channel count, but received "
                f"c1={c1}, c2={c2}."
            )

        mid = max(8, c1 // reduction)

        self.conv1 = nn.Conv2d(
            c1,
            mid,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(
            mid,
            c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.conv_w = nn.Conv2d(
            mid,
            c2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply coordinate attention.

        Args:
            x: Tensor of shape (B, C, H, W).

        Returns:
            Attention-refined tensor of the same shape.
        """
        b, c, h, w = x.size()

        # Pool along width.
        x_h = x.mean(dim=-1, keepdim=True)  # B, C, H, 1

        # Pool along height.
        x_w = x.mean(dim=-2, keepdim=True).permute(0, 1, 3, 2)  # B, C, W, 1

        # Combine spatial descriptors.
        y = torch.cat([x_h, x_w], dim=2)

        y = self.conv1(y)
        y = self.bn(y)
        y = self.act(y)

        # Recover height and width branches.
        x_h, x_w = torch.split(y, [h, w], dim=2)

        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return x * a_h * a_w