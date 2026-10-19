"""CNN classifier trained from scratch on competition audio only.

Rules note (NPC DSC 2026): this network is randomly initialised and learns
only from the provided training clips. No external data, no pretrained
weights -- so it is unambiguously compliant with "hanya data panitia".

The backbone is a compact EfficientNet-style separable-convolution network.
It is small enough to train on a single Kaggle GPU in well under the time
limit, and because it is from scratch its behaviour is fully explainable
(every learned weight comes from the competition data).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fnn


def _swish(x):
    return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gating."""

    def __init__(self, ch, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(1, ch // r), 1)
        self.fc2 = nn.Conv2d(max(1, ch // r), ch, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = _swish(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MBConv(nn.Module):
    """Mobile inverted bottleneck with depthwise conv + SE (MobileNetV2/EffNet)."""

    def __init__(self, cin, cout, stride=1, expand=4, k=3, drop=0.0):
        super().__init__()
        mid = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.expand = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid)) if expand != 1 else None
        self.dw = nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid)
        self.se = SEBlock(mid)
        self.pw = nn.Conv2d(mid, cout, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout2d(drop) if drop > 0 else None

    def forward(self, x):
        h = x
        if self.expand is not None:
            h = _swish(self.expand(h))
        h = _swish(self.bn_dw(self.dw(h)))
        h = self.se(h)
        h = self.bn_pw(self.pw(h))
        if self.drop is not None:
            h = self.drop(h)
        if self.use_res:
            h = h + x
        return h


class BirdCNN(nn.Module):
    """Small EfficientNet-style classifier for 1-channel log-mel/PCEN images.

    Input : (B, 1, n_mels, frames)
    Output: (B, n_classes) logits
    """

    def __init__(self, n_classes=3, width=(16, 24, 40, 80, 112), drop=0.1, in_ch=1):
        super().__init__()
        c0 = width[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c0, 3, 2, 1, bias=False), nn.BatchNorm2d(c0))
        # (channels_out, stride, expand, kernel, n_repeat)
        stages = [
            (width[1], 1, 1, 3, 1),
            (width[2], 2, 4, 3, 2),
            (width[3], 2, 4, 5, 2),
            (width[4], 2, 6, 5, 3),
        ]
        blocks, cin = [], c0
        for cout, stride, expand, k, rep in stages:
            for i in range(rep):
                blocks.append(MBConv(cin, cout, stride if i == 0 else 1, expand, k, drop))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head_conv = nn.Sequential(
            nn.Conv2d(cin, cin * 2, 1, bias=False), nn.BatchNorm2d(cin * 2))
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(cin * 2, n_classes)
        self._embed_dim = cin * 2

    @property
    def embed_dim(self):
        return self._embed_dim

    def features(self, x):
        x = _swish(self.stem(x))
        x = self.blocks(x)
        x = _swish(self.head_conv(x))
        return x.mean((2, 3))               # global average pool -> (B, embed_dim)

    def forward(self, x, return_embed=False):
        emb = self.features(x)
        logits = self.fc(self.dropout(emb))
        if return_embed:
            return logits, emb
        return logits


def build_model(n_classes=3, size="s", drop=0.1):
    widths = {
        "xs": (12, 16, 24, 48, 64),
        "s": (16, 24, 40, 80, 112),
        "m": (24, 32, 56, 112, 160),
    }[size]
    return BirdCNN(n_classes=n_classes, width=widths, drop=drop)
