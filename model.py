
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SeparableConv2d(nn.Module):
    """
    Depthwise separable convolution to heavily reduce parameters and FLOPs.
    Upgraded to use Hardswish for better accuracy at negligible computational cost.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size,
                                   padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Hardswish(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))

class GlobalContext(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, in_ch, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        w = self.act(self.conv(self.pool(x)))
        return x * w

class MobileNetV3LightSeg(nn.Module):
    """
    Ultra-lightweight Semantic Segmentation Model for the PaDIS repository.
    Designed for strict end-to-end inference (outputs 300x300 integer masks natively).
    """
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()

        # 1. ENCODER
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained).features[:10]

        # Optimized Skip Indices for MobileNetV3-Small:
        self.out_indices = [3, 8, len(self.backbone)-1]

        # 2. GLOBAL CONTEXT (Matches last backbone output channels: 96)
        self.global_context = GlobalContext(96)
        self.reduce = nn.Conv2d(96, 64, kernel_size=1)

        # 3. ULTRA-LIGHT FUSION DECODER
        self.fuse_38 = SeparableConv2d(64 + 48, 32)
        self.fuse_75 = SeparableConv2d(32 + 24, 16)

        # 4. REGULARIZATION
        self.dropout = nn.Dropout2d(p=0.1)

        # 5. FINAL CLASSIFIER
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # --- ENCODER PASS ---
        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.out_indices:
                feats.append(x)

        f_75, f_38, f_10 = feats

        # --- DECODER PASS ---
        x = self.global_context(f_10)
        x = self.reduce(x)

        x = F.interpolate(x, size=f_38.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_38(torch.cat([x, f_38], dim=1))

        x = F.interpolate(x, size=f_75.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse_75(torch.cat([x, f_75], dim=1))

        # --- CLASSIFICATION & POST-PROCESSING ---
        x = self.dropout(x)
        logits_75 = self.head(x)
        logits = F.interpolate(logits_75, size=input_shape, mode='bilinear', align_corners=False)

        # --- STRICT END-TO-END COMPLIANCE ---
        # During training, return logits for CrossEntropyLoss.
        # During testing/inference, return the integer mask directly.
        if self.training:
            return logits
        else:
            return torch.argmax(logits, dim=1)