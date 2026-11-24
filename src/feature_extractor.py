"""Frozen, multiscale ImageNet feature extraction for MVTec images."""

import time
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class WideResNetFeatureExtractor(nn.Module):
    """Concatenate aligned layer1-layer3 features from a frozen Wide-ResNet-50-2.

    For 512x512 inputs, the output has shape (B, 1792, 32, 32): layer1,
    layer2, and layer3 contribute 256, 512, and 1024 channels, respectively.
    """

    feature_channels = 1792
    feature_layers = ("layer1", "layer2", "layer3")

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = (
            Wide_ResNet50_2_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.backbone = wide_resnet50_2(weights=weights)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "WideResNetFeatureExtractor":
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (batch, 3, height, width).")
        if images.shape[-2] < 32 or images.shape[-1] < 32:
            raise ValueError("image height and width must both be at least 32.")

        backbone = self.backbone
        features = backbone.conv1(images)
        features = backbone.bn1(features)
        features = backbone.relu(features)
        features = backbone.maxpool(features)

        layer1 = backbone.layer1(features)
        layer2 = backbone.layer2(layer1)
        layer3 = backbone.layer3(layer2)
        target_size = layer3.shape[-2:]

        aligned = (
            F.interpolate(layer1, size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(layer2, size=target_size, mode="bilinear", align_corners=False),
            layer3,
        )
        return torch.cat(aligned, dim=1)


def benchmark_feature_extractor(
    extractor: WideResNetFeatureExtractor,
    input_shape: Tuple[int, int, int, int] = (1, 3, 512, 512),
    device: Optional[Union[str, torch.device]] = None,
    warmup: int = 2,
    iterations: int = 10,
) -> Dict[str, object]:
    """Measure inference latency and CUDA peak memory for a synthetic batch."""
    if len(input_shape) != 4 or input_shape[1] != 3:
        raise ValueError("input_shape must be (batch, 3, height, width).")
    if input_shape[0] < 1 or min(input_shape[2:]) < 32:
        raise ValueError("batch must be non-empty and image sides at least 32.")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations at least 1.")

    selected_device = torch.device(
        device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    extractor = extractor.to(selected_device).eval()
    images = torch.randn(input_shape, device=selected_device)

    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)

    with torch.no_grad():
        for _ in range(warmup):
            extractor(images)
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)

        start = time.perf_counter()
        for _ in range(iterations):
            output = extractor(images)
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations

    peak_allocated_mb = None
    peak_reserved_mb = None
    if selected_device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(selected_device) / (
            1024 * 1024
        )
        peak_reserved_mb = torch.cuda.max_memory_reserved(selected_device) / (
            1024 * 1024
        )

    return {
        "device": str(selected_device),
        "batch_size": input_shape[0],
        "output_shape": tuple(output.shape),
        "latency_ms_per_batch": elapsed_ms,
        "peak_cuda_allocated_mb": peak_allocated_mb,
        "peak_cuda_reserved_mb": peak_reserved_mb,
    }
