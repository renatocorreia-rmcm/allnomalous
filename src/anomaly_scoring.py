"""Batch-vectorized aggregation of pixel anomaly maps into image scores."""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _as_batched_maps(anomaly_maps: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if anomaly_maps.ndim == 3:
        return anomaly_maps.unsqueeze(1), True
    if anomaly_maps.ndim == 4 and anomaly_maps.shape[1] == 1:
        return anomaly_maps, False
    raise ValueError("anomaly_maps must have shape (B, H, W) or (B, 1, H, W).")


def gaussian_smooth(
    anomaly_maps: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply a channel-wise 2D Gaussian filter, preserving map dimensions."""
    maps, squeezed = _as_batched_maps(anomaly_maps)
    if not maps.is_floating_point():
        raise TypeError("anomaly_maps must have a floating-point dtype.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if kernel_size == 1:
        return anomaly_maps

    coordinates = torch.arange(
        kernel_size, device=maps.device, dtype=maps.dtype
    ) - kernel_size // 2
    one_dimensional = torch.exp(-(coordinates**2) / (2 * sigma**2))
    one_dimensional = one_dimensional / one_dimensional.sum()
    kernel = torch.outer(one_dimensional, one_dimensional)
    kernel = kernel / kernel.sum()
    kernel = kernel.reshape(1, 1, kernel_size, kernel_size)

    radius = kernel_size // 2
    padded = F.pad(maps, (radius, radius, radius, radius), mode="replicate")
    smoothed = F.conv2d(padded, kernel)
    return smoothed.squeeze(1) if squeezed else smoothed


def aggregate_anomaly_maps(
    anomaly_maps: torch.Tensor,
    method: str = "topk_mean",
    top_fraction: float = 0.01,
    smoothing_kernel_size: int = 5,
    smoothing_sigma: float = 1.0,
) -> torch.Tensor:
    """Return one score per map using top-k mean or a smoothed maximum.

    ``topk_mean`` averages ceil(top_fraction * H * W) largest pixels (at
    least one). ``max`` first applies Gaussian smoothing, then takes the
    largest value. A (B,H,W) or (B,1,H,W) input produces a (B,) result.
    """
    maps, _ = _as_batched_maps(anomaly_maps)
    if not maps.is_floating_point():
        raise TypeError("anomaly_maps must have a floating-point dtype.")
    if maps.shape[0] == 0 or min(maps.shape[-2:]) < 1:
        raise ValueError("anomaly maps must have a non-empty batch and spatial size.")

    flattened = maps.flatten(start_dim=1)
    if method == "topk_mean":
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1].")
        count = max(1, math.ceil(flattened.shape[1] * top_fraction))
        return torch.topk(flattened, k=count, dim=1).values.mean(dim=1)
    if method == "max":
        smoothed = gaussian_smooth(
            maps,
            kernel_size=smoothing_kernel_size,
            sigma=smoothing_sigma,
        )
        return smoothed.flatten(start_dim=1).amax(dim=1)
    raise ValueError("method must be 'topk_mean' or 'max'.")
