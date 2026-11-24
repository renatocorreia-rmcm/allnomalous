"""Pixel-level anomaly maps from feature reconstruction discrepancies."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.anomaly_scoring import gaussian_smooth


def compute_anomaly_maps(
    original_features: torch.Tensor,
    reconstructed_features: torch.Tensor,
    output_size: Tuple[int, int],
    smoothing_kernel_size: Optional[int] = None,
    smoothing_sigma: float = 1.0,
) -> torch.Tensor:
    """Compute raw float32 L2 discrepancy maps and resize to image resolution.

    Inputs must have identical ``(B,C,h,w)`` shapes. The output has shape
    ``(B,H,W)`` and remains unnormalized for metrics. Smoothing is optional.
    """
    if original_features.shape != reconstructed_features.shape:
        raise ValueError(
            "feature tensors must have identical shapes; "
            f"got {tuple(original_features.shape)} and "
            f"{tuple(reconstructed_features.shape)}."
        )
    if original_features.ndim != 4:
        raise ValueError("features must have shape (batch, channels, height, width).")
    if not original_features.is_floating_point() or not reconstructed_features.is_floating_point():
        raise TypeError("feature tensors must have floating-point dtypes.")
    if original_features.shape[0] == 0 or min(original_features.shape[1:]) < 1:
        raise ValueError("feature batch, channels, and spatial dimensions must be non-empty.")
    if len(output_size) != 2 or any(
        not isinstance(side, int) or side <= 0 for side in output_size
    ):
        raise ValueError("output_size must contain two positive integers.")

    difference = original_features.float() - reconstructed_features.float()
    feature_map = torch.linalg.vector_norm(difference, ord=2, dim=1)
    anomaly_maps = F.interpolate(
        feature_map.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    if smoothing_kernel_size is not None:
        anomaly_maps = gaussian_smooth(
            anomaly_maps,
            kernel_size=smoothing_kernel_size,
            sigma=smoothing_sigma,
        )
    return anomaly_maps.to(dtype=torch.float32)


def normalize_map_for_visualization(anomaly_maps: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each map independently for visualization only."""
    squeezed = anomaly_maps.ndim == 3
    if squeezed:
        maps = anomaly_maps.unsqueeze(1)
    elif anomaly_maps.ndim == 4 and anomaly_maps.shape[1] == 1:
        maps = anomaly_maps
    else:
        raise ValueError("anomaly_maps must have shape (B, H, W) or (B, 1, H, W).")
    if not maps.is_floating_point():
        raise TypeError("anomaly_maps must have a floating-point dtype.")
    if maps.shape[0] == 0 or min(maps.shape[-2:]) < 1:
        raise ValueError("anomaly maps must have a non-empty batch and spatial size.")

    maps = maps.float()
    flattened = maps.flatten(start_dim=2)
    minimum = flattened.amin(dim=2).view(-1, 1, 1, 1)
    maximum = flattened.amax(dim=2).view(-1, 1, 1, 1)
    normalized = (maps - minimum) / (maximum - minimum).clamp_min(
        torch.finfo(maps.dtype).eps
    )
    return normalized.squeeze(1) if squeezed else normalized
