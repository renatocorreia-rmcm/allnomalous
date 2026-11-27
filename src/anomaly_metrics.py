"""Memory-bounded anomaly detection metrics and model evaluation."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from scipy import ndimage
from scipy.integrate import trapezoid

from src.anomaly_maps import compute_anomaly_maps
from src.anomaly_scoring import aggregate_anomaly_maps


def _binary_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Compute exact binary AUROC using average ranks for tied scores."""
    score_array = np.asarray(scores, dtype=np.float64)
    label_array = np.asarray(labels)
    if not np.isfinite(score_array).all():
        raise ValueError("image scores must be finite.")
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("image labels must be binary (0 or 1).")
    label_array = label_array.astype(np.uint8, copy=False)
    positives = int(label_array.sum())
    negatives = len(label_array) - positives
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(score_array, kind="mergesort")
    sorted_scores = score_array[order]
    ranks = np.empty(len(score_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positive_rank_sum = ranks[label_array == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def _histogram_auroc(
    positive_histogram: np.ndarray, negative_histogram: np.ndarray
) -> Optional[float]:
    """Approximate AUROC from fixed-size, monotonically binned score counts."""
    positives = int(positive_histogram.sum())
    negatives = int(negative_histogram.sum())
    if positives == 0 or negatives == 0:
        return None

    true_positive_rate = np.cumsum(positive_histogram[::-1], dtype=np.float64) / positives
    false_positive_rate = np.cumsum(negative_histogram[::-1], dtype=np.float64) / negatives
    tpr = np.concatenate(([0.0], true_positive_rate))
    fpr = np.concatenate(([0.0], false_positive_rate))
    return float(trapezoid(tpr, fpr))


def _histogram_aupro(
    negative_histogram: np.ndarray,
    region_histograms: Sequence[np.ndarray],
    max_fpr: float,
) -> Optional[float]:
    """Approximate PRO AUC up to max_fpr using per-region score histograms."""
    negatives = int(negative_histogram.sum())
    if negatives == 0 or not region_histograms:
        return None

    false_positive_rate = (
        np.cumsum(negative_histogram[::-1], dtype=np.float64) / negatives
    )
    pro = np.zeros(len(negative_histogram), dtype=np.float64)
    for histogram in region_histograms:
        region_size = int(histogram.sum())
        if region_size:
            pro += np.cumsum(histogram[::-1], dtype=np.float64) / region_size
    pro /= len(region_histograms)

    fpr = np.concatenate(([0.0], false_positive_rate))
    pro = np.concatenate(([0.0], pro))
    crossing = np.flatnonzero(fpr >= max_fpr)
    if crossing.size:
        end = int(crossing[0])
        clipped_pro = float(np.interp(max_fpr, fpr[: end + 1], pro[: end + 1]))
        fpr = np.concatenate((fpr[:end], [max_fpr]))
        pro = np.concatenate((pro[:end], [clipped_pro]))
    else:
        fpr = np.concatenate((fpr, [max_fpr]))
        pro = np.concatenate((pro, [pro[-1]]))

    area = float(trapezoid(pro, fpr))
    return min(1.0, max(0.0, area / max_fpr))


class _CategoryAccumulator:
    def __init__(self, histogram_bins: int) -> None:
        self.image_scores: List[float] = []
        self.image_labels: List[int] = []
        self.positive_histogram = np.zeros(histogram_bins, dtype=np.int64)
        self.negative_histogram = np.zeros(histogram_bins, dtype=np.int64)
        self.region_histograms: List[np.ndarray] = []
        self.pixel_count = 0

    def add(
        self,
        image_score: float,
        image_label: int,
        anomaly_map: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if image_label not in (0, 1):
            raise ValueError("image labels must be binary (0 or 1).")
        score_map = np.asarray(anomaly_map, dtype=np.float32)
        binary_mask = np.asarray(mask, dtype=bool)
        if score_map.shape != binary_mask.shape:
            raise ValueError("anomaly maps and masks must have identical spatial sizes.")
        if not np.isfinite(score_map).all() or np.any(score_map < 0):
            raise ValueError("anomaly maps must contain finite, non-negative scores.")

        self.image_scores.append(float(image_score))
        self.image_labels.append(int(image_label))
        self.pixel_count += score_map.size

        # A monotonic transform maps unbounded distances to a fixed histogram range.
        bounded_scores = score_map / (score_map + 1.0)
        bin_indices = np.minimum(
            (bounded_scores * len(self.positive_histogram)).astype(np.int64),
            len(self.positive_histogram) - 1,
        )
        self.positive_histogram += np.bincount(
            bin_indices[binary_mask], minlength=len(self.positive_histogram)
        )
        self.negative_histogram += np.bincount(
            bin_indices[~binary_mask], minlength=len(self.negative_histogram)
        )

        connected_regions, region_count = ndimage.label(
            binary_mask, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )
        for region_id in range(1, region_count + 1):
            region_bins = bin_indices[connected_regions == region_id]
            self.region_histograms.append(
                np.bincount(region_bins, minlength=len(self.positive_histogram))
            )

    def metrics(self, max_fpr: float) -> Dict[str, Optional[float]]:
        return {
            "image_auroc": _binary_auroc(self.image_scores, self.image_labels),
            "pixel_auroc": _histogram_auroc(
                self.positive_histogram, self.negative_histogram
            ),
            "aupro": _histogram_aupro(
                self.negative_histogram, self.region_histograms, max_fpr
            ),
        }


def evaluate_model(
    extractor: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: Iterable,
    device: Optional[torch.device] = None,
    score_method: str = "topk_mean",
    top_fraction: float = 0.01,
    histogram_bins: int = 2048,
    aupro_max_fpr: float = 0.3,
    map_smoothing_kernel_size: Optional[int] = None,
    collect_artifacts: bool = False,
) -> Dict[str, object]:
    """Evaluate an extractor/decoder on a labeled loader, grouped by category.

    Pixel AUROC and AUPRO use bounded-size score histograms; image AUROC uses
    exact rank statistics. The input loader must yield
    ``(images, masks, labels, categories)`` as ``MVTecTestDataset`` does.
    """
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be at least 2.")
    if not 0 < aupro_max_fpr <= 1:
        raise ValueError("aupro_max_fpr must be in (0, 1].")
    if score_method not in ("topk_mean", "max"):
        raise ValueError("score_method must be 'topk_mean' or 'max'.")
    if device is None:
        first_parameter = next(extractor.parameters(), None)
        device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    device = torch.device(device)

    accumulators: Dict[str, _CategoryAccumulator] = defaultdict(
        lambda: _CategoryAccumulator(histogram_bins)
    )
    artifact_batches: Dict[str, List[np.ndarray]] = {
        "image_scores": [],
        "image_labels": [],
        "categories": [],
        "anomaly_maps": [],
        "masks": [],
    }
    extractor.eval()
    decoder.eval()

    with torch.inference_mode():
        for batch in loader:
            if len(batch) != 4:
                raise ValueError("loader batches must be (images, masks, labels, categories).")
            images, masks, labels, categories = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).reshape(-1)
            if masks.ndim == 4:
                if masks.shape[1] != 1:
                    raise ValueError("masks must have shape (B, H, W) or (B, 1, H, W).")
                masks = masks[:, 0]
            elif masks.ndim != 3:
                raise ValueError("masks must have shape (B, H, W) or (B, 1, H, W).")
            if masks.shape[0] != images.shape[0] or labels.numel() != images.shape[0]:
                raise ValueError("images, masks, and labels must have matching batch sizes.")
            original_features = extractor(images)
            reconstructed_features = decoder(original_features)
            anomaly_maps = compute_anomaly_maps(
                original_features,
                reconstructed_features,
                output_size=tuple(masks.shape[-2:]),
                smoothing_kernel_size=map_smoothing_kernel_size,
            )
            image_scores = aggregate_anomaly_maps(
                anomaly_maps, method=score_method, top_fraction=top_fraction
            )

            if len(categories) != images.shape[0]:
                raise ValueError("one category name is required for each image.")

            scores_cpu = image_scores.float().cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            if not np.isin(labels_cpu, (0, 1)).all():
                raise ValueError("image labels must be binary (0 or 1).")
            maps_cpu = anomaly_maps.float().cpu().numpy()
            masks_cpu = masks.cpu().numpy()
            if collect_artifacts:
                artifact_batches["image_scores"].append(scores_cpu.astype(np.float32))
                artifact_batches["image_labels"].append(labels_cpu.astype(np.uint8))
                artifact_batches["categories"].append(
                    np.asarray([str(category) for category in categories], dtype=np.str_)
                )
                artifact_batches["anomaly_maps"].append(maps_cpu.copy())
                artifact_batches["masks"].append((masks_cpu > 0).astype(np.uint8))
            for index, category in enumerate(categories):
                accumulators[str(category)].add(
                    scores_cpu[index],
                    labels_cpu[index],
                    maps_cpu[index],
                    masks_cpu[index] > 0,
                )

    per_category = {
        category: {
            **accumulator.metrics(aupro_max_fpr),
            "image_count": len(accumulator.image_labels),
            "pixel_count": accumulator.pixel_count,
            "region_count": len(accumulator.region_histograms),
        }
        for category, accumulator in sorted(accumulators.items())
    }
    macro = {}
    for metric_name in ("image_auroc", "pixel_auroc", "aupro"):
        values = [
            result[metric_name]
            for result in per_category.values()
            if result[metric_name] is not None
        ]
        macro[metric_name] = float(np.mean(values)) if values else None
        macro[f"{metric_name}_categories"] = len(values)

    result = {
        "per_category": per_category,
        "macro": macro,
        "settings": {
            "score_method": score_method,
            "top_fraction": top_fraction,
            "histogram_bins": histogram_bins,
            "aupro_max_fpr": aupro_max_fpr,
            "map_smoothing_kernel_size": map_smoothing_kernel_size,
        },
    }
    if collect_artifacts:
        result["artifacts"] = {
            key: np.concatenate(values, axis=0) if values else np.empty((0,))
            for key, values in artifact_batches.items()
        }
    return result
