import unittest

import numpy as np
import torch
from torch import nn

from src.anomaly_metrics import (
    _binary_auroc,
    _histogram_auroc,
    _histogram_aupro,
    evaluate_model,
)


class _IdentityExtractor(nn.Module):
    def forward(self, images):
        return images


class _ZeroDecoder(nn.Module):
    def forward(self, features):
        return torch.zeros_like(features)


def _batch_from_maps(maps, masks, labels, categories):
    return (
        maps.unsqueeze(1),
        masks.unsqueeze(1).float(),
        torch.tensor(labels, dtype=torch.int64),
        categories,
    )


class AnomalyMetricTests(unittest.TestCase):
    def test_exact_image_auroc_handles_perfect_order_and_ties(self):
        self.assertEqual(_binary_auroc([0.1, 0.9], [0, 1]), 1.0)
        self.assertEqual(_binary_auroc([0.5, 0.5], [0, 1]), 0.5)
        self.assertIsNone(_binary_auroc([0.1, 0.2], [0, 0]))

    def test_histogram_pixel_auroc_and_aupro_are_one_for_perfect_maps(self):
        positive = np.array([0, 0, 0, 10, 10], dtype=np.int64)
        negative = np.array([10, 10, 0, 0, 0], dtype=np.int64)
        region = np.array([0, 0, 0, 10, 10], dtype=np.int64)

        self.assertEqual(_histogram_auroc(positive, negative), 1.0)
        self.assertEqual(_histogram_aupro(negative, [region], 0.3), 1.0)

    def test_evaluate_model_perfect_maps_and_macro_category_metrics(self):
        maps = torch.zeros(4, 8, 8)
        masks = torch.zeros_like(maps)
        masks[2:, 2:4, 3:5] = 1
        maps[2:] = masks[2:] * 5
        loader = [
            _batch_from_maps(
                maps,
                masks,
                [0, 0, 1, 1],
                ["bottle", "bottle", "bottle", "bottle"],
            )
        ]

        result = evaluate_model(
            _IdentityExtractor(), _ZeroDecoder(), loader, collect_artifacts=True
        )

        category = result["per_category"]["bottle"]
        self.assertAlmostEqual(category["image_auroc"], 1.0)
        self.assertAlmostEqual(category["pixel_auroc"], 1.0)
        self.assertAlmostEqual(category["aupro"], 1.0)
        self.assertEqual(category["region_count"], 2)
        self.assertEqual(result["macro"]["image_auroc"], 1.0)
        self.assertEqual(tuple(result["artifacts"]["anomaly_maps"].shape), (4, 8, 8))
        self.assertEqual(result["artifacts"]["anomaly_maps"].dtype, np.float16)
        self.assertEqual(result["artifacts"]["masks"].dtype, np.uint8)

    def test_macro_metrics_are_unweighted_means_of_valid_categories(self):
        maps = torch.zeros(8, 8, 8)
        masks = torch.zeros_like(maps)
        masks[2:4, 2:4, 3:5] = 1
        maps[2:4] = masks[2:4] * 5
        masks[6:8, 2:4, 3:5] = 1
        maps[4:6] = 5
        categories = ["perfect"] * 4 + ["reversed"] * 4
        labels = [0, 0, 1, 1] * 2
        loader = [
            _batch_from_maps(maps, masks, labels, categories)
        ]

        result = evaluate_model(_IdentityExtractor(), _ZeroDecoder(), loader)

        self.assertAlmostEqual(result["per_category"]["perfect"]["image_auroc"], 1.0)
        self.assertAlmostEqual(result["per_category"]["reversed"]["image_auroc"], 0.0)
        self.assertAlmostEqual(result["macro"]["image_auroc"], 0.5)
        self.assertEqual(result["macro"]["image_auroc_categories"], 2)

    def test_random_maps_are_near_chance_and_ignore_single_class_auc(self):
        generator = torch.Generator().manual_seed(17)
        maps = torch.rand((80, 16, 16), generator=generator)
        masks = torch.zeros_like(maps)
        masks[40:, 2:6, 3:7] = 1
        loader = [
            _batch_from_maps(
                maps,
                masks,
                [0] * 40 + [1] * 40,
                ["grid"] * 80,
            )
        ]

        result = evaluate_model(_IdentityExtractor(), _ZeroDecoder(), loader)
        metrics = result["per_category"]["grid"]
        for metric in ("image_auroc", "pixel_auroc"):
            self.assertAlmostEqual(metrics[metric], 0.5, delta=0.15, msg=metric)
        # With PRO integrated only to FPR=0.3 and normalized by that limit,
        # a random diagonal curve has area 0.3 / 2.
        self.assertAlmostEqual(metrics["aupro"], 0.15, delta=0.05)

        single_class = evaluate_model(
            _IdentityExtractor(),
            _ZeroDecoder(),
            [_batch_from_maps(maps[:2], masks[:2], [0, 0], ["good", "good"])],
        )
        self.assertIsNone(single_class["per_category"]["good"]["image_auroc"])
        self.assertIsNone(single_class["per_category"]["good"]["pixel_auroc"])
        self.assertIsNone(single_class["per_category"]["good"]["aupro"])

    def test_histogram_auroc_is_chance_for_equal_distributions(self):
        positive = np.array([0, 5, 5, 0], dtype=np.int64)
        negative = np.array([0, 5, 5, 0], dtype=np.int64)
        self.assertAlmostEqual(_histogram_auroc(positive, negative), 0.5)


if __name__ == "__main__":
    unittest.main()
