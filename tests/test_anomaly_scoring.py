import unittest

import torch

from src.anomaly_scoring import aggregate_anomaly_maps, gaussian_smooth


class AnomalyScoringTests(unittest.TestCase):
    def test_topk_mean_is_vectorized_over_a_batch(self):
        maps = torch.stack(
            (
                torch.arange(100, dtype=torch.float32).reshape(1, 10, 10),
                torch.arange(100, 200, dtype=torch.float32).reshape(1, 10, 10),
            )
        )
        scores = aggregate_anomaly_maps(maps, method="topk_mean", top_fraction=0.1)
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.allclose(scores, torch.tensor([94.5, 194.5])))

    def test_topk_mean_uses_at_least_one_pixel_and_accepts_bhw(self):
        maps = torch.tensor([[[1.0, 8.0], [3.0, 5.0]]])
        score = aggregate_anomaly_maps(maps, top_fraction=0.001)
        self.assertEqual(tuple(score.shape), (1,))
        self.assertEqual(score.item(), 8.0)

    def test_maximum_uses_gaussian_smoothed_maps(self):
        maps = torch.zeros(2, 1, 9, 9)
        maps[0, 0, 4, 4] = 1
        maps[1, 0, 4, 4] = 2
        smoothed = gaussian_smooth(maps, kernel_size=5, sigma=1.0)
        scores = aggregate_anomaly_maps(maps, method="max")
        self.assertEqual(tuple(scores.shape), (2,))
        self.assertTrue(torch.allclose(scores, smoothed.flatten(1).amax(dim=1)))
        self.assertTrue(torch.all(scores < maps.flatten(1).amax(dim=1)))

    def test_maximum_preserves_constant_maps_including_boundaries(self):
        maps = torch.full((2, 5, 7), 3.0)
        scores = aggregate_anomaly_maps(maps, method="max")
        self.assertTrue(torch.allclose(scores, torch.tensor([3.0, 3.0])))

    def test_invalid_inputs_and_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            aggregate_anomaly_maps(torch.zeros(2, 3, 4, 5))
        with self.assertRaisesRegex(ValueError, "top_fraction"):
            aggregate_anomaly_maps(torch.zeros(1, 4, 4), top_fraction=0)
        with self.assertRaisesRegex(ValueError, "method"):
            aggregate_anomaly_maps(torch.zeros(1, 4, 4), method="median")
        with self.assertRaisesRegex(ValueError, "odd"):
            aggregate_anomaly_maps(
                torch.zeros(1, 4, 4), method="max", smoothing_kernel_size=2
            )


if __name__ == "__main__":
    unittest.main()
