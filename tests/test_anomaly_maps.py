import unittest

import torch

from src.anomaly_maps import compute_anomaly_maps, normalize_map_for_visualization


class AnomalyMapTests(unittest.TestCase):
    def test_l2_discrepancy_is_upsampled_as_float32_without_normalization(self):
        original = torch.tensor([[[[3.0]], [[4.0]]]], dtype=torch.float16)
        reconstructed = torch.zeros_like(original)

        maps = compute_anomaly_maps(original, reconstructed, output_size=(5, 7))

        self.assertEqual(tuple(maps.shape), (1, 5, 7))
        self.assertEqual(maps.dtype, torch.float32)
        self.assertTrue(torch.allclose(maps, torch.full((1, 5, 7), 5.0)))

    def test_optional_gaussian_smoothing_preserves_shape_and_float_scale(self):
        original = torch.zeros(2, 3, 5, 5)
        original[:, :, 2, 2] = 4
        reconstructed = torch.zeros_like(original)

        raw = compute_anomaly_maps(original, reconstructed, output_size=(10, 10))
        smoothed = compute_anomaly_maps(
            original,
            reconstructed,
            output_size=(10, 10),
            smoothing_kernel_size=5,
            smoothing_sigma=1.0,
        )

        self.assertEqual(tuple(smoothed.shape), (2, 10, 10))
        self.assertTrue(torch.isfinite(smoothed).all())
        self.assertTrue(torch.all(smoothed >= 0))
        self.assertLess(smoothed.max().item(), raw.max().item())

    def test_visualization_normalization_does_not_change_raw_map(self):
        maps = torch.tensor([[[2.0, 4.0], [6.0, 8.0]], [[7.0, 7.0], [7.0, 7.0]]])
        original = maps.clone()

        normalized = normalize_map_for_visualization(maps)

        self.assertTrue(torch.equal(maps, original))
        self.assertEqual(normalized[0].min().item(), 0.0)
        self.assertEqual(normalized[0].max().item(), 1.0)
        self.assertTrue(torch.equal(normalized[1], torch.zeros(2, 2)))

    def test_feature_shapes_and_output_size_are_validated(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            compute_anomaly_maps(torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 2, 3), (8, 8))
        with self.assertRaisesRegex(ValueError, "output_size"):
            compute_anomaly_maps(torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 3, 3), (8, 0))


if __name__ == "__main__":
    unittest.main()
