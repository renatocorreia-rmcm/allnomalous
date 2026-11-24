import unittest

import torch

from src.feature_extractor import (
    WideResNetFeatureExtractor,
    benchmark_feature_extractor,
)


class WideResNetFeatureExtractorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extractor = WideResNetFeatureExtractor(pretrained=False)

    def test_features_are_aligned_concatenated_frozen_and_without_gradients(self):
        images = torch.randn(1, 3, 64, 64, requires_grad=True)
        features = self.extractor(images)

        self.assertEqual(tuple(features.shape), (1, 1792, 4, 4))
        self.assertFalse(features.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.extractor.parameters()))
        self.assertFalse(self.extractor.backbone.training)

    def test_extractor_remains_in_eval_mode_when_parent_mode_changes(self):
        self.extractor.train()
        self.assertTrue(self.extractor.training)
        self.assertFalse(self.extractor.backbone.training)
        self.extractor.eval()
        self.assertFalse(self.extractor.backbone.training)

    def test_invalid_images_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            self.extractor(torch.randn(1, 1, 64, 64))
        with self.assertRaisesRegex(ValueError, "at least 32"):
            self.extractor(torch.randn(1, 3, 16, 16))

    def test_cpu_benchmark_reports_latency_without_cuda_memory(self):
        result = benchmark_feature_extractor(
            self.extractor,
            input_shape=(1, 3, 32, 32),
            device="cpu",
            warmup=0,
            iterations=1,
        )
        self.assertEqual(result["device"], "cpu")
        self.assertGreater(result["latency_ms_per_batch"], 0)
        self.assertIsNone(result["peak_cuda_allocated_mb"])
        self.assertIsNone(result["peak_cuda_reserved_mb"])


if __name__ == "__main__":
    unittest.main()
