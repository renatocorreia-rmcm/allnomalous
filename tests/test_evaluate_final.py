import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_final import (
    _build_decoder,
    _read_model_inputs,
    _save_prediction_archive,
    _write_metrics_csv,
)


class FinalEvaluationPreparationTests(unittest.TestCase):
    def test_build_decoder_reproduces_baseline_and_complete_options(self):
        baseline = _build_decoder(
            {
                "decoder_hidden_dim": 8,
                "decoder_num_encoder_layers": 1,
                "decoder_num_decoder_layers": 1,
                "jitter": False,
                "neighbor_mask": False,
            }
        )
        complete = _build_decoder(
            {
                "decoder_hidden_dim": 8,
                "decoder_num_encoder_layers": 1,
                "decoder_num_decoder_layers": 1,
                "jitter": True,
                "neighbor_mask": True,
            }
        )

        self.assertIsNone(baseline.neighbor_size)
        self.assertEqual(baseline.jitter_scale, 0.0)
        self.assertEqual(complete.neighbor_size, 7)
        self.assertEqual(complete.jitter_scale, 20.0)
        self.assertEqual(complete.jitter_probability, 1.0)

    def test_checkpoint_config_must_match_metadata_and_variant(self):
        configuration = {
            "image_size": 512,
            "jitter": False,
            "neighbor_mask": False,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "best_decoder.pt"
            metadata_path = root / "training_config.json"
            torch.save(
                {"config": configuration, "decoder_state_dict": {}}, checkpoint_path
            )
            metadata_path.write_text(
                json.dumps({"config": configuration, "repo_commit": "abc123"}),
                encoding="utf-8",
            )

            checkpoint, metadata, loaded_config = _read_model_inputs(
                checkpoint_path, metadata_path, "baseline"
            )

            self.assertEqual(checkpoint["config"], configuration)
            self.assertEqual(metadata["repo_commit"], "abc123")
            self.assertEqual(loaded_config, configuration)
            with self.assertRaisesRegex(ValueError, "complete checkpoint"):
                _read_model_inputs(checkpoint_path, metadata_path, "complete")

            metadata_path.write_text(
                json.dumps({"config": {"image_size": 256}}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                _read_model_inputs(checkpoint_path, metadata_path, "baseline")

    def test_prediction_archive_packs_masks_and_preserves_float16_maps(self):
        artifacts = {
            "image_scores": np.array([0.1, 0.9], dtype=np.float32),
            "image_labels": np.array([0, 1], dtype=np.uint8),
            "categories": np.array(["bottle", "bottle"]),
            "anomaly_maps": np.arange(32, dtype=np.float16).reshape(2, 4, 4),
            "masks": np.zeros((2, 4, 4), dtype=np.uint8),
        }
        artifacts["masks"][1, 1:3, 1:3] = 1

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "predictions.npz"
            _save_prediction_archive(path, artifacts, {"commit": "abc123"})

            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["anomaly_maps"].dtype, np.float16)
                self.assertEqual(tuple(archive["anomaly_maps"].shape), (2, 4, 4))
                restored_masks = np.unpackbits(
                    archive["packed_masks"], axis=1, count=16
                ).reshape(2, 4, 4)
                self.assertTrue(np.array_equal(restored_masks, artifacts["masks"]))
                self.assertEqual(json.loads(str(archive["metadata_json"])), {"commit": "abc123"})

    def test_metrics_csv_has_per_category_and_macro_rows(self):
        results = {
            "baseline": {
                "per_category": {
                    "bottle": {
                        "image_count": 10,
                        "pixel_count": 1000,
                        "region_count": 5,
                        "image_auroc": 0.8,
                        "pixel_auroc": 0.9,
                        "aupro": 0.7,
                    }
                },
                "macro": {
                    "image_auroc": 0.8,
                    "pixel_auroc": 0.9,
                    "aupro": 0.7,
                    "image_auroc_categories": 1,
                    "pixel_auroc_categories": 1,
                    "aupro_categories": 1,
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "metrics.csv"
            _write_metrics_csv(path, results)
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
        self.assertEqual([row["category"] for row in rows], ["bottle", "macro"])
        self.assertEqual(rows[1]["image_auroc"], "0.8")


if __name__ == "__main__":
    unittest.main()
