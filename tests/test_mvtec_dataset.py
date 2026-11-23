import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler

from src.mvtec_dataset import (
    MVTecTestDataset,
    MVTecTrainDataset,
    MVTecValidationDataset,
    create_dataloaders,
    create_test_dataloader,
)


class MVTecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        self._create_category("bottle")
        self._create_category("grid")
        self._write_split("train_split.csv", ["bottle", "grid"])
        self._write_split("val_split.csv", ["bottle", "grid"])

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_category(self, category):
        category_dir = self.data_root / category
        for split in ("train/good", "test/good", "test/crack", "ground_truth/crack"):
            (category_dir / split).mkdir(parents=True, exist_ok=True)

        image = np.full((8, 8, 3), 127, dtype=np.uint8)
        Image.fromarray(image).save(category_dir / "train/good/000.png")
        Image.fromarray(image).save(category_dir / "test/good/000.png")
        Image.fromarray(image).save(category_dir / "test/crack/000.png")
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0, 0] = 255
        Image.fromarray(mask).save(
            category_dir / "ground_truth/crack/000_mask.png"
        )

    def _write_split(self, filename, categories):
        with (self.data_root / filename).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("category", "path"))
            for category in categories:
                writer.writerow((category, f"{category}/train/good/000.png"))

    def test_train_and_validation_return_only_normalized_images(self):
        train = MVTecTrainDataset(self.data_root, image_size=16)
        validation = MVTecValidationDataset(self.data_root, image_size=(16, 16))
        image = train[0]
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 2)
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertAlmostEqual(
            float(image[0, 4, 4]), (127 / 255 - 0.485) / 0.229, places=6
        )

        class ImageOnlyModel(nn.Module):
            def forward(self, batch):
                self.assert_tensor(batch)
                return batch

            @staticmethod
            def assert_tensor(batch):
                if not isinstance(batch, torch.Tensor):
                    raise TypeError("The model must receive images only.")

        train_loader, _ = create_dataloaders(
            self.data_root, batch_size=2, image_size=16
        )
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        batch = next(iter(train_loader))
        self.assertEqual(tuple(ImageOnlyModel()(batch).shape), (2, 3, 16, 16))

    def test_test_dataset_resizes_image_and_mask_together(self):
        dataset = MVTecTestDataset(self.data_root, image_size=(16, 16))
        defect_index = next(
            index
            for index, (_, mask_path, label, _) in enumerate(dataset.samples)
            if mask_path is not None and label == 1
        )
        image, mask, label, category = dataset[defect_index]
        self.assertEqual(tuple(image.shape), (3, 16, 16))
        self.assertEqual(tuple(mask.shape), (1, 16, 16))
        self.assertEqual(label, 1)
        self.assertIn(category, ("bottle", "grid"))
        self.assertEqual(int(mask.sum()), 4)
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})

        normal_index = next(
            index
            for index, (_, mask_path, label, _) in enumerate(dataset.samples)
            if mask_path is None and label == 0
        )
        _, normal_mask, normal_label, _ = dataset[normal_index]
        self.assertEqual(normal_label, 0)
        self.assertEqual(int(normal_mask.sum()), 0)

        loader = create_test_dataloader(
            self.data_root, batch_size=4, image_size=16
        )
        self.assertIsInstance(loader.sampler, SequentialSampler)
        test_images, test_masks, labels, categories = next(iter(loader))
        self.assertEqual(tuple(test_images.shape[1:]), (3, 16, 16))
        self.assertEqual(tuple(test_masks.shape[1:]), (1, 16, 16))
        self.assertEqual(len(labels), 4)
        self.assertEqual(len(categories), 4)


if __name__ == "__main__":
    unittest.main()
