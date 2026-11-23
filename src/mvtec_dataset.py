"""PyTorch datasets for class-independent MVTec AD training and evaluation."""

import csv
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DEFAULT_IMAGE_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ImageSize = Union[int, Tuple[int, int]]


def _normalize_image_size(image_size: ImageSize) -> Tuple[int, int]:
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    if (
        len(image_size) != 2
        or not all(isinstance(side, int) and side > 0 for side in image_size)
    ):
        raise ValueError("image_size must be a positive integer or (height, width).")
    return image_size


def _image_to_tensor(path: Path, image_size: Tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _mask_to_tensor(path: Path, image_size: Tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    with Image.open(path) as source:
        mask = source.convert("L").resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
    array = np.array(mask, dtype=np.uint8, copy=True)
    return torch.from_numpy((array > 0).astype(np.float32)).unsqueeze(0)


class _NormalImageDataset(Dataset):
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str,
        image_size: ImageSize = DEFAULT_IMAGE_SIZE,
        split_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.image_size = _normalize_image_size(image_size)
        csv_path = (
            Path(split_file)
            if split_file is not None
            else self.data_root / f"{split}_split.csv"
        )
        if not csv_path.is_absolute():
            csv_path = self.data_root / csv_path
        if not csv_path.is_file():
            raise FileNotFoundError(f"{split} split CSV not found: {csv_path}")

        with csv_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames or "path" not in reader.fieldnames:
                raise ValueError(f"{csv_path} must contain a 'path' column.")
            self.image_paths = [
                (self.data_root / row["path"]).resolve()
                for row in reader
                if row.get("path")
            ]

        if not self.image_paths:
            raise ValueError(f"No image paths found in {csv_path}.")
        for path in self.image_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Image in split CSV not found: {path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        return _image_to_tensor(self.image_paths[index], self.image_size)


class MVTecTrainDataset(_NormalImageDataset):
    """Combined normal training images; returns only an ImageNet-normalized image."""

    def __init__(
        self,
        data_root: Union[str, Path],
        image_size: ImageSize = DEFAULT_IMAGE_SIZE,
        split_file: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(data_root, "train", image_size, split_file)


class MVTecValidationDataset(_NormalImageDataset):
    """Combined normal validation images; returns only an ImageNet-normalized image."""

    def __init__(
        self,
        data_root: Union[str, Path],
        image_size: ImageSize = DEFAULT_IMAGE_SIZE,
        split_file: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__(data_root, "val", image_size, split_file)


class MVTecTestDataset(Dataset):
    """Official test images with binary masks, image labels, and category metadata."""

    def __init__(
        self,
        data_root: Union[str, Path],
        image_size: ImageSize = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.image_size = _normalize_image_size(image_size)
        self.samples: List[Tuple[Path, Optional[Path], int, str]] = []
        categories = sorted(
            path
            for path in self.data_root.iterdir()
            if path.is_dir()
            and (path / "train" / "good").is_dir()
            and (path / "test").is_dir()
        )
        if not categories:
            raise FileNotFoundError(f"No MVTec categories found in {self.data_root}")

        for category_dir in categories:
            for defect_dir in sorted(
                path for path in (category_dir / "test").iterdir() if path.is_dir()
            ):
                label = int(defect_dir.name != "good")
                for image_path in sorted(defect_dir.iterdir()):
                    if (
                        not image_path.is_file()
                        or image_path.suffix.lower() not in IMAGE_EXTENSIONS
                    ):
                        continue
                    mask_path = None
                    if label:
                        mask_path = (
                            category_dir
                            / "ground_truth"
                            / defect_dir.name
                            / f"{image_path.stem}_mask.png"
                        )
                        if not mask_path.is_file():
                            raise FileNotFoundError(
                                f"Ground-truth mask not found for {image_path}: "
                                f"{mask_path}"
                            )
                    self.samples.append(
                        (image_path, mask_path, label, category_dir.name)
                    )

        if not self.samples:
            raise ValueError(f"No test images found in {self.data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        image_path, mask_path, label, category = self.samples[index]
        image = _image_to_tensor(image_path, self.image_size)
        if mask_path is None:
            mask = torch.zeros((1, *self.image_size), dtype=torch.float32)
        else:
            with Image.open(image_path) as source_image:
                image_dimensions = source_image.size
            with Image.open(mask_path) as source_mask:
                mask_dimensions = source_mask.size
            if image_dimensions != mask_dimensions:
                raise ValueError(
                    f"Image and mask dimensions differ for {image_path}: "
                    f"{image_dimensions} vs {mask_dimensions}"
                )
            mask = _mask_to_tensor(mask_path, self.image_size)
        return image, mask, label, category


def create_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    image_size: ImageSize = DEFAULT_IMAGE_SIZE,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create mixed-category train and validation loaders; only images are batched."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    train_dataset = MVTecTrainDataset(data_root, image_size)
    validation_dataset = MVTecValidationDataset(data_root, image_size)
    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    )


def create_test_dataloader(
    data_root: Union[str, Path],
    batch_size: int = 32,
    image_size: ImageSize = DEFAULT_IMAGE_SIZE,
    num_workers: int = 0,
) -> DataLoader:
    """Create the ordered test loader; category strings are evaluation metadata."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    return DataLoader(
        MVTecTestDataset(data_root, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
