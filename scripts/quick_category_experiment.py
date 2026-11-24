"""Short baseline experiment using normal train/validation images only."""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.anomaly_maps import compute_anomaly_maps, normalize_map_for_visualization
from src.anomaly_scoring import aggregate_anomaly_maps
from src.feature_extractor import WideResNetFeatureExtractor
from src.mvtec_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MVTecTrainDataset,
    MVTecValidationDataset,
)
from src.reconstruction_decoder import ReconstructionDecoder, reconstruction_loss


def _category_indices(dataset, category: str) -> List[int]:
    return [
        index
        for index, path in enumerate(dataset.image_paths)
        if category in path.parts
    ]


def _display_rgb(tensor: torch.Tensor) -> Image.Image:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    rgb = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    values = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(values, mode="RGB")


def _heatmap(normalized_map: torch.Tensor) -> Image.Image:
    values = normalized_map.detach().cpu().numpy()
    channels = (
        np.clip(1.5 - np.abs(4 * values - 3), 0, 1),
        np.clip(1.5 - np.abs(4 * values - 2), 0, 1),
        np.clip(1.5 - np.abs(4 * values - 1), 0, 1),
    )
    return Image.fromarray(
        (np.stack(channels, axis=-1) * 255).astype(np.uint8), mode="RGB"
    )


def _save_loss_curve(
    history: List[Dict[str, float]], output_path: Path
) -> None:
    width, height = 1000, 600
    left, top, right, bottom = 100, 70, 940, 490
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 22), "Quick baseline training: bottle (decoder only)", fill="black", font=font)
    draw.text((left, 42), "Train and normal-validation reconstruction MSE", fill="black", font=font)
    draw.line((left, top, left, bottom, right, bottom), fill=(60, 60, 60), width=2)

    values = [entry["train_loss"] for entry in history] + [
        entry["val_loss"] for entry in history
    ]
    low, high = min(values), max(values)
    if high - low < 1e-12:
        high = low + 1

    def point(index: int, value: float) -> Tuple[int, int]:
        x = left + int(index * (right - left) / max(1, len(history) - 1))
        y = bottom - int((value - low) * (bottom - top) / (high - low))
        return x, y

    for index, entry in enumerate(history):
        x, _ = point(index, entry["train_loss"])
        draw.line((x, top, x, bottom), fill=(235, 235, 235), width=1)
    for key, color in (
        ("train_loss", (9, 105, 218)),
        ("val_loss", (207, 34, 46)),
    ):
        points = [point(index, entry[key]) for index, entry in enumerate(history)]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    draw.text((left, bottom + 22), f"Epochs: {len(history)}", fill="black", font=font)
    draw.text((left, bottom + 40), f"MSE range: {low:.5g} - {high:.5g}", fill="black", font=font)
    draw.line((700, 35, 735, 35), fill=(9, 105, 218), width=4)
    draw.text((742, 29), "Train", fill="black", font=font)
    draw.line((810, 35, 845, 35), fill=(207, 34, 46), width=4)
    draw.text((852, 29), "Normal validation", fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _save_examples(
    normal_image: torch.Tensor,
    normal_map: torch.Tensor,
    synthetic_image: torch.Tensor,
    synthetic_mask: torch.Tensor,
    synthetic_map: torch.Tensor,
    patch_box: Tuple[int, int, int, int],
    normal_score: float,
    synthetic_score: float,
    output_path: Path,
) -> None:
    panel_size = normal_image.shape[-1]
    margin, header = 24, 82
    columns = [
        ("Normal validation image", _display_rgb(normal_image)),
        (
            "Normal validation map",
            _heatmap(normalize_map_for_visualization(normal_map.unsqueeze(0))[0]),
        ),
        ("Synthetic pasted-patch image", _display_rgb(synthetic_image)),
        (
            "Artificial patch mask",
            Image.fromarray(
                (synthetic_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                mode="L",
            ).convert("RGB"),
        ),
        (
            "Synthetic defect + map",
            Image.blend(
                _display_rgb(synthetic_image),
                _heatmap(
                    normalize_map_for_visualization(synthetic_map.unsqueeze(0))[0]
                ),
                alpha=0.45,
            ),
        ),
    ]
    width = margin * 2 + panel_size * len(columns) + 32 * (len(columns) - 1)
    image = Image.new("RGB", (width, header + panel_size + 2 * margin), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (margin, 12),
        "Bottle quick validation — no official test images used",
        fill="black",
        font=font,
    )
    draw.text(
        (margin, 34),
        "Decoder trained briefly; maps min-max normalized for display only.",
        fill="black",
        font=font,
    )
    draw.text(
        (margin, 55),
        f"Normal top-1% score={normal_score:.4f}; synthetic top-1% score={synthetic_score:.4f}",
        fill="black",
        font=font,
    )
    left = margin
    for label, panel in columns:
        draw.text((left, header - 20), label, fill="black", font=font)
        image.paste(panel, (left, header))
        if label == "Synthetic pasted-patch image":
            draw.rectangle(
                (
                    left + patch_box[0],
                    header + patch_box[1],
                    left + patch_box[2],
                    header + patch_box[3],
                ),
                outline=(255, 0, 0),
                width=3,
            )
        left += panel_size + 32

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def run_experiment(
    data_root: Path,
    output_dir: Path,
    category: str,
    image_size: int,
    epochs: int,
    batch_size: int,
    max_train_samples: int,
    max_val_samples: int,
    device_name: str,
) -> Dict[str, object]:
    if image_size < 32 or image_size % 16:
        raise ValueError("image_size must be at least 32 and divisible by 16.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if max_train_samples < 0 or max_val_samples < 0:
        raise ValueError("sample limits must be non-negative.")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_dataset = MVTecTrainDataset(data_root, image_size=image_size)
    val_dataset = MVTecValidationDataset(data_root, image_size=image_size)
    train_indices = _category_indices(train_dataset, category)
    val_indices = _category_indices(val_dataset, category)
    if not train_indices or not val_indices:
        raise ValueError(f"Missing train or validation samples for {category!r}.")
    if max_train_samples > 0:
        train_indices = train_indices[:max_train_samples]
    if max_val_samples > 0:
        val_indices = val_indices[:max_val_samples]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    extractor = WideResNetFeatureExtractor(pretrained=True).to(device).eval()
    decoder = ReconstructionDecoder(
        base_grid_size=(image_size // 16, image_size // 16)
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    history = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        decoder.train()
        train_losses = []
        for images in train_loader:
            images = images.to(device)
            with torch.no_grad():
                targets = extractor(images)
            optimizer.zero_grad(set_to_none=True)
            reconstructions = decoder(targets)
            loss = reconstruction_loss(targets, reconstructions)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        decoder.eval()
        val_losses = []
        with torch.no_grad():
            for images in val_loader:
                targets = extractor(images.to(device))
                reconstructions = decoder(targets)
                loss = reconstruction_loss(targets, reconstructions)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite validation loss at epoch {epoch}."
                    )
                val_losses.append(loss.item())
        elapsed = time.perf_counter() - epoch_started
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "epoch_seconds": elapsed,
        }
        history.append(record)
        print(
            f"epoch={epoch}/{epochs} train_mse={record['train_loss']:.6f} "
            f"val_mse={record['val_loss']:.6f} seconds={elapsed:.1f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "losses.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=("epoch", "train_loss", "val_loss", "epoch_seconds")
        )
        writer.writeheader()
        writer.writerows(history)
    _save_loss_curve(history, output_dir / "loss_curve.png")

    # Use a normal validation image and a crop from a different category.
    normal_index = val_indices[0]
    normal_image = val_dataset[normal_index]
    donor_index = next(
        (
            index
            for index, path in enumerate(val_dataset.image_paths)
            if category not in path.parts
        ),
        None,
    )
    if donor_index is None:
        donor_image = torch.flip(normal_image, dims=(2,))
    else:
        donor_image = val_dataset[donor_index]

    patch_size = max(16, image_size // 4)
    x0 = image_size // 2
    y0 = image_size // 2
    x1 = min(image_size, x0 + patch_size)
    y1 = min(image_size, y0 + patch_size)
    synthetic_image = normal_image.clone()
    donor_x0 = max(0, (image_size - (x1 - x0)) // 3)
    donor_y0 = max(0, (image_size - (y1 - y0)) // 3)
    synthetic_image[:, y0:y1, x0:x1] = donor_image[
        :, donor_y0 : donor_y0 + (y1 - y0), donor_x0 : donor_x0 + (x1 - x0)
    ]
    patch_box = (x0, y0, x1, y1)

    decoder.eval()
    validation_maps = []
    validation_scores = []
    with torch.no_grad():
        for images in val_loader:
            targets = extractor(images.to(device))
            reconstructions = decoder(targets)
            maps = compute_anomaly_maps(
                targets,
                reconstructions,
                output_size=(image_size, image_size),
            )
            scores = aggregate_anomaly_maps(
                maps, method="topk_mean", top_fraction=0.01
            )
            validation_maps.extend(maps.cpu().unbind(0))
            validation_scores.extend(scores.cpu().unbind(0))

        pair = torch.stack((normal_image, synthetic_image)).to(device)
        pair_targets = extractor(pair)
        pair_reconstructions = decoder(pair_targets)
        pair_maps = compute_anomaly_maps(
            pair_targets, pair_reconstructions, output_size=(image_size, image_size)
        )
        pair_scores = aggregate_anomaly_maps(
            pair_maps, method="topk_mean", top_fraction=0.01
        )

    normal_score = float(validation_scores[0].item())
    synthetic_score = float(pair_scores[1].item())
    patch_mean = float(pair_maps[1, y0:y1, x0:x1].mean().item())
    outside_mask = torch.ones_like(pair_maps[1], dtype=torch.bool)
    outside_mask[y0:y1, x0:x1] = False
    outside_mean = float(pair_maps[1][outside_mask].mean().item())
    normal_validation_mean = float(torch.stack(validation_scores).mean().item())
    synthetic_mask = torch.zeros((image_size, image_size))
    synthetic_mask[y0:y1, x0:x1] = 1
    _save_examples(
        normal_image,
        validation_maps[0],
        synthetic_image,
        synthetic_mask,
        pair_maps[1],
        patch_box,
        normal_score,
        synthetic_score,
        output_dir / "validation_and_synthetic_examples.png",
    )
    np.save(
        output_dir / "normal_validation_map_raw.npy",
        validation_maps[0].numpy().astype(np.float32),
    )
    np.save(output_dir / "synthetic_defect_map_raw.npy", pair_maps[1].cpu().numpy())

    epoch_seconds = float(np.mean([entry["epoch_seconds"] for entry in history]))
    full_epochs = 30
    full_samples = 3262
    estimated_one_model_seconds = (
        epoch_seconds * full_samples / len(train_indices) * full_epochs
    )
    result = {
        "category": category,
        "device": str(device),
        "image_size": image_size,
        "epochs": epochs,
        "train_samples_per_epoch": len(train_indices),
        "validation_samples": len(val_indices),
        "train_loss_first": history[0]["train_loss"],
        "train_loss_last": history[-1]["train_loss"],
        "validation_loss_first": history[0]["val_loss"],
        "validation_loss_last": history[-1]["val_loss"],
        "epoch_seconds_mean": epoch_seconds,
        "validation_normal_top1_percent_score_mean": normal_validation_mean,
        "normal_pair_top1_percent_score": normal_score,
        "synthetic_pair_top1_percent_score": synthetic_score,
        "synthetic_patch_map_mean": patch_mean,
        "synthetic_outside_patch_map_mean": outside_mean,
        "synthetic_patch_to_outside_map_ratio": patch_mean / max(outside_mean, 1e-12),
        "estimated_cpu_30_epoch_full_dataset_one_model_hours": (
            estimated_one_model_seconds / 3600
        ),
        "estimate_note": (
            "Linear extrapolation from this CPU run at reduced resolution; "
            "not a GPU estimate and excludes validation/overhead."
        ),
        "official_test_used": False,
        "decoder_trained": True,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"Artifacts: {output_dir.resolve()}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="analysis/quick_validation_bottle")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run_experiment(
        data_root=Path(args.data_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        category=args.category,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
