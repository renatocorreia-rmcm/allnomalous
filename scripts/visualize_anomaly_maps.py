"""Export an untrained-decoder map smoke test beside MVTec defect masks."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_paths import DEFAULT_DATA_ROOT

from src.anomaly_maps import compute_anomaly_maps, normalize_map_for_visualization
from src.feature_extractor import WideResNetFeatureExtractor
from src.mvtec_dataset import MVTecTestDataset, IMAGENET_MEAN, IMAGENET_STD
from src.reconstruction_decoder import ReconstructionDecoder


def _heatmap(normalized_map: torch.Tensor) -> Image.Image:
    values = normalized_map.cpu().numpy()
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    rgb = np.stack((red, green, blue), axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")


def _display_image(normalized: torch.Tensor) -> Image.Image:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = (normalized.cpu() * std + mean).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray((image * 255).astype(np.uint8), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", default="figures/anomaly_maps_smoke_test.png")
    parser.add_argument("--categories", nargs="+", default=["bottle", "capsule"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    dataset = MVTecTestDataset(args.data_root, image_size=512)
    selected = []
    for category in args.categories:
        sample_index = next(
            (
                index
                for index, (_, _, label, sample_category) in enumerate(dataset.samples)
                if label == 1 and sample_category == category
            ),
            None,
        )
        if sample_index is None:
            raise ValueError(f"No defective test image found for {category!r}.")
        selected.append((category, sample_index))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    extractor = WideResNetFeatureExtractor(pretrained=True).to(device).eval()
    # This preview checks the map pipeline and spatial alignment, not model quality.
    decoder = ReconstructionDecoder(
        neighbor_size=7, jitter_scale=20, jitter_probability=1
    ).to(device).eval()

    samples = [dataset[index] for _, index in selected]
    images = torch.stack([sample[0] for sample in samples]).to(device)
    masks = torch.stack([sample[1][0] for sample in samples])
    with torch.no_grad():
        features = extractor(images)
        reconstructions = decoder(features)
        raw_maps = compute_anomaly_maps(
            features, reconstructions, output_size=tuple(images.shape[-2:])
        )
    if tuple(raw_maps.shape[-2:]) != tuple(masks.shape[-2:]):
        raise RuntimeError("Anomaly maps and transformed ground-truth masks are misaligned.")

    labels = ["Image (512x512)", "Ground-truth mask", "Anomaly map overlay"]
    font = ImageFont.load_default()
    panel_width, panel_height = 512, 552
    margin, header_height = 24, 76
    canvas = Image.new(
        "RGB",
        (margin * 2 + panel_width * 3, header_height + panel_height * len(samples)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), "Anomaly-map pipeline smoke test", fill="black", font=font)
    draw.text(
        (margin, 34),
        "Decoder untrained; map normalized only for display. Raw float32 map is used for metrics.",
        fill="black",
        font=font,
    )

    visual_maps = normalize_map_for_visualization(raw_maps)
    for index, ((category, sample_index), sample) in enumerate(zip(selected, samples)):
        source_path = dataset.samples[sample_index][0]
        original = _display_image(sample[0])
        mask_array = (masks[index].numpy() * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_array, mode="L").convert("RGB")
        heat = _heatmap(visual_maps[index])
        overlay = Image.blend(original, heat, alpha=0.45)
        panels = (original, mask_image, overlay)
        row_top = header_height + index * panel_height
        draw.text(
            (margin, row_top),
            f"{category} / {source_path.parent.name}",
            fill="black",
            font=font,
        )
        for column, (label, panel) in enumerate(zip(labels, panels)):
            left = margin + column * panel_width
            draw.text((left, row_top + 20), label, fill="black", font=font)
            canvas.paste(panel, (left, row_top + 40))
        print(
            f"{category}: map={tuple(raw_maps[index].shape)}, "
            f"dtype={raw_maps.dtype}, range="
            f"[{raw_maps[index].min().item():.6f}, {raw_maps[index].max().item():.6f}]"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"device: {device}")
    print(f"saved: {output.resolve()}")


if __name__ == "__main__":
    main()
