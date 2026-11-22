import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SMALL_DEFECT_THRESHOLD = 0.01


def _image_paths(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _categories(data_root: Path) -> List[Path]:
    categories = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir()
        and (path / "train" / "good").is_dir()
        and (path / "test").is_dir()
    )
    if not categories:
        raise FileNotFoundError(f"No MVTec categories found in {data_root}")
    return categories


def _inspect_image_profiles(
    categories: List[Path],
) -> Tuple[List[Dict[str, object]], Dict[Path, List[Path]], Dict[str, List[int]]]:
    profile_counts: Counter = Counter()
    test_images: Dict[Path, List[Path]] = {}
    image_sides: Dict[str, List[int]] = defaultdict(list)

    for category in categories:
        image_directories = [category / "train" / "good"]
        image_directories.extend(
            sorted(
                path
                for path in (category / "test").iterdir()
                if path.is_dir()
            )
        )
        for directory in image_directories:
            paths = _image_paths(directory)
            if directory.parent.name == "test":
                test_images[directory] = paths
            for path in paths:
                with Image.open(path) as image:
                    width, height = image.size
                    mode = image.mode
                profile_counts[(category.name, height, width, mode)] += 1
                image_sides[category.name].append(min(height, width))

    profile_rows = [
        {
            "category": category,
            "height": height,
            "width": width,
            "color_mode": mode,
            "image_count": count,
        }
        for (category, height, width, mode), count in sorted(profile_counts.items())
    ]
    return profile_rows, test_images, image_sides


def _inspect_masks(
    categories: List[Path],
    test_images: Dict[Path, List[Path]],
) -> Tuple[List[Dict[str, object]], Dict[str, List[float]]]:
    mask_rows: List[Dict[str, object]] = []
    ratios_by_category: Dict[str, List[float]] = defaultdict(list)

    for category in categories:
        ground_truth = category / "ground_truth"
        if not ground_truth.is_dir():
            continue
        for mask_path in sorted(ground_truth.rglob("*_mask.png")):
            defect_type = mask_path.parent.name
            image_stem = mask_path.stem[:-5]
            image_directory = category / "test" / defect_type
            image_matches = [
                path
                for path in test_images.get(image_directory, [])
                if path.stem == image_stem
            ]
            if len(image_matches) != 1:
                raise FileNotFoundError(
                    f"Expected one test image for mask {mask_path}, "
                    f"found {len(image_matches)}."
                )
            image_path = image_matches[0]
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            with Image.open(mask_path) as mask:
                mask_width, mask_height = mask.size
                histogram = mask.convert("L").histogram()
            anomalous_pixels = sum(histogram[1:])
            image_pixels = image_width * image_height
            proportion = anomalous_pixels / image_pixels
            size_matches = (image_width, image_height) == (mask_width, mask_height)
            ratios_by_category[category.name].append(proportion)
            mask_rows.append(
                {
                    "category": category.name,
                    "defect_type": defect_type,
                    "image_path": image_path.relative_to(image_path.parents[2]).as_posix(),
                    "mask_path": mask_path.relative_to(mask_path.parents[2]).as_posix(),
                    "image_height": image_height,
                    "image_width": image_width,
                    "mask_height": mask_height,
                    "mask_width": mask_width,
                    "anomalous_pixel_count": anomalous_pixels,
                    "image_pixel_count": image_pixels,
                    "anomaly_proportion": f"{proportion:.12g}",
                    "anomaly_percent": f"{proportion * 100:.8g}",
                    "image_mask_size_match": size_matches,
                }
            )
    return mask_rows, ratios_by_category


def _summary_rows(
    categories: List[Path],
    mask_rows: List[Dict[str, object]],
    ratios_by_category: Dict[str, List[float]],
) -> List[Dict[str, object]]:
    rows = []
    for category in categories:
        ratios = ratios_by_category.get(category.name, [])
        if not ratios:
            continue
        category_masks = [
            row for row in mask_rows if row["category"] == category.name
        ]
        smallest_index = min(range(len(ratios)), key=ratios.__getitem__)
        smallest_mask = category_masks[smallest_index]
        rows.append(
            {
                "category": category.name,
                "mask_count": len(ratios),
                "mean_anomaly_proportion": f"{statistics.mean(ratios):.12g}",
                "median_anomaly_proportion": f"{statistics.median(ratios):.12g}",
                "minimum_anomaly_proportion": f"{min(ratios):.12g}",
                "minimum_anomaly_percent": f"{min(ratios) * 100:.8g}",
                "smallest_defect_mask": smallest_mask["mask_path"],
                "masks_below_1_percent": sum(
                    ratio < SMALL_DEFECT_THRESHOLD for ratio in ratios
                ),
                "masks_below_0_1_percent": sum(
                    ratio < SMALL_DEFECT_THRESHOLD / 10 for ratio in ratios
                ),
                "image_mask_size_mismatches": sum(
                    not bool(row["image_mask_size_match"]) for row in category_masks
                ),
            }
        )
    return rows


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_histogram(
    path: Path,
    mask_rows: List[Dict[str, object]],
    dpi: int,
) -> None:
    ratios = [float(row["anomaly_proportion"]) for row in mask_rows]
    positive_ratios = [ratio for ratio in ratios if ratio > 0]
    zero_count = len(ratios) - len(positive_ratios)
    figure, axis = plt.subplots(figsize=(10, 6))

    if positive_ratios:
        lower = min(positive_ratios)
        upper = max(max(positive_ratios), SMALL_DEFECT_THRESHOLD * 1.1)
        if upper <= lower:
            upper = lower * 1.1
        bins = np.geomspace(lower, upper, 45)
        axis.hist(positive_ratios, bins=bins, color="#3b82a0", edgecolor="white")
        axis.set_xscale("log")
        axis.set_xlim(left=lower, right=upper)
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value * 100:g}%")
        )
        axis.axvline(
            SMALL_DEFECT_THRESHOLD,
            color="#d1495b",
            linestyle="--",
            linewidth=1.5,
            label="Limite de referência: 1%",
        )
    elif zero_count:
        axis.bar([0], [zero_count], width=0.005, color="#3b82a0")
        axis.set_xlim(-0.005, 0.02)

    below_threshold = sum(ratio < SMALL_DEFECT_THRESHOLD for ratio in ratios)
    axis.set_title("Distribuição da proporção de pixels anômalos")
    axis.set_xlabel("Pixels marcados / pixels da imagem")
    axis.set_ylabel("Quantidade de máscaras")
    axis.text(
        0.98,
        0.95,
        f"{len(ratios)} máscaras; {below_threshold} abaixo de 1%"
        + (f"; {zero_count} sem pixels marcados" if zero_count else ""),
        ha="right",
        va="top",
        transform=axis.transAxes,
    )
    if positive_ratios:
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def analyze_dataset(
    data_root: Path, output_dir: Path, dpi: int = 300
) -> Tuple[Path, Path, Path, Path]:
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = _categories(data_root)
    profile_rows, test_images, image_sides = _inspect_image_profiles(categories)
    mask_rows, ratios_by_category = _inspect_masks(categories, test_images)
    if not mask_rows:
        raise ValueError(f"No ground-truth masks found in {data_root}")
    summary_rows = _summary_rows(categories, mask_rows, ratios_by_category)

    profile_path = output_dir / "image_profiles_by_category.csv"
    mask_path = output_dir / "mask_anomaly_ratios.csv"
    summary_path = output_dir / "anomaly_summary_by_category.csv"
    histogram_path = output_dir / "anomaly_ratio_histogram.png"
    _write_csv(profile_path, profile_rows)
    _write_csv(mask_path, mask_rows)
    _write_csv(summary_path, summary_rows)
    _save_histogram(histogram_path, mask_rows, dpi)

    all_ratios = [float(row["anomaly_proportion"]) for row in mask_rows]
    percentile_5 = float(np.percentile(all_ratios, 5))
    minimum_ratio = min(all_ratios)
    minimum_native_side = min(
        side for sides in image_sides.values() for side in sides
    )
    target_pixels = 64
    required_side = (
        math.ceil(math.sqrt(target_pixels / minimum_ratio))
        if minimum_ratio > 0
        else minimum_native_side
    )
    required_side = 32 * math.ceil(required_side / 32)
    candidate_side = min(max(512, required_side), minimum_native_side)
    estimated_minimum_pixels = minimum_ratio * candidate_side * candidate_side
    smallest_mask = min(
        mask_rows, key=lambda row: float(row["anomaly_proportion"])
    )
    print(f"Categories: {len(categories)}")
    print(f"Images profiled by header: {sum(row['image_count'] for row in profile_rows)}")
    print(f"Masks analyzed: {len(mask_rows)}")
    print(
        f"Image-mask size mismatches: "
        f"{sum(not bool(row['image_mask_size_match']) for row in mask_rows)}"
    )
    print(f"5th percentile anomaly proportion: {percentile_5:.8g}")
    print(f"Smallest anomaly proportion: {minimum_ratio:.8g}")
    print(
        f"Input resolution candidate: {candidate_side}x{candidate_side} "
        f"(about {estimated_minimum_pixels:.0f} marked pixels for the smallest "
        f"mask; 256x256 would retain about "
        f"{minimum_ratio * 256 * 256:.0f} by area scaling)"
    )
    print(f"Smallest mask: {smallest_mask['mask_path']} ({smallest_mask['category']})")
    print(f"Profile table: {profile_path}")
    print(f"Per-mask ratios: {mask_path}")
    print(f"Category summary: {summary_path}")
    print(f"Histogram: {histogram_path}")
    return profile_path, mask_path, summary_path, histogram_path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Analyze MVTec image profiles and ground-truth mask sizes."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=script_dir,
        help="MVTec AD root directory (defaults to this script's directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "analysis" / "dataset_profile",
        help="Directory for CSV tables and the histogram.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    analyze_dataset(args.data_root, args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
