import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image

from src.dataset_paths import DEFAULT_DATA_ROOT, resolve_data_root


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFECT_EXAMPLES = (
    ("bottle", "broken_large"),
    ("carpet", "cut"),
    ("transistor", "damaged_case"),
)


def _images_in(directory: Path) -> list:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    images = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {directory}")
    return images


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def plot_example(
    data_root: Path,
    category: str,
    defect_type: str,
    image_name: Optional[str] = None,
    axes: Optional[Tuple[Axes, Axes, Axes]] = None,
) -> Tuple[Figure, Tuple[Axes, Axes, Axes]]:
    """Plot a normal image, a defective image, and its ground-truth overlay."""
    data_root = resolve_data_root(data_root)
    normal_path = _images_in(data_root / category / "train" / "good")[0]
    defective_dir = data_root / category / "test" / defect_type
    defective_paths = _images_in(defective_dir)
    defective_path = (
        defective_dir / image_name if image_name else defective_paths[0]
    )
    if not defective_path.is_file():
        raise FileNotFoundError(f"Defective image not found: {defective_path}")
    mask_path = (
        data_root
        / category
        / "ground_truth"
        / defect_type
        / f"{defective_path.stem}_mask.png"
    )
    if not mask_path.is_file():
        raise FileNotFoundError(f"Ground-truth mask not found: {mask_path}")

    normal = _read_rgb(normal_path)
    defective = _read_rgb(defective_path)
    mask = _read_mask(mask_path)
    if defective.shape[:2] != mask.shape:
        raise ValueError(
            f"Image and mask dimensions differ: {defective_path} "
            f"({defective.shape[1]}x{defective.shape[0]}) vs "
            f"{mask_path} ({mask.shape[1]}x{mask.shape[0]})"
        )

    if axes is None:
        figure, axis_array = plt.subplots(1, 3, figsize=(12, 4))
        axes = tuple(axis_array)
    else:
        figure = axes[0].figure

    titles = (
        "Normal",
        f"Defeituosa: {defect_type.replace('_', ' ')}",
        "Defeito + máscara",
    )
    for axis, image, title in zip(axes, (normal, defective, defective), titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask > 0] = (0.9, 0.12, 0.16, 0.5)
    axes[2].imshow(overlay)
    figure.suptitle(
        f"{category.replace('_', ' ').title()} — "
        f"{defect_type.replace('_', ' ')}",
        fontsize=14,
    )
    return figure, axes


def _save_category_figure(data_root: Path, output_dir: Path, dpi: int) -> Path:
    categories = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir() and (path / "train" / "good").is_dir()
    )
    if not categories:
        raise FileNotFoundError(f"No MVTec categories found in {data_root}")

    columns = 5
    rows = (len(categories) + columns - 1) // columns
    figure, axis_array = plt.subplots(
        rows, columns, figsize=(15, 4 * rows), squeeze=False
    )
    for axis, category in zip(axis_array.flat, categories):
        image_path = _images_in(category / "train" / "good")[0]
        axis.imshow(_read_rgb(image_path))
        axis.set_title(category.name.replace("_", " "), fontsize=10)
        axis.axis("off")
    for axis in list(axis_array.flat)[len(categories) :]:
        axis.axis("off")

    figure.suptitle(
        "Imagens normais de treino por categoria — MVTec AD",
        fontsize=16,
        y=0.99,
    )
    figure.text(
        0.5,
        0.015,
        f"Figura 1 — Uma imagem normal de treino (good) para cada uma das "
        f"{len(categories)} categorias do MVTec AD.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.4)
    path = output_dir / "mvtec_normal_15_categories.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _save_defect_figure(
    data_root: Path,
    output_dir: Path,
    category: str,
    defect_type: str,
    figure_number: int,
    dpi: int,
) -> Path:
    figure, axes = plot_example(data_root, category, defect_type)
    figure.subplots_adjust(top=0.78, bottom=0.2, wspace=0.04)
    figure.legend(
        handles=[
            Patch(facecolor="#e61f29", alpha=0.5, label="Região anômala (ground truth)")
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.09),
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.025,
        f"Figura {figure_number} — Imagem normal de treino, imagem de teste com "
        f"defeito {defect_type.replace('_', ' ')} e máscara ground truth sobreposta "
        f"em vermelho ({category.replace('_', ' ')}).",
        ha="center",
        fontsize=9,
    )
    path = output_dir / f"mvtec_{category}_{defect_type}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def generate_figures(data_root: Path, output_dir: Path, dpi: int = 300) -> list:
    """Generate the overview and three representative defect figures."""
    data_root = resolve_data_root(data_root)
    output_dir = Path(output_dir).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [_save_category_figure(data_root, output_dir, dpi)]
    for index, (category, defect_type) in enumerate(DEFECT_EXAMPLES, start=2):
        paths.append(
            _save_defect_figure(
                data_root, output_dir, category, defect_type, index, dpi
            )
        )
    return paths


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate high-resolution MVTec AD example figures."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="MVTec AD root directory (defaults to dataset/ in this checkout).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "figures" / "dataset_examples",
        help="Directory for generated PNG files.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    for path in generate_figures(args.data_root, args.output_dir, args.dpi):
        print(path)


if __name__ == "__main__":
    main()
