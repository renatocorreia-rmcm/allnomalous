import argparse
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple

from src.dataset_paths import DEFAULT_DATA_ROOT, resolve_data_root


DEFAULT_SEED = 42
DEFAULT_VALIDATION_FRACTION = 0.1
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def create_splits(
    data_root: Path,
    seed: int = DEFAULT_SEED,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
) -> List[Tuple[str, int, int, int]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    data_root = resolve_data_root(data_root)
    categories = sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir()
        and (path / "train" / "good").is_dir()
        and (path / "test").is_dir()
    )
    if not categories:
        raise FileNotFoundError(f"No MVTec categories found in {data_root}.")

    rng = random.Random(seed)
    train_rows: List[Tuple[str, str]] = []
    validation_rows: List[Tuple[str, str]] = []
    counts: List[Tuple[str, int, int, int]] = []

    for category in categories:
        good_dir = category / "train" / "good"
        image_paths = sorted(
            path.relative_to(data_root).as_posix()
            for path in good_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(image_paths) < 2:
            raise ValueError(
                f"Category {category.name} needs at least two normal training images."
            )

        rng.shuffle(image_paths)
        validation_count = min(
            len(image_paths) - 1,
            max(1, math.ceil(len(image_paths) * validation_fraction)),
        )
        validation_paths = image_paths[:validation_count]
        train_paths = image_paths[validation_count:]

        train_rows.extend((category.name, path) for path in train_paths)
        validation_rows.extend((category.name, path) for path in validation_paths)
        counts.append(
            (category.name, len(image_paths), len(train_paths), len(validation_paths))
        )

    train_paths = {path for _, path in train_rows}
    validation_paths = {path for _, path in validation_rows}
    if train_paths & validation_paths:
        raise RuntimeError("Training and validation splits overlap.")

    for filename, rows in (
        ("train_split.csv", train_rows),
        ("val_split.csv", validation_rows),
    ):
        with (data_root / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("category", "path"))
            writer.writerows(rows)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reproducible normal-only MVTec AD training splits."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="MVTec AD root directory (defaults to dataset/ in this checkout).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
    )
    args = parser.parse_args()

    counts = create_splits(args.data_root.resolve(), args.seed, args.validation_fraction)
    print("Categoria | Normais originais | Treino | Validacao")
    for category, original, train_count, validation_count in counts:
        print(f"{category} | {original} | {train_count} | {validation_count}")
    print(
        f"TOTAL | {sum(row[1] for row in counts)} | "
        f"{sum(row[2] for row in counts)} | {sum(row[3] for row in counts)}"
    )
    print(f"Seed: {args.seed}; validacao: {args.validation_fraction:.0%}")


if __name__ == "__main__":
    main()
