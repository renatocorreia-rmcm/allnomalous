"""Run a pretrained feature-extractor batch and report latency and GPU memory."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_extractor import (
    WideResNetFeatureExtractor,
    benchmark_feature_extractor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available.")
    args = parser.parse_args()

    extractor = WideResNetFeatureExtractor(pretrained=True)
    result = benchmark_feature_extractor(
        extractor,
        input_shape=(args.batch_size, 3, args.image_size, args.image_size),
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    for name, value in result.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
