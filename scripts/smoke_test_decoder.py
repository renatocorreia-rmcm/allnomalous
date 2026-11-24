"""Run the complete decoder on a normal sample from one MVTec category."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_extractor import WideResNetFeatureExtractor
from src.mvtec_dataset import MVTecTrainDataset
from src.reconstruction_decoder import ReconstructionDecoder, reconstruction_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    dataset = MVTecTrainDataset(data_root)
    sample_index = next(
        (
            index
            for index, path in enumerate(dataset.image_paths)
            if path.parts[-4] == args.category
        ),
        None,
    )
    if sample_index is None:
        raise ValueError(f"No normal training image found for {args.category!r}.")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    image = dataset[sample_index].unsqueeze(0).to(device)
    extractor = WideResNetFeatureExtractor(pretrained=True).to(device).eval()
    decoder = ReconstructionDecoder(
        neighbor_size=7, jitter_scale=20, jitter_probability=1
    ).to(device).eval()

    with torch.no_grad():
        target = extractor(image)
        reconstruction = decoder(target)
        loss = reconstruction_loss(target, reconstruction)
    if not torch.isfinite(loss):
        raise RuntimeError("Decoder smoke test produced a non-finite loss.")
    print(f"category: {args.category}")
    print(f"device: {device}")
    print(f"feature shape: {tuple(target.shape)}")
    print(f"reconstruction shape: {tuple(reconstruction.shape)}")
    print(f"normal-sample reconstruction MSE: {loss.item():.6f}")


if __name__ == "__main__":
    main()
