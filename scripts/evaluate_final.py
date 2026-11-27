"""Run the frozen, one-time final test for baseline and full MVTec models."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.anomaly_metrics import evaluate_model
from src.dataset_paths import DEFAULT_DATA_ROOT
from src.feature_extractor import WideResNetFeatureExtractor
from src.mvtec_dataset import create_test_dataloader
from src.reconstruction_decoder import ReconstructionDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline and full checkpoints once on the official MVTec "
            "test split. This command loads test images and ground-truth masks."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-training-metadata", type=Path, required=True)
    parser.add_argument("--complete-checkpoint", type=Path, required=True)
    parser.add_argument("--complete-training-metadata", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cpu", "cuda"),
    )
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required acknowledgement: this command reads the official test split.",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--expected-complete-epoch", type=int)
    parser.add_argument("--expected-complete-sha256")
    parser.add_argument("--expected-baseline-sha256")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_torch_file(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _read_model_inputs(
    checkpoint_path: Path, metadata_path: Path, variant: str
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{variant} checkpoint not found: {checkpoint_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{variant} training metadata not found: {metadata_path}")
    checkpoint = _load_torch_file(checkpoint_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    configuration = checkpoint.get("config")
    metadata_configuration = metadata.get("config")
    if not isinstance(configuration, dict) or configuration != metadata_configuration:
        raise ValueError(
            f"{variant} checkpoint config does not match its training metadata."
        )

    if variant == "baseline":
        if configuration.get("jitter", False) or configuration.get("neighbor_mask", False):
            raise ValueError("The baseline checkpoint must disable jitter and neighbor masking.")
    elif not configuration.get("jitter", False) or not configuration.get(
        "neighbor_mask", False
    ):
        raise ValueError(
            "The complete checkpoint must enable both jitter and neighbor masking."
        )
    return checkpoint, metadata, configuration


def _build_decoder(configuration: Dict[str, object]) -> ReconstructionDecoder:
    hidden_dim = int(configuration.get("decoder_hidden_dim", 128))
    neighbor_size = configuration.get("neighbor_size")
    if neighbor_size is None and configuration.get("neighbor_mask", False):
        neighbor_size = 7
    jitter_scale = float(
        configuration.get("jitter_scale", 20.0 if configuration.get("jitter") else 0.0)
    )
    jitter_probability = float(
        configuration.get(
            "jitter_probability", 1.0 if configuration.get("jitter") else 0.0
        )
    )
    return ReconstructionDecoder(
        in_channels=1792,
        hidden_dim=hidden_dim,
        num_heads=int(configuration.get("decoder_num_heads", 4)),
        num_encoder_layers=int(configuration.get("decoder_num_encoder_layers", 2)),
        num_decoder_layers=int(configuration.get("decoder_num_decoder_layers", 2)),
        feedforward_dim=int(configuration.get("decoder_feedforward_dim", hidden_dim * 4)),
        dropout=float(configuration.get("decoder_dropout", 0.1)),
        base_grid_size=(32, 32),
        neighbor_size=int(neighbor_size) if neighbor_size is not None else None,
        jitter_scale=jitter_scale,
        jitter_probability=jitter_probability,
    )


def _save_prediction_archive(
    path: Path, artifacts: Dict[str, np.ndarray], metadata: Dict[str, object]
) -> None:
    maps = artifacts["anomaly_maps"]
    masks = artifacts["masks"]
    if maps.ndim != 3 or masks.shape != maps.shape:
        raise ValueError("Collected maps and masks must have matching (N, H, W) shapes.")
    packed_masks = np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            **({"image_paths": artifacts["image_paths"]} if "image_paths" in artifacts else {}),
            image_scores=artifacts["image_scores"],
            image_labels=artifacts["image_labels"],
            categories=artifacts["categories"],
            anomaly_maps=maps,
            packed_masks=packed_masks,
            mask_height=np.asarray(masks.shape[1], dtype=np.int32),
            mask_width=np.asarray(masks.shape[2], dtype=np.int32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(temporary_path, path)


def _write_metrics_csv(
    path: Path, model_results: Dict[str, Dict[str, object]]
) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "model",
        "category",
        "image_count",
        "pixel_count",
        "region_count",
        "image_auroc",
        "pixel_auroc",
        "aupro",
    ]
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for model_name, result in model_results.items():
            for category, metrics in result["per_category"].items():
                writer.writerow(
                    {
                        "model": model_name,
                        "category": category,
                        **metrics,
                    }
                )
            writer.writerow(
                {
                    "model": model_name,
                    "category": "macro",
                    "image_count": "",
                    "pixel_count": "",
                    "region_count": "",
                    "image_auroc": result["macro"]["image_auroc"],
                    "pixel_auroc": result["macro"]["pixel_auroc"],
                    "aupro": result["macro"]["aupro"],
                }
            )
    os.replace(temporary_path, path)


def _assert_frozen_repository() -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    changes = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
    ).strip()
    if changes:
        raise RuntimeError(
            "Final evaluation requires a clean, committed repository. "
            "Commit or move local changes before running."
        )
    return commit


def main() -> None:
    args = parse_args()
    if not args.confirm_final_test and not args.preflight:
        raise SystemExit(
            "Refusing to read the official test split without --confirm-final-test."
        )
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    frozen_commit = _assert_frozen_repository()
    baseline = _read_model_inputs(
        args.baseline_checkpoint, args.baseline_training_metadata, "baseline"
    )
    complete = _read_model_inputs(
        args.complete_checkpoint, args.complete_training_metadata, "complete"
    )
    baseline_checkpoint, baseline_metadata, baseline_config = baseline
    complete_checkpoint, complete_metadata, complete_config = complete
    image_size = int(baseline_config["image_size"])
    if int(complete_config["image_size"]) != image_size:
        raise ValueError("Baseline and complete checkpoints must use the same image size.")

    if args.expected_complete_epoch is not None and complete_checkpoint["epoch"] != args.expected_complete_epoch:
        raise ValueError("The complete checkpoint is not from the frozen epoch.")
    for path, expected in (
        (args.baseline_checkpoint, args.expected_baseline_sha256),
        (args.complete_checkpoint, args.expected_complete_sha256),
    ):
        if expected is not None and _sha256(path) != expected:
            raise ValueError(f"Checkpoint hash differs from frozen input: {path}")
    extractor = WideResNetFeatureExtractor(pretrained=True).to(args.device).eval()
    if args.preflight:
        image = torch.zeros((1, 3, image_size, image_size), device=args.device)
        with torch.inference_mode():
            features = extractor(image)
            for name, checkpoint, _, config in (("baseline", *baseline), ("complete", *complete)):
                decoder = _build_decoder(config).to(args.device).eval()
                decoder.load_state_dict(checkpoint["decoder_state_dict"])
                if not torch.isfinite(decoder(features)).all():
                    raise ValueError(f"Non-finite {name} preflight output")
                print(f"{name} checkpoint epoch {checkpoint['epoch']}", flush=True)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_marker = args.output_dir / "final_test_run.json"

    run_metadata = {
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_code_commit": frozen_commit,
        "runtime": {"torch": torch.__version__, "numpy": np.__version__,
                    "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"},
        "protocol": {
            "image_size": image_size,
            "batch_size": args.batch_size,
            "score_method": "topk_mean",
            "top_fraction": 0.01,
            "histogram_bins": 2048,
            "aupro_max_fpr": 0.3,
            "map_smoothing_kernel_size": None,
            "official_test_used": True,
        },
        "models": {
            "baseline": {
                "checkpoint": str(args.baseline_checkpoint.resolve()),
                "checkpoint_sha256": _sha256(args.baseline_checkpoint),
                "epoch": baseline_checkpoint["epoch"],
                "training_commit": baseline_metadata.get("repo_commit"),
                "config": baseline_config,
            },
            "complete": {
                "checkpoint": str(args.complete_checkpoint.resolve()),
                "checkpoint_sha256": _sha256(args.complete_checkpoint),
                "epoch": complete_checkpoint["epoch"],
                "training_commit": complete_metadata.get("repo_commit"),
                "config": complete_config,
            },
        },
    }
    temporary_marker = run_marker.with_suffix(".json.tmp")
    with run_marker.open("x", encoding="utf-8") as marker:
        json.dump(run_metadata, marker, indent=2)

    loader = create_test_dataloader(
        args.data_root,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
    )
    paths = np.asarray([str(path.relative_to(loader.dataset.data_root))
                        for path, _, _, _ in loader.dataset.samples])
    model_results = {}
    for model_name, checkpoint, _, configuration in (
        ("baseline", baseline_checkpoint, baseline_metadata, baseline_config),
        ("complete", complete_checkpoint, complete_metadata, complete_config),
    ):
        print(f"Evaluating {model_name} on {len(paths)} official images.", flush=True)
        decoder = _build_decoder(configuration).to(args.device).eval()
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        result = evaluate_model(
            extractor,
            decoder,
            loader,
            device=torch.device(args.device),
            score_method="topk_mean",
            top_fraction=0.01,
            histogram_bins=2048,
            aupro_max_fpr=0.3,
            collect_artifacts=True,
        )
        artifacts = result.pop("artifacts")
        artifacts["image_paths"] = paths
        artifact_metadata = {
            "model": model_name,
            "evaluation_code_commit": frozen_commit,
            "training_commit": run_metadata["models"][model_name]["training_commit"],
            "checkpoint_sha256": run_metadata["models"][model_name][
                "checkpoint_sha256"
            ],
            "config": configuration,
            "settings": result["settings"],
        }
        _save_prediction_archive(
            args.output_dir / f"{model_name}_predictions.npz",
            artifacts,
            artifact_metadata,
        )
        model_results[model_name] = result
        del decoder, artifacts
        if args.device == "cuda":
            torch.cuda.empty_cache()

    _write_metrics_csv(args.output_dir / "final_metrics.csv", model_results)
    run_metadata["status"] = "completed"
    run_metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_metadata["output_files"] = [
        "baseline_predictions.npz",
        "complete_predictions.npz",
        "final_metrics.csv",
    ]
    temporary_marker.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    os.replace(temporary_marker, run_marker)
    print(f"Final metrics: {args.output_dir / 'final_metrics.csv'}")
    print(f"Predictions and maps: {args.output_dir}")


if __name__ == "__main__":
    main()
