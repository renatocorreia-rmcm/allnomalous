"""Train the complete 15-category model using only normal train/validation splits."""

import hashlib
from statistics import mean

import os, subprocess, sys
from pathlib import Path
REPO_URL = "https://github.com/renatocorreia-rmcm/allnomalous.git"
REPO_DIR = Path(sys.argv[1]).resolve()
assert (REPO_DIR / "train_split.csv").is_file()
assert (REPO_DIR / "val_split.csv").is_file()
assert (REPO_DIR / "bottle/train/good/000.png").is_file()
print("Codigo, splits e imagens disponiveis em", REPO_DIR)

DRIVE_DIR = Path(sys.argv[2]).resolve()
DRIVE_DIR.mkdir(parents=True, exist_ok=True)

# Ambiente ja preparado no Apuana.
import sys
from pathlib import Path
sys.path.insert(0, str(REPO_DIR))
import torch
import torchvision
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Ative uma GPU no job Slurm antes de continuar.")
print("GPU:", torch.cuda.get_device_name(0))
free_bytes, total_bytes = torch.cuda.mem_get_info()
print(f"GPU memory free/total: {free_bytes / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB")


# Configuracao registrada no checkpoint. Nao a altere ao retomar um treino.
import random
import numpy as np

CONFIG = {
    "image_size": 512,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "effective_batch_size": 8,
    "learning_rate": 3e-4,
    "decoder_hidden_dim": 128,
    "decoder_num_encoder_layers": 2,
    "decoder_num_decoder_layers": 2,
    "weight_decay": 1e-4,
    "epochs_max": 1000,
    "early_stopping_window": 25,
    "early_stopping_min_delta": 0.0001,
    "early_stopping_patience": 70,
    "checkpoint_every_epochs": 1,
    "seed": 42,
    "num_workers": 8,
    "mixed_precision": True,
    "model": "WideResNet50_2_IMAGENET1K_V2_layer1-layer3 + complete ReconstructionDecoder",
    "jitter": True,
    "jitter_scale": 20.0,
    "jitter_probability": 1.0,
    "neighbor_mask": True,
    "neighbor_size": 7,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
torch.cuda.manual_seed_all(CONFIG["seed"])
torch.backends.cudnn.benchmark = False
assert torch.cuda.is_available(), "GPU CUDA indisponivel"
DEVICE = torch.device("cuda")



# Criar loaders combinados e confirmar que os splits contem 15 categorias.
from torch.utils.data import DataLoader
from src.mvtec_dataset import MVTecTrainDataset, MVTecValidationDataset

train_dataset = MVTecTrainDataset(REPO_DIR, image_size=CONFIG["image_size"])
val_dataset = MVTecValidationDataset(REPO_DIR, image_size=CONFIG["image_size"])
train_generator = torch.Generator().manual_seed(CONFIG["seed"])
train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
    generator=train_generator,
    persistent_workers=CONFIG["num_workers"] > 0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    num_workers=CONFIG["num_workers"],
    pin_memory=True,
    persistent_workers=CONFIG["num_workers"] > 0,
)

train_categories = sorted({path.parts[-4] for path in train_dataset.image_paths})
val_categories = sorted({path.parts[-4] for path in val_dataset.image_paths})
assert len(train_categories) == 15 and len(val_categories) == 15
print(f"Train: {len(train_dataset)} normal images / {len(train_categories)} mixed categories")
print(f"Validation: {len(val_dataset)} normal images / {len(val_categories)} mixed categories")
print("The datasets return images only; category is not passed to the model.")
print("Official test data is not instantiated or read in this script.")

from src.feature_extractor import WideResNetFeatureExtractor

extractor = WideResNetFeatureExtractor(pretrained=True).to(DEVICE).eval()
assert all(not parameter.requires_grad for parameter in extractor.parameters())

# Inicializar extrator e decoder completo. O extrator fica congelado.
from src.reconstruction_decoder import ReconstructionDecoder, reconstruction_loss

decoder = ReconstructionDecoder(
    in_channels=1792,
    hidden_dim=CONFIG["decoder_hidden_dim"],
    num_heads=4,
    num_encoder_layers=CONFIG["decoder_num_encoder_layers"],
    num_decoder_layers=CONFIG["decoder_num_decoder_layers"],
    feedforward_dim=CONFIG["decoder_hidden_dim"] * 4,
    dropout=0.1,
    base_grid_size=(32, 32),
    neighbor_size=CONFIG["neighbor_size"],
    jitter_scale=CONFIG["jitter_scale"],
    jitter_probability=CONFIG["jitter_probability"],
).to(DEVICE)
optimizer = torch.optim.AdamW(
    decoder.parameters(),
    lr=CONFIG["learning_rate"],
    betas=(0.9, 0.999),
    weight_decay=CONFIG["weight_decay"],
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG["epochs_max"]
)
scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["mixed_precision"])
total_parameters = sum(parameter.numel() for parameter in extractor.parameters()) + sum(
    parameter.numel() for parameter in decoder.parameters()
)
trainable_parameters = sum(
    parameter.numel() for parameter in list(extractor.parameters()) + list(decoder.parameters())
    if parameter.requires_grad
)
print(f"Model parameters: {total_parameters:,} total / {trainable_parameters:,} trainable")
print("Frozen extractor and complete decoder initialized.")

# Checkpoints atomicos no armazenamento persistente.
import csv
import json
import math
import os
import time

LAST_PATH = DRIVE_DIR / "last_checkpoint.pt"
SELECTED_PATH = DRIVE_DIR / "selected_decoder.pt"
HISTORY_PATH = DRIVE_DIR / "loss_history.csv"
META_PATH = DRIVE_DIR / "training_config.json"
metadata = {
    "config": CONFIG,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "repo_url": REPO_URL,
    "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip(),
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "total_parameters": total_parameters,
    "trainable_parameters": trainable_parameters,
}
if META_PATH.exists():
    previous = json.loads(META_PATH.read_text(encoding="utf-8"))
    if any(previous[key] != metadata[key] for key in ("config", "repo_commit", "script_sha256")):
        raise ValueError("Training metadata differs from this run.")
else:
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

history = []
start_epoch = 0
best_window_loss = math.inf
selected_epoch = 0
total_train_seconds = 0.0

def _atomic_torch_save(payload, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)

def _save_history():
    if not history:
        return
    fields = ["epoch", "train_loss", "val_loss", "epoch_seconds", "learning_rate"]
    temporary_csv = HISTORY_PATH.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary_csv, HISTORY_PATH)

def save_epoch_checkpoint(epoch):
    payload = {
        "epoch": epoch,
        "config": CONFIG,
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_window_loss": best_window_loss,
        "selected_epoch": selected_epoch,
        "history": history,
        "total_train_seconds": total_train_seconds,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "loader_generator_state": train_generator.get_state(),
    }
    _atomic_torch_save(payload, LAST_PATH)
    _save_history()

def load_latest_checkpoint():
    global history, start_epoch, best_window_loss, selected_epoch, total_train_seconds
    if not LAST_PATH.exists():
        print("Starting a new run.")
        return
    try:
        checkpoint = torch.load(LAST_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(LAST_PATH, map_location=DEVICE)
    if checkpoint["config"] != CONFIG:
        raise ValueError("Checkpoint configuration differs from this run.")
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_window_loss = checkpoint["best_window_loss"]
    selected_epoch = checkpoint["selected_epoch"]
    history = checkpoint["history"]
    total_train_seconds = checkpoint["total_train_seconds"]
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    train_generator.set_state(checkpoint["loader_generator_state"].cpu())
    print(f"Resumed after epoch {start_epoch}; selected epoch={selected_epoch}")

def run_epoch(epoch):
    decoder.train()
    total_loss = 0.0
    seen = 0
    optimizer.zero_grad(set_to_none=True)
    total_batches = len(train_loader)
    epoch_started = time.perf_counter()

    for batch_index, images in enumerate(train_loader):
        images = images.to(DEVICE, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", enabled=CONFIG["mixed_precision"]):
            targets = extractor(images)
        with torch.autocast("cuda", enabled=CONFIG["mixed_precision"]):
            reconstruction = decoder(targets)
            loss = reconstruction_loss(targets, reconstruction)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

        group_start = (batch_index // CONFIG["gradient_accumulation_steps"]) * CONFIG["gradient_accumulation_steps"]
        group_size = min(CONFIG["gradient_accumulation_steps"], total_batches - group_start)
        scaler.scale(loss / group_size).backward()
        batch_count = images.shape[0]
        total_loss += loss.item() * batch_count
        seen += batch_count

        step_now = (
            (batch_index + 1) % CONFIG["gradient_accumulation_steps"] == 0
            or batch_index + 1 == total_batches
        )
        if step_now:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    train_loss = total_loss / seen
    decoder.eval()
    validation_total = 0.0
    validation_seen = 0
    with torch.no_grad():
        for images in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", enabled=CONFIG["mixed_precision"]):
                targets = extractor(images)
                reconstruction = decoder(targets)
                loss = reconstruction_loss(targets, reconstruction)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
            batch_count = images.shape[0]
            validation_total += loss.item() * batch_count
            validation_seen += batch_count

    val_loss = validation_total / validation_seen
    scheduler.step()
    epoch_seconds = time.perf_counter() - epoch_started
    return train_loss, val_loss, epoch_seconds



load_latest_checkpoint()
window = CONFIG["early_stopping_window"]
min_delta = CONFIG["early_stopping_min_delta"]
patience = CONFIG["early_stopping_patience"]
for epoch in range(start_epoch + 1, CONFIG["epochs_max"] + 1):
    if selected_epoch and epoch - 1 - selected_epoch >= patience:
        break

    current_lr = optimizer.param_groups[0]["lr"]
    train_loss, val_loss, epoch_seconds = run_epoch(epoch)
    total_train_seconds += epoch_seconds
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch_seconds": epoch_seconds,
        "learning_rate": current_lr,
    })
    if len(history) >= window:
        window_loss = mean(row["val_loss"] for row in history[-window:])
        if window_loss <= best_window_loss - min_delta:
            best_window_loss = window_loss
            selected_epoch = epoch
            _atomic_torch_save({
                "epoch": epoch,
                "config": CONFIG,
                "validation_loss": val_loss,
                "validation_window_loss": window_loss,
                "decoder_state_dict": decoder.state_dict(),
            }, SELECTED_PATH)
    save_epoch_checkpoint(epoch)
    print(
        f"epoch {epoch}/{CONFIG['epochs_max']} | train MSE={train_loss:.6f} | "
        f"val MSE={val_loss:.6f} | selected epoch={selected_epoch} | "
        f"{epoch_seconds / 60:.1f} min",
        flush=True,
    )

last_epoch = history[-1]["epoch"]
if selected_epoch and last_epoch - selected_epoch >= patience:
    print(f"Early stopping at epoch {last_epoch}. Selected checkpoint: epoch {selected_epoch}.", flush=True)
else:
    print(f"Training complete: {last_epoch} epochs. Selected checkpoint: epoch {selected_epoch}.", flush=True)
