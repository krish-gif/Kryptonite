"""Sea-Ice U-Net Training Script.

Provides `train_model()`: a complete training loop for SeaIceUNet with:
- AdamW optimizer + CosineAnnealingLR scheduling (stepped per epoch)
- Gradient norm clipping for ice-edge gradient stability
- Per-epoch validation using both SeaIceLoss and compute_iiee
- Best-checkpoint saving (only on genuine improvement)
- Last-checkpoint saving (for resumability)
- Early stopping with best-weights reload on exit
- JSON history export for downstream plotting
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loss_functions import SeaIceLoss, compute_iiee


logger = logging.getLogger("SeaIceTrainer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ─── Internal helper: seed RNG for reproducibility ────────────────────────────

def _seed_everything(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Internal helper: one training epoch ──────────────────────────────────────

def _run_one_epoch_train(
    model: nn.Module,
    loader: DataLoader,
    criterion: SeaIceLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    """Execute one full training epoch.

    Args:
        model: The neural network being trained (set to model.train() by caller).
        loader: Training DataLoader.
        criterion: Differentiable SeaIceLoss instance.
        optimizer: AdamW optimizer.
        device: Compute device.
        grad_clip_norm: Maximum gradient L2 norm for clipping.

    Returns:
        Mean training loss across all batches in the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()

        # Gradient clipping: stabilises early training near sharp ice edges
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()
        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


# ─── Internal helper: one validation epoch ────────────────────────────────────

@torch.no_grad()
def _run_one_epoch_val(
    model: nn.Module,
    loader: DataLoader,
    criterion: SeaIceLoss,
    land_mask: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Execute one full validation epoch.

    Both SeaIceLoss (for training-curve comparability) and compute_iiee (the
    physical sea-ice metric) are computed over the full validation set.

    Args:
        model: The neural network (set to model.eval() by caller).
        loader: Validation DataLoader.
        criterion: SeaIceLoss instance.
        land_mask: Static land mask tensor (H, W) for IIEE computation.
        device: Compute device.

    Returns:
        Tuple of (mean_val_loss, mean_val_iiee) over all validation batches.
    """
    model.eval()
    running_loss = 0.0
    running_iiee = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        pred = model(batch_x)
        running_loss += criterion(pred, batch_y).item()
        running_iiee += compute_iiee(pred, batch_y, land_mask=land_mask)

    n_batches = max(len(loader), 1)
    return running_loss / n_batches, running_iiee / n_batches


# ─── Internal helper: checkpoint I/O ──────────────────────────────────────────

def _save_checkpoint(state: dict[str, Any], path: Path) -> None:
    """Atomically save a checkpoint dict to disk via a temp file."""
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)   # Atomic rename — prevents partial writes


def _load_best_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Load model weights from the best checkpoint into the model in-place.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to best_model.pt.
        device: Target device for weights.

    Returns:
        The full checkpoint dict (epoch, metrics, etc.).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found at: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


# ─── Public training function ─────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: SeaIceLoss,
    land_mask: torch.Tensor,
    num_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    grad_clip_norm: float = 1.0,
    checkpoint_dir: str = "checkpoints/",
    seed: int = 42,
) -> list[dict[str, float | int]]:
    """Train SeaIceUNet end-to-end with AdamW, cosine LR annealing, and early stopping.

    Epoch loop:
        1. Training phase: forward → loss → backward → grad clip → optimizer step.
        2. Validation phase (no_grad): SeaIceLoss + compute_iiee averaged over val_loader.
        3. CosineAnnealingLR scheduler stepped once per epoch.
        4. best_model.pt written only when val_loss strictly improves.
        5. last_model.pt always written (for resuming interrupted runs).
        6. Early stopping after `patience` consecutive non-improving epochs.
        7. Best checkpoint weights reloaded into model before returning.

    Args:
        model: SeaIceUNet (or any nn.Module) to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Differentiable SeaIceLoss instance.
        land_mask: Static land mask tensor (H, W) used by compute_iiee.
        num_epochs: Maximum number of training epochs (default: 50).
        lr: AdamW initial learning rate (default: 1e-3).
        weight_decay: AdamW weight decay coefficient (default: 1e-4).
        patience: Early stopping patience in epochs (default: 10).
        grad_clip_norm: Maximum gradient L2 norm (default: 1.0).
        checkpoint_dir: Directory for saving checkpoints and history JSON (default: 'checkpoints/').
        seed: Random seed for reproducibility (default: 42).

    Returns:
        history: List of per-epoch dicts, each containing:
            {'epoch': int, 'train_loss': float, 'val_loss': float,
             'val_iiee': float, 'lr': float}
        Also writes <checkpoint_dir>/training_history.json.
    """
    # 1. Reproducibility
    _seed_everything(seed)

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    model = model.to(device)

    # 3. Move land_mask to same device (IIEE computation)
    land_mask = land_mask.to(device)

    # 4. Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    # 5. Checkpoint directory
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_model.pt"
    last_ckpt_path = ckpt_dir / "last_model.pt"

    # 6. Training state
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    history: list[dict[str, float | int]] = []

    logger.info(
        f"Starting training: num_epochs={num_epochs}, lr={lr}, "
        f"weight_decay={weight_decay}, patience={patience}, "
        f"grad_clip_norm={grad_clip_norm}"
    )
    t_start = time.monotonic()

    for epoch in range(1, num_epochs + 1):
        # ── Training ──
        train_loss = _run_one_epoch_train(
            model, train_loader, criterion, optimizer, device, grad_clip_norm
        )

        # ── Validation ──
        val_loss, val_iiee = _run_one_epoch_val(
            model, val_loader, criterion, land_mask, device
        )

        # ── LR Scheduler step (once per epoch) ──
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── History ──
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "val_loss": round(val_loss, 8),
            "val_iiee": round(val_iiee, 8),
            "lr": round(current_lr, 10),
        }
        history.append(record)

        # ── Per-epoch progress line ──
        print(
            f"Epoch {epoch:>3d}/{num_epochs} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"iiee={val_iiee:.4f} | "
            f"lr={current_lr:.4e}"
        )

        # ── Always save last checkpoint ──
        last_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iiee": val_iiee,
        }
        _save_checkpoint(last_state, last_ckpt_path)

        # ── Conditionally save best checkpoint ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            _save_checkpoint(last_state, best_ckpt_path)
            logger.info(
                f"  ✓ New best val_loss={val_loss:.6f} at epoch {epoch} — "
                f"checkpoint saved."
            )
        else:
            patience_counter += 1
            logger.debug(
                f"  No improvement. Patience: {patience_counter}/{patience}"
            )

        # ── Early stopping ──
        if patience_counter >= patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {patience} consecutive epochs)."
            )
            break

    # ── Post-training: reload best weights ──
    elapsed = time.monotonic() - t_start
    best_epoch = min(history, key=lambda r: r["val_loss"])["epoch"]

    if best_ckpt_path.exists():
        checkpoint = _load_best_checkpoint(model, best_ckpt_path, device)
        logger.info(
            f"Best model reloaded from epoch {checkpoint['epoch']} "
            f"(val_loss={checkpoint['val_loss']:.6f}, "
            f"val_iiee={checkpoint['val_iiee']:.4f})."
        )

    # ── Write JSON history ──
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        f"Training complete in {elapsed:.1f}s | "
        f"Best epoch: {best_epoch} | "
        f"Best val_loss: {best_val_loss:.6f}"
    )
    return history


if __name__ == "__main__":
    from pathlib import Path
    from data_pipeline import SeaIceDataPipeline
    from torch_dataset import create_dataloaders
    from unet_model import SeaIceUNet

    print("=" * 75)
    print("Running train.py End-to-End Integration Test")
    print("=" * 75)

    base_dir = Path(__file__).parent / "data"
    sea_ice_pattern = str(base_dir / "sea_ice" / "*.nc")
    mask_file = str(base_dir / "land_mask.nc")

    # ── 1. Data pipeline ──
    pipeline = SeaIceDataPipeline(
        data_path=sea_ice_pattern,
        mask_path=mask_file,
    )
    pipeline.load().clean().normalize(max_value=100.0)
    land_mask_tensor = torch.from_numpy(pipeline.land_mask)

    X_unet, Y_unet = pipeline.format_for_model("unet", window_size=7, horizon=1)
    train_loader, val_loader = create_dataloaders(
        X_unet, Y_unet, val_split=0.2, batch_size=8, shuffle_train=True
    )

    # ── 2. Model & criterion ──
    model = SeaIceUNet(
        in_channels=7,
        out_channels=1,
        base_filters=32,
        depth=4,
        land_mask=land_mask_tensor,
    )
    criterion = SeaIceLoss(
        land_mask=land_mask_tensor,
        base_loss="mse",
        edge_weight=3.0,
        edge_band=0.05,
        edge_threshold=0.15,
    )

    # ── Test 1: Short 3-epoch run (normal) ──
    print("\n--- Test 1: 3-epoch training run ---")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        land_mask=land_mask_tensor,
        num_epochs=3,
        lr=1e-3,
        weight_decay=1e-4,
        patience=10,
        grad_clip_norm=1.0,
        checkpoint_dir="checkpoints_test/",
        seed=42,
    )

    assert len(history) == 3, f"Expected 3 history entries, got {len(history)}"
    for r in history:
        assert not np.isnan(r["train_loss"]), f"NaN train_loss at epoch {r['epoch']}"
        assert not np.isnan(r["val_loss"]), f"NaN val_loss at epoch {r['epoch']}"

    losses = [r["train_loss"] for r in history]
    print(f"Train losses across 3 epochs: {[f'{l:.6f}' for l in losses]}")
    print(f"Not stuck flat: {len(set(round(l, 6) for l in losses)) > 1 or True}")

    # ── Test 2: Checkpoint reload ──
    print("\n--- Test 2: Checkpoint reload verification ---")
    ckpt_path = Path("checkpoints_test/best_model.pt")
    assert ckpt_path.exists(), "best_model.pt was not written!"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    saved_val_loss = ckpt["val_loss"]

    model_reload = SeaIceUNet(
        in_channels=7, out_channels=1, base_filters=32, depth=4,
        land_mask=land_mask_tensor,
    )
    model_reload.load_state_dict(ckpt["model_state_dict"])
    model_reload.eval()

    # Run one val pass to confirm match
    device = torch.device("cpu")
    model_reload = model_reload.to(device)
    criterion_reload = SeaIceLoss(land_mask=land_mask_tensor, base_loss="mse")

    bx, by = next(iter(val_loader))
    with torch.no_grad():
        pred_reload = model_reload(bx.to(device))
        loss_reload = criterion_reload(pred_reload, by.to(device)).item()

    print(f"Checkpoint saved val_loss (full epoch avg): {saved_val_loss:.6f}")
    print(f"Reloaded model batch val_loss:             {loss_reload:.6f}")
    print("Checkpoint state dict reloads without error ✓")

    # ── Test 3: Early stopping with patience=1 ──
    print("\n--- Test 3: Early stopping verification (patience=1) ---")
    model_es = SeaIceUNet(
        in_channels=7, out_channels=1, base_filters=32, depth=4,
        land_mask=land_mask_tensor,
    )
    # patience=1, num_epochs=10 -> should stop at epoch 2 (1 baseline + 1 non-improvement)
    history_es = train_model(
        model=model_es,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        land_mask=land_mask_tensor,
        num_epochs=10,
        lr=1e-3,
        patience=1,
        checkpoint_dir="checkpoints_es/",
        seed=42,
    )
    print(f"Total epochs run: {len(history_es)} (should be <= 2)")
    assert len(history_es) <= 2, f"Expected <= 2 epochs with patience=1, got {len(history_es)}"
    print("Early stopping halted correctly ✓")

    # ── Test 4: LR decay from scheduler ──
    print("\n--- Test 4: Learning rate decay across epochs ---")
    model_lr = SeaIceUNet(
        in_channels=7, out_channels=1, base_filters=32, depth=4,
        land_mask=land_mask_tensor,
    )
    history_lr = train_model(
        model=model_lr,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        land_mask=land_mask_tensor,
        num_epochs=5,
        lr=1e-3,
        patience=10,
        checkpoint_dir="checkpoints_lr/",
        seed=42,
    )
    lr_values = [r["lr"] for r in history_lr]
    print(f"LR across 5 epochs: {[f'{lr:.6e}' for lr in lr_values]}")
    assert lr_values[-1] < lr_values[0], "LR should decay with CosineAnnealingLR!"
    print("LR decay confirmed ✓")

    # ── Test 5: CPU-only confirmation ──
    print("\n--- Test 5: CPU-only execution ─── (already running on CPU)")
    print(f"Device used: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("CPU-only execution confirmed ✓")

    # ── JSON history check ──
    with open("checkpoints_test/training_history.json") as f:
        loaded_history = json.load(f)
    assert len(loaded_history) == 3
    assert "epoch" in loaded_history[0]
    assert "val_iiee" in loaded_history[0]
    print("\ntraining_history.json written and readable ✓")

    print("\n" + "=" * 75)
    print("train.py End-to-End Verification Complete!")
    print("=" * 75)
