"""Local Training Script for Sea-Ice U-Net on RTX 4050 Laptop GPU (6 GB VRAM).

This module provides a complete, memory-safe training loop that reuses the
existing data pipeline (`data_pipeline.py`, `torch_dataset.py`) and model code
(`unet_model.py`, `loss_functions.py`) without modifying them.

Key features (all required for fitting a U-Net in 6 GB, not optional):
  - batch_size = 1
  - Automatic Mixed Precision (torch.cuda.amp.autocast + GradScaler)
  - Gradient checkpointing on the U-Net bottleneck + decoder up-blocks
  - Spatial crop lever (crop_h × crop_w) as an extra memory knob
  - Sequence-length lever (window_size, horizon) as a last-resort knob
  - Dry-run memory gate: one forward+backward, prints peak VRAM, warns if close
  - Per-epoch numbered checkpoints (epoch_NNNN.pt), resume support
  - Train/val loss + estimated time-per-epoch printed each epoch

Cloud / Lightning AI path:
  Use `train.py` as before.  This script has zero imports from `train.py` and
  vice-versa; the two paths are fully independent.

Quick start
-----------
    python local_train.py                                  # default paths
    python local_train.py --crop_h 256 --crop_w 256       # spatial crop lever
    python local_train.py --window_size 5                  # shorten sequence
    python local_train.py --dry_run_only                   # memory budget only
    python local_train.py --num_epochs 3 --no_confirm      # scripted short run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from data_pipeline import SeaIceDataPipeline
from local_config import LocalTrainConfig
from loss_functions import SeaIceLoss, compute_iiee
from torch_dataset import create_dataloaders
from unet_model import SeaIceUNet


# ─── Logger ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("LocalTrainer")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ─── GPU Check ────────────────────────────────────────────────────────────────

def gpu_check() -> torch.device:
    """Print GPU info and hard-stop if CUDA is unavailable.

    Returns:
        torch.device("cuda") if a GPU is found.

    Raises:
        SystemExit: If CUDA is not available.
    """
    print("=" * 65)
    print("GPU CHECK")
    print("=" * 65)

    cuda_available = torch.cuda.is_available()
    print(f"  torch.cuda.is_available()  → {cuda_available}")

    if not cuda_available:
        print(
            "\n  ✗  CUDA is NOT available.  Training on CPU would be orders of"
            "\n     magnitude slower and will likely OOM.  Please ensure:"
            "\n       1. NVIDIA drivers are installed (nvidia-smi should work)"
            "\n       2. A CUDA-enabled PyTorch build is installed"
            "\n          e.g.  pip install torch --index-url https://download.pytorch.org/whl/cu121"
            "\n  Stopping."
        )
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_mem_bytes = props.total_memory
    total_mem_gb = total_mem_bytes / (1024 ** 3)

    print(f"  torch.cuda.get_device_name(0)  → '{device_name}'")
    print(f"  total_memory (bytes)           → {total_mem_bytes:,}")
    print(f"  total_memory (GB)              → {total_mem_gb:.2f} GB")
    print(f"  compute capability             → {props.major}.{props.minor}")
    print(f"  multiprocessor_count           → {props.multi_processor_count}")

    if "4050" not in device_name and "RTX" not in device_name:
        print(
            f"\n  ⚠  Expected 'RTX 4050 Laptop GPU' but found '{device_name}'."
            f"\n     Memory settings are tuned for 6 GB.  Proceed with caution"
            f"\n     on a different card — the dry-run gate will still protect you."
        )

    print("=" * 65)
    return torch.device("cuda")


# ─── Seed ─────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Spatial crop helper ──────────────────────────────────────────────────────

def apply_spatial_crop(
    X: np.ndarray,
    Y: np.ndarray,
    land_mask: np.ndarray,
    cfg: LocalTrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre-crop X, Y arrays and land_mask to (crop_h, crop_w) if configured.

    X shape for U-Net:   (N, window_size, H, W)
    Y shape for U-Net:   (N, 1, H, W)
    land_mask shape:     (H, W)

    If cfg.crop_h and cfg.crop_w are both None, returns inputs unchanged.

    Args:
        X: Model input array.
        Y: Target array.
        land_mask: 2-D land mask NumPy array.
        cfg: LocalTrainConfig instance.

    Returns:
        Cropped (X, Y, land_mask) arrays.

    Raises:
        ValueError: If the requested crop is larger than the grid.
    """
    if cfg.crop_h is None and cfg.crop_w is None:
        return X, Y, land_mask

    if cfg.crop_h is None or cfg.crop_w is None:
        raise ValueError(
            "Both crop_h and crop_w must be set together (or both left as None). "
            f"Got crop_h={cfg.crop_h}, crop_w={cfg.crop_w}."
        )

    # Spatial dims: last two axes for X (N, C, H, W) and Y (N, 1, H, W)
    H, W = X.shape[-2], X.shape[-1]
    ch, cw = cfg.crop_h, cfg.crop_w

    if ch > H or cw > W:
        raise ValueError(
            f"Crop ({ch}×{cw}) is larger than the grid ({H}×{W}). "
            "Reduce crop_h / crop_w or remove the crop lever."
        )

    # Centre crop offsets
    row_start = (H - ch) // 2
    col_start = (W - cw) // 2

    X_crop = X[..., row_start: row_start + ch, col_start: col_start + cw]
    Y_crop = Y[..., row_start: row_start + ch, col_start: col_start + cw]
    mask_crop = land_mask[row_start: row_start + ch, col_start: col_start + cw]

    logger.info(
        f"Spatial crop applied: ({H}×{W}) → ({ch}×{cw})  "
        f"[rows {row_start}:{row_start+ch}, cols {col_start}:{col_start+cw}]"
    )
    return X_crop, Y_crop, mask_crop


# ─── Training epoch ───────────────────────────────────────────────────────────

def _train_epoch_local(
    model: nn.Module,
    loader: DataLoader,
    criterion: SeaIceLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: LocalTrainConfig,
) -> float:
    """One AMP-wrapped training epoch with optional gradient checkpointing.

    Args:
        model: SeaIceUNet (must already be on device).
        loader: Training DataLoader (batch_size=1 recommended for 6 GB).
        criterion: SeaIceLoss instance.
        optimizer: AdamW optimiser.
        scaler: GradScaler for AMP loss scaling.
        device: CUDA device.
        cfg: LocalTrainConfig (use_amp, use_grad_checkpoint, grad_clip_norm).

    Returns:
        Mean training loss over all batches in the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # set_to_none=True saves a memset

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            if cfg.use_grad_checkpoint:
                pred = model.forward_checkpointed(batch_x)
            else:
                pred = model(batch_x)
            loss = criterion(pred, batch_y)

        # AMP backward
        scaler.scale(loss).backward()
        # Unscale before grad clip so clip operates on true gradient magnitudes
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


# ─── Validation epoch ─────────────────────────────────────────────────────────

@torch.no_grad()
def _val_epoch_local(
    model: nn.Module,
    loader: DataLoader,
    criterion: SeaIceLoss,
    land_mask: torch.Tensor,
    device: torch.device,
    cfg: LocalTrainConfig,
) -> tuple[float, float]:
    """One validation epoch (no grad, AMP for speed).

    Args:
        model: SeaIceUNet in eval mode.
        loader: Validation DataLoader.
        criterion: SeaIceLoss instance.
        land_mask: Static land mask tensor on device.
        device: CUDA device.
        cfg: LocalTrainConfig (use_amp flag).

    Returns:
        Tuple of (mean_val_loss, mean_val_iiee).
    """
    model.eval()
    running_loss = 0.0
    running_iiee = 0.0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            pred = model(batch_x)
            loss = criterion(pred, batch_y)

        running_loss += loss.item()
        running_iiee += compute_iiee(pred, batch_y, land_mask=land_mask)

    n = max(len(loader), 1)
    return running_loss / n, running_iiee / n


# ─── Dry-run memory gate ──────────────────────────────────────────────────────

def _dry_run(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: SeaIceLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    cfg: LocalTrainConfig,
) -> float:
    """Execute a single forward+backward pass and report peak VRAM usage.

    If the peak exceeds ``cfg.dry_run_warn_mb`` MB (default 5 500 MB), a
    warning is printed.  When ``cfg.no_confirm`` is False the user is asked
    for confirmation; when True the script exits automatically.

    Args:
        model: SeaIceUNet on device in train mode.
        train_loader: Training DataLoader.
        criterion: SeaIceLoss.
        optimizer: AdamW optimiser.
        scaler: GradScaler.
        device: CUDA device.
        cfg: LocalTrainConfig.

    Returns:
        Peak memory allocated in MB.
    """
    print("\n" + "─" * 65)
    print("DRY-RUN MEMORY CHECK  (1 forward + backward pass)")
    print("─" * 65)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    model.train()
    batch_x, batch_y = next(iter(train_loader))
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=cfg.use_amp):
        if cfg.use_grad_checkpoint:
            pred = model.forward_checkpointed(batch_x)
        else:
            pred = model(batch_x)
        loss = criterion(pred, batch_y)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
    scaler.step(optimizer)
    scaler.update()

    # Reset optimizer state so epoch 1 is clean
    optimizer.zero_grad(set_to_none=True)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    peak_mb = peak_bytes / (1024 ** 2)
    total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)

    print(f"  Peak VRAM used : {peak_mb:,.1f} MB  /  {total_mb:,.0f} MB total")
    print(f"  Headroom       : {total_mb - peak_mb:,.1f} MB")
    print(f"  AMP            : {'ON (fp16)' if cfg.use_amp else 'OFF (fp32)'}")
    print(f"  Grad checkpoint: {'ON' if cfg.use_grad_checkpoint else 'OFF'}")
    print(
        f"  Batch shape    : {tuple(batch_x.shape)}  "
        f"(batch={cfg.batch_size}, window={cfg.window_size}, "
        f"{'crop=' + str(cfg.crop_h) + 'x' + str(cfg.crop_w) if cfg.crop_h else 'full grid'})"
    )

    if peak_mb > cfg.dry_run_warn_mb:
        print(
            f"\n  ⚠  WARNING: Peak usage {peak_mb:,.0f} MB exceeds the safety"
            f" threshold of {cfg.dry_run_warn_mb:,.0f} MB."
            f"\n     With only {total_mb - peak_mb:,.0f} MB headroom, a real training"
            f"\n     run may OOM mid-epoch (CUDA OOM errors are not recoverable)."
            f"\n\n  Suggested remedies (pick one or combine):"
            f"\n    1.  --crop_h 256 --crop_w 256    (spatial crop lever)"
            f"\n    2.  --window_size 5               (shorter input sequence)"
            f"\n    3.  --base_filters 16             (smaller model)"
            f"\n    4.  --depth 3                     (shallower U-Net)"
        )
        if not cfg.no_confirm:
            answer = input("\n  Proceed anyway? [y/N]: ").strip().lower()
            if answer != "y":
                print("  Aborted by user.")
                sys.exit(0)
        else:
            print("  --no_confirm set: exiting to avoid a likely OOM crash.")
            sys.exit(1)
    else:
        print(f"\n  ✓  Memory looks safe — proceeding to full training.")

    print("─" * 65 + "\n")
    return peak_mb


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_local_checkpoint(
    state: dict[str, Any],
    ckpt_dir: Path,
    epoch: int,
    is_best: bool,
) -> None:
    """Save per-epoch checkpoint and optionally overwrite best_model.pt.

    Writes:
        epoch_NNNN.pt   — numbered checkpoint for this epoch (always)
        last_model.pt   — always overwritten (same content, for quick access)
        best_model.pt   — only when is_best=True

    All writes are atomic (write to .tmp then rename).

    Args:
        state: Checkpoint dict (model, optimizer, scheduler, scaler states + metadata).
        ckpt_dir: Directory to write checkpoints into.
        epoch: Current epoch number (1-indexed), used in filename.
        is_best: Whether to also overwrite best_model.pt.
    """
    def _atomic_save(obj: dict, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    epoch_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
    last_path = ckpt_dir / "last_model.pt"

    _atomic_save(state, epoch_path)
    _atomic_save(state, last_path)

    if is_best:
        _atomic_save(state, ckpt_dir / "best_model.pt")


def _try_resume(
    ckpt_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> tuple[int, float]:
    """Attempt to resume from the highest-numbered epoch_NNNN.pt checkpoint.

    Args:
        ckpt_dir: Directory to search for epoch_NNNN.pt files.
        model: Model to load weights into.
        optimizer: Optimiser to restore state for.
        scaler: GradScaler to restore state for.
        scheduler: LR scheduler to restore state for.
        device: Target device for tensors.

    Returns:
        (start_epoch, best_val_loss): The epoch to resume FROM (i.e. the next
        epoch to run) and the best val_loss seen so far.
        Returns (1, inf) if no checkpoint is found.
    """
    candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not candidates:
        logger.info("No existing checkpoint found — starting from epoch 1.")
        return 1, float("inf")

    latest = candidates[-1]
    logger.info(f"Resuming from checkpoint: {latest}")

    ckpt = torch.load(latest, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    resumed_epoch = ckpt["epoch"]
    best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))

    logger.info(
        f"Resumed from epoch {resumed_epoch}  "
        f"(best val_loss so far: {best_val_loss:.6f})"
    )
    return resumed_epoch + 1, best_val_loss


# ─── Main local training function ─────────────────────────────────────────────

def local_train(cfg: LocalTrainConfig) -> list[dict[str, float | int]]:
    """Run the full local training loop with all 6 GB memory-fit settings.

    Step order:
        1.  gpu_check()
        2.  Seed RNG
        3.  Build SeaIceDataPipeline → load, clean, normalise, format
        4.  Apply spatial crop (if configured)
        5.  create_dataloaders (batch_size=1, pin_memory=True)
        6.  Build SeaIceUNet + SeaIceLoss
        7.  Build AdamW + CosineAnnealingLR + GradScaler
        8.  Optionally resume from checkpoint
        9.  Dry-run memory gate
        10. Epoch loop: train → val → scheduler → checkpoint → print stats + ETA

    Args:
        cfg: LocalTrainConfig instance.

    Returns:
        List of per-epoch history dicts (epoch, train_loss, val_loss, val_iiee, lr).
        Also writes <checkpoint_dir>/training_history.json.
    """
    # 1. GPU check
    device = gpu_check()
    print(f"\n  Config: {cfg.describe()}\n")

    # 2. Seed
    _seed_everything(cfg.seed)

    # 3. Data pipeline
    logger.info("Building data pipeline...")
    pipeline = SeaIceDataPipeline(
        data_path=cfg.data_path,
        mask_path=cfg.mask_path,
        variable_name=cfg.variable_name,
    )
    pipeline.load().clean().normalize(max_value=100.0)
    land_mask_np = pipeline.land_mask  # (H, W) numpy

    logger.info(
        f"Formatting data for U-Net  "
        f"(window_size={cfg.window_size}, horizon={cfg.horizon})..."
    )
    X, Y = pipeline.format_for_model(
        "unet", window_size=cfg.window_size, horizon=cfg.horizon
    )

    # 4. Spatial crop (optional memory lever)
    X, Y, land_mask_np = apply_spatial_crop(X, Y, land_mask_np, cfg)

    land_mask_tensor = torch.from_numpy(land_mask_np)

    # 5. DataLoaders
    logger.info(f"Creating DataLoaders  (batch_size={cfg.batch_size})...")
    train_loader, val_loader = create_dataloaders(
        X,
        Y,
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        shuffle_train=True,
        num_workers=0,          # 0 workers avoids pickling issues on Windows
        pin_memory=True,        # pinned memory accelerates CPU→GPU transfers
    )

    # 6. Model + criterion
    logger.info(
        f"Instantiating SeaIceUNet  "
        f"(base_filters={cfg.base_filters}, depth={cfg.depth})..."
    )
    model = SeaIceUNet(
        in_channels=cfg.window_size,
        out_channels=1,
        base_filters=cfg.base_filters,
        depth=cfg.depth,
        land_mask=land_mask_tensor,
    ).to(device)

    criterion = SeaIceLoss(
        land_mask=land_mask_tensor.to(device),
        base_loss="mse",
        edge_weight=3.0,
        edge_band=0.05,
        edge_threshold=0.15,
    )

    land_mask_device = land_mask_tensor.to(device)

    # 7. Optimiser + scheduler + scaler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs
    )
    scaler = GradScaler(enabled=cfg.use_amp)

    # 8. Resume
    ckpt_dir = cfg.ckpt_dir_path()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_loss = float("inf")

    if cfg.resume:
        start_epoch, best_val_loss = _try_resume(
            ckpt_dir, model, optimizer, scaler, scheduler, device
        )

    # 9. Dry-run memory gate
    peak_mb = _dry_run(
        model, train_loader, criterion, optimizer, scaler, device, cfg
    )

    if cfg.dry_run_only:
        logger.info("--dry_run_only set: stopping after dry run.")
        return []

    # 10. Epoch loop
    history: list[dict[str, float | int]] = []
    patience_counter = 0
    epoch_times: list[float] = []

    logger.info(
        f"Starting training from epoch {start_epoch} / {cfg.num_epochs}  "
        f"(best val_loss so far: {best_val_loss:.6f})"
    )

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t_epoch_start = time.monotonic()

        # ── Train ──
        train_loss = _train_epoch_local(
            model, train_loader, criterion, optimizer, scaler, device, cfg
        )

        # ── Validate ──
        val_loss, val_iiee = _val_epoch_local(
            model, val_loader, criterion, land_mask_device, device, cfg
        )

        # ── LR step ──
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Timing + ETA ──
        epoch_elapsed = time.monotonic() - t_epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = sum(epoch_times[-10:]) / len(epoch_times[-10:])  # rolling 10
        remaining_epochs = cfg.num_epochs - epoch
        eta_s = avg_epoch_time * remaining_epochs
        eta_str = _format_duration(eta_s)

        # ── History ──
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "val_loss": round(val_loss, 8),
            "val_iiee": round(val_iiee, 8),
            "lr": round(current_lr, 10),
            "epoch_time_s": round(epoch_elapsed, 2),
        }
        history.append(record)

        # ── Print progress ──
        vram_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(
            f"Epoch {epoch:>4d}/{cfg.num_epochs} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"iiee={val_iiee:.4f} | "
            f"lr={current_lr:.3e} | "
            f"time={epoch_elapsed:.0f}s | "
            f"ETA={eta_str} | "
            f"VRAM={vram_mb:.0f}MB"
        )

        # ── Checkpoint ──
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            logger.info(
                f"  ✓ New best val_loss={val_loss:.6f} at epoch {epoch}"
            )
        else:
            patience_counter += 1
            logger.debug(
                f"  No improvement. Patience: {patience_counter}/{cfg.patience}"
            )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if cfg.use_amp else None,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iiee": val_iiee,
            "best_val_loss": best_val_loss,
            "cfg": cfg,
        }
        _save_local_checkpoint(state, ckpt_dir, epoch, is_best)

        # ── Early stopping ──
        if patience_counter >= cfg.patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {cfg.patience} consecutive epochs)."
            )
            break

    # ── Post-training: reload best weights ──
    best_ckpt = ckpt_dir / "best_model.pt"
    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"])
        logger.info(
            f"Best model reloaded from epoch {best_state['epoch']} "
            f"(val_loss={best_state['val_loss']:.6f}, "
            f"val_iiee={best_state['val_iiee']:.4f})."
        )

    # ── Write JSON history ──
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history written to {history_path}")

    return history


# ─── Duration formatter ───────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 0:
        return "0s"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser, mirroring LocalTrainConfig fields."""
    p = argparse.ArgumentParser(
        prog="local_train.py",
        description=(
            "Local RTX 4050 training for Sea-Ice U-Net  "
            "(6 GB VRAM memory-safe mode)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    data = p.add_argument_group("Data")
    data.add_argument("--data_path", default="data/sea_ice/*.nc",
                      help="Glob / path to sea-ice NetCDF files.")
    data.add_argument("--mask_path", default="data/land_mask.nc",
                      help="Path to land mask NetCDF or .npy file.")
    data.add_argument("--variable_name", default="siconc",
                      help="Sea-ice variable name inside the NetCDF files.")

    # Sequence / spatial
    seq = p.add_argument_group("Sequence / Spatial")
    seq.add_argument("--window_size", type=int, default=7,
                     help="N input days (reduce as last-resort memory lever).")
    seq.add_argument("--horizon", type=int, default=1,
                     help="K target days ahead.")
    seq.add_argument("--crop_h", type=int, default=None,
                     help="Centre-crop rows (spatial memory lever). Must pair with --crop_w.")
    seq.add_argument("--crop_w", type=int, default=None,
                     help="Centre-crop cols (spatial memory lever). Must pair with --crop_h.")

    # Model
    arch = p.add_argument_group("Model Architecture")
    arch.add_argument("--base_filters", type=int, default=32,
                      help="Filters in first encoder level (16 halves feature-map memory).")
    arch.add_argument("--depth", type=int, default=4,
                      help="U-Net depth (3 = shallower, less memory).")

    # Training
    tr = p.add_argument_group("Training")
    tr.add_argument("--batch_size", type=int, default=1,
                    help="Batch size. Keep at 1 for 6 GB GPU.")
    tr.add_argument("--num_epochs", type=int, default=50,
                    help="Maximum training epochs.")
    tr.add_argument("--lr", type=float, default=1e-3,
                    help="AdamW initial learning rate.")
    tr.add_argument("--weight_decay", type=float, default=1e-4,
                    help="AdamW weight decay.")
    tr.add_argument("--patience", type=int, default=10,
                    help="Early-stopping patience in epochs.")
    tr.add_argument("--grad_clip_norm", type=float, default=1.0,
                    help="Gradient clipping max L2 norm.")
    tr.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    tr.add_argument("--val_split", type=float, default=0.2,
                    help="Fraction of samples for chronological validation.")

    # Memory flags
    mem = p.add_argument_group("Memory-fit Flags")
    mem.add_argument("--no_amp", action="store_true",
                     help="Disable Automatic Mixed Precision (not recommended).")
    mem.add_argument("--no_grad_checkpoint", action="store_true",
                     help="Disable gradient checkpointing (not recommended on 6 GB).")

    # Dry run
    dr = p.add_argument_group("Dry-run / Memory Gate")
    dr.add_argument("--dry_run_only", action="store_true",
                    help="Run only the dry-run memory check then exit.")
    dr.add_argument("--dry_run_warn_mb", type=float, default=5_500.0,
                    help="Warn if dry-run peak exceeds this many MB.")
    dr.add_argument("--no_confirm", action="store_true",
                    help="Exit (instead of prompting) if dry-run warning triggers.")

    # Checkpointing
    ckpt = p.add_argument_group("Checkpointing")
    ckpt.add_argument("--checkpoint_dir", default="checkpoints_local/",
                      help="Directory for per-epoch checkpoints and history JSON.")
    ckpt.add_argument("--no_resume", action="store_true",
                      help="Ignore existing checkpoints and start from epoch 1.")

    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = LocalTrainConfig(
        # Data
        data_path=args.data_path,
        mask_path=args.mask_path,
        variable_name=args.variable_name,
        # Sequence / spatial
        window_size=args.window_size,
        horizon=args.horizon,
        crop_h=args.crop_h,
        crop_w=args.crop_w,
        # Model
        base_filters=args.base_filters,
        depth=args.depth,
        # Training
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        val_split=args.val_split,
        # Memory flags (inverted store_true flags)
        use_amp=not args.no_amp,
        use_grad_checkpoint=not args.no_grad_checkpoint,
        # Dry-run
        dry_run_only=args.dry_run_only,
        dry_run_warn_mb=args.dry_run_warn_mb,
        no_confirm=args.no_confirm,
        # Checkpointing
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
    )

    history = local_train(cfg)

    if history:
        best = min(history, key=lambda r: r["val_loss"])
        print(
            f"\n{'='*65}"
            f"\nTraining complete."
            f"\n  Best epoch    : {best['epoch']}"
            f"\n  Best val_loss : {best['val_loss']:.6f}"
            f"\n  Best val_iiee : {best['val_iiee']:.4f}"
            f"\n  Checkpoints   : {cfg.checkpoint_dir}"
            f"\n{'='*65}"
        )
