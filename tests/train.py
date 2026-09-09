"""ConvLSTM Training Loop with per-epoch checkpointing and resume support.

Features:
    - GPU verification (prints device info, runs nvidia-smi).
    - Adam optimizer + MSE loss.
    - Gradient clipping (clip_grad_norm_) for stable BPTT through ConvLSTM.
    - Per-epoch checkpoint saving with epoch number in filename.
    - Resume from latest checkpoint if one exists.
    - Train and validation loss printed every epoch.
"""

from __future__ import annotations

import glob
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config


logger = logging.getLogger("ConvLSTMTrainer")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s — %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms (slight perf cost, full reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  GPU Verification
# ═══════════════════════════════════════════════════════════════════════════════

def verify_gpu() -> torch.device:
    """Check CUDA availability and print GPU info.

    Returns:
        device: ``torch.device('cuda')`` if GPU available, else ``torch.device('cpu')``.

    Prints a clear warning if no GPU is detected.
    """
    print("\n" + "=" * 60)
    print("GPU Verification")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"  torch.cuda.is_available(): {cuda_available}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"  torch.cuda.get_device_name(0): {device_name}")
        mem_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"  GPU memory: {mem_total:.1f} GB")
        device = torch.device("cuda")
    else:
        print("  ⚠ WARNING: No GPU detected! Training will run on CPU (very slow).")
        print("  If running in Colab, go to Runtime → Change runtime type → GPU.")
        device = torch.device("cpu")

    # Try nvidia-smi (works in Colab, may fail locally)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"\n  nvidia-smi output:\n{result.stdout}")
        else:
            print("  nvidia-smi: not available or errored")
    except Exception:
        print("  nvidia-smi: not found on this system")

    print("=" * 60 + "\n")
    return device


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Checkpoint I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(
    state: dict[str, Any],
    checkpoint_dir: str,
    epoch: int,
) -> str:
    """Save a checkpoint with the epoch number in the filename.

    File: ``{checkpoint_dir}/checkpoint_epoch_{epoch:04d}.pt``

    Uses atomic write (save to .tmp, then rename) to prevent corruption
    if the process is interrupted mid-write.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    filename = f"checkpoint_epoch_{epoch:04d}.pt"
    filepath = ckpt_dir / filename
    tmp_path = filepath.with_suffix(".tmp")

    torch.save(state, tmp_path)
    tmp_path.replace(filepath)

    return str(filepath)


def _find_latest_checkpoint(checkpoint_dir: str) -> tuple[str | None, int]:
    """Scan checkpoint_dir for the highest-epoch checkpoint file.

    Returns:
        path: Path to latest checkpoint, or None if none found.
        epoch: Epoch number of latest checkpoint, or 0 if none found.
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    files = sorted(glob.glob(pattern))

    if not files:
        return None, 0

    # Extract epoch numbers and find the max
    best_epoch = 0
    best_path = None
    for f in files:
        match = re.search(r"checkpoint_epoch_(\d+)\.pt$", f)
        if match:
            ep = int(match.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_path = f

    return best_path, best_epoch


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load model and optimizer state from a checkpoint file.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into.
        device: Target device.

    Returns:
        epoch: The epoch number that was saved (training resumes from epoch+1).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt["epoch"]
    logger.info(
        f"Resumed from checkpoint: {path} (epoch {epoch}, "
        f"train_loss={ckpt.get('train_loss', '?'):.6f}, "
        f"val_loss={ckpt.get('val_loss', '?'):.6f})"
    )
    return epoch


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device | None = None,
) -> list[dict[str, float | int]]:
    """Train the ConvLSTM encoder-decoder with per-epoch checkpointing and resume.

    Args:
        model: ConvLSTMEncoderDecoder instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Pipeline configuration (epochs, lr, batch_size, checkpoint_dir, etc.).
        device: Compute device. If None, auto-detects GPU/CPU.

    Returns:
        history: List of per-epoch dicts with keys
            ``{'epoch', 'train_loss', 'val_loss', 'elapsed_s'}``.
    """
    # ── Reproducibility ──
    seed_everything(cfg.seed)

    # ── Device ──
    if device is None:
        device = verify_gpu()
    model = model.to(device)

    # ── Optimizer & Loss ──
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    # ── Resume from checkpoint if available ──
    start_epoch = 1
    ckpt_path, ckpt_epoch = _find_latest_checkpoint(cfg.checkpoint_dir)
    if ckpt_path is not None:
        start_epoch = _load_checkpoint(ckpt_path, model, optimizer, device) + 1
        logger.info(f"Continuing training from epoch {start_epoch}")
    else:
        logger.info("No existing checkpoint found — starting from scratch.")

    # ── Training state ──
    history: list[dict[str, float | int]] = []

    print("\n" + "=" * 70)
    print(f"Training ConvLSTM: epochs {start_epoch}–{cfg.epochs}, "
          f"lr={cfg.lr}, batch={cfg.batch_size}, grad_clip={cfg.grad_clip_max_norm}")
    print("=" * 70)

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.monotonic()

        # ── Train phase ───────────────────────────────────────────────────────
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)  # (B, N, C, H, W)
            batch_y = batch_y.to(device, non_blocking=True)  # (B, K, C, H, W)

            optimizer.zero_grad()
            pred = model(batch_x)  # (B, K, C, H, W)
            loss = criterion(pred, batch_y)
            loss.backward()

            # Gradient clipping — critical for ConvLSTM stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip_max_norm
            )

            optimizer.step()
            train_loss_accum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_accum / max(train_batches, 1)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss_accum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_accum / max(val_batches, 1)

        elapsed = time.monotonic() - t0

        # ── Log ───────────────────────────────────────────────────────────────
        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 8),
            "val_loss": round(avg_val_loss, 8),
            "elapsed_s": round(elapsed, 2),
        }
        history.append(record)

        print(
            f"Epoch {epoch:>3d}/{cfg.epochs} │ "
            f"train_loss={avg_train_loss:.6f} │ "
            f"val_loss={avg_val_loss:.6f} │ "
            f"time={elapsed:.1f}s"
        )

        # ── Checkpoint (every epoch) ──────────────────────────────────────────
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        saved_path = _save_checkpoint(ckpt_state, cfg.checkpoint_dir, epoch)
        logger.info(f"  Checkpoint saved: {saved_path}")

    print("=" * 70)
    print("Training complete!")
    if history:
        best = min(history, key=lambda r: r["val_loss"])
        print(f"Best epoch: {best['epoch']} (val_loss={best['val_loss']:.6f})")
    print("=" * 70 + "\n")

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from model import build_model
    from data import prepare_data

    cfg = Config(epochs=2, batch_size=2)
    print(cfg.summary())

    device = verify_gpu()
    train_loader, val_loader, _, _ = prepare_data(cfg)
    model = build_model(cfg)

    history = train_model(model, train_loader, val_loader, cfg, device=device)
    print(f"\nHistory: {history}")
