"""Colab Notebook Entry Point — ConvLSTM Sea-Ice Concentration Forecasting.

This single file is structured as sequential "cells" that run top-to-bottom
in a Colab notebook connected to this repo.

Usage in Colab:
    1. Mount Google Drive
    2. cd to sea-ice-pipeline/tests/
    3. Run this file: %run notebook.py

Or copy each section between #%% markers into separate Colab cells.
"""

# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 1: Configuration
# All tuneable parameters — edit this cell and re-run to experiment.
# ═══════════════════════════════════════════════════════════════════════════════

from config import Config

cfg = Config(
    # ── Paths (adjust for your Colab mount) ──
    data_root="../data/South",                     # relative to tests/
    checkpoint_dir="checkpoints",                  # or "/content/drive/MyDrive/sea_ice_ckpts/"

    # ── Sliding window ──
    N=10,       # past days as input
    K=3,        # future days to forecast

    # ── Data split ──
    train_frac=0.8,
    val_frac=0.1,

    # ── ConvLSTM architecture ──
    num_layers=2,
    hidden_channels=64,
    kernel_size=3,

    # ── Training ──
    epochs=5,
    batch_size=4,
    lr=1e-3,
    grad_clip_max_norm=1.0,
    seed=42,

    # ── Performance ──
    num_workers=0,      # 0 works in Colab; increase on multi-CPU machines
    pin_memory=False,   # set True when using GPU
    downsample_factor=1,  # set to 2 to halve spatial dims (faster experiments)
)

print(cfg.summary())

# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 2: GPU Verification
# Must show a GPU — if not, stop and fix runtime settings.
# ═══════════════════════════════════════════════════════════════════════════════

from train import verify_gpu

device = verify_gpu()

# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 3: Data Pipeline
# Load NetCDF → mask → normalize → sliding-window Dataset → DataLoaders
# ═══════════════════════════════════════════════════════════════════════════════

from data import prepare_data

train_loader, val_loader, test_loader, data_cube = prepare_data(cfg)

# Quick visual check: print shapes from first batch
bx, by = next(iter(train_loader))
print(f"\nFirst train batch:")
print(f"  Input  X: {bx.shape}  (B, N, C, H, W)")
print(f"  Target Y: {by.shape}  (B, K, C, H, W)")

# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 4: Model Instantiation
# ═══════════════════════════════════════════════════════════════════════════════

from model import build_model

model = build_model(cfg)
print(f"\nModel parameter count: {sum(p.numel() for p in model.parameters()):,}")

# Quick forward-pass sanity check on CPU/GPU
import torch
model_test = model.to(device)
with torch.no_grad():
    test_out = model_test(bx.to(device))
print(f"Forward pass test — output shape: {test_out.shape}")
print(f"Output range: [{test_out.min().item():.4f}, {test_out.max().item():.4f}]")
del test_out  # free GPU memory

# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 5: Training
# Runs the full training loop with per-epoch checkpointing and resume.
# ═══════════════════════════════════════════════════════════════════════════════

from train import train_model

history = train_model(model, train_loader, val_loader, cfg, device=device)

# %% ═══════════════════════════════════════════════════════════════════════════
# CELL 6: Results Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Training History")
print("=" * 60)
print(f"{'Epoch':>6} │ {'Train Loss':>12} │ {'Val Loss':>12} │ {'Time (s)':>10}")
print("─" * 50)
for r in history:
    print(f"{r['epoch']:>6} │ {r['train_loss']:>12.6f} │ {r['val_loss']:>12.6f} │ {r['elapsed_s']:>10.1f}")

if history:
    best = min(history, key=lambda r: r["val_loss"])
    print(f"\n★ Best epoch: {best['epoch']} with val_loss={best['val_loss']:.6f}")
