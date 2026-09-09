"""Centralized configuration for the ConvLSTM sea-ice forecasting pipeline.

All hyperparameters, paths, and tuneable constants live here so that
Colab users can tweak a single cell instead of hunting through multiple files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Pipeline-wide configuration dataclass.

    Attributes:
        data_root: Root directory containing Antarctic_NetCDF_{year}/ subdirs.
        checkpoint_dir: Where to save per-epoch checkpoints (Google Drive path in Colab).
        hemisphere: 'south' for Antarctic, 'north' for Arctic.
        concentration_var: NetCDF variable name for sea-ice concentration.
        N: Number of past daily frames used as encoder input.
        K: Number of future daily frames the decoder must predict.
        train_frac: Fraction of total sliding-window samples for training.
        val_frac: Fraction for validation. Test = 1 - train_frac - val_frac.
        num_layers: Number of stacked ConvLSTM layers in encoder and decoder.
        hidden_channels: Hidden channel count per ConvLSTM layer.
        kernel_size: Spatial convolution kernel size in ConvLSTM cells.
        epochs: Maximum training epochs.
        batch_size: Samples per mini-batch.
        lr: Adam optimizer learning rate.
        grad_clip_max_norm: Max L2 norm for gradient clipping.
        seed: Random seed for reproducibility.
        num_workers: DataLoader worker processes (0 = main process).
        pin_memory: Pin CUDA memory in DataLoader (set True when GPU available).
        downsample_factor: Spatial downsample factor (1 = full resolution).
    """

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_root: str = str(Path(__file__).parent / "data" / "South")
    checkpoint_dir: str = "checkpoints"
    hemisphere: str = "south"
    concentration_var: str = "cdr_seaice_conc"

    # ── Sliding Window ─────────────────────────────────────────────────────────
    N: int = 10   # input sequence length  (past days)
    K: int = 3    # target sequence length  (forecast days)

    # ── Data Split (chronological, by sliding-window index) ────────────────────
    train_frac: float = 0.8
    val_frac: float = 0.1
    # test_frac is implicitly 1 - train_frac - val_frac

    # ── ConvLSTM Model ─────────────────────────────────────────────────────────
    num_layers: int = 2
    hidden_channels: int = 64
    kernel_size: int = 3

    # ── Training ───────────────────────────────────────────────────────────────
    epochs: int = 5
    batch_size: int = 4
    lr: float = 1e-3
    grad_clip_max_norm: float = 1.0
    seed: int = 42

    # ── DataLoader ─────────────────────────────────────────────────────────────
    num_workers: int = 0
    pin_memory: bool = False

    # ── Optional spatial downsample (1 = full resolution) ──────────────────────
    downsample_factor: int = 1

    # ── Derived ────────────────────────────────────────────────────────────────
    @property
    def test_frac(self) -> float:
        return round(1.0 - self.train_frac - self.val_frac, 4)

    def __post_init__(self) -> None:
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError(
                f"train_frac ({self.train_frac}) + val_frac ({self.val_frac}) must be < 1.0"
            )
        if self.N < 1 or self.K < 1:
            raise ValueError(f"N and K must be >= 1, got N={self.N}, K={self.K}")
        if self.downsample_factor < 1:
            raise ValueError(f"downsample_factor must be >= 1, got {self.downsample_factor}")

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            "=" * 60,
            "ConvLSTM Sea-Ice Pipeline Configuration",
            "=" * 60,
            f"  Data root:         {self.data_root}",
            f"  Checkpoint dir:    {self.checkpoint_dir}",
            f"  Hemisphere:        {self.hemisphere}",
            f"  Concentration var: {self.concentration_var}",
            f"  Input window (N):  {self.N} days",
            f"  Forecast horizon (K): {self.K} days",
            f"  Split:             train={self.train_frac:.0%} / val={self.val_frac:.0%} / test={self.test_frac:.0%}",
            f"  ConvLSTM layers:   {self.num_layers}",
            f"  Hidden channels:   {self.hidden_channels}",
            f"  Kernel size:       {self.kernel_size}",
            f"  Epochs:            {self.epochs}",
            f"  Batch size:        {self.batch_size}",
            f"  Learning rate:     {self.lr}",
            f"  Grad clip norm:    {self.grad_clip_max_norm}",
            f"  Downsample factor: {self.downsample_factor}x",
            f"  Seed:              {self.seed}",
            "=" * 60,
        ]
        return "\n".join(lines)
