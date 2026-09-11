"""Local Training Configuration for RTX 4050 Laptop GPU (6 GB VRAM).

Defines `LocalTrainConfig`, a frozen-friendly dataclass that centralises every
memory-fit knob required for running the sea-ice U-Net locally.  Importing this
file has zero side-effects on the Lightning AI / cloud training path.

Usage example
-------------
>>> from local_config import LocalTrainConfig
>>> cfg = LocalTrainConfig(crop_h=256, crop_w=256)   # enable spatial crop lever
>>> cfg = LocalTrainConfig(window_size=5, horizon=1)  # shorten sequence lever
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LocalTrainConfig:
    """All knobs for local RTX 4050 training in one place.

    Memory-fit settings (all ON by default — ConvLSTM/U-Net at 6 GB needs all of them):
        - batch_size = 1
        - use_amp = True          (torch.cuda.amp autocast + GradScaler)
        - use_grad_checkpoint = True  (torch.utils.checkpoint on bottleneck + decoder)

    Extra memory levers (disabled by default, enable if still OOM after the dry-run):
        - crop_h / crop_w: centre-crop the spatial grid to this size before training.
        - window_size / horizon: reduce N input days or K target days.

    Attributes
    ----------
    # ── Data paths ──────────────────────────────────────────────────────────────
    data_path:
        Glob pattern, directory, single file path, or list of paths to NetCDF
        sea-ice concentration files (forwarded to SeaIceDataPipeline).
    mask_path:
        Path to the static land mask NetCDF or .npy file.
    variable_name:
        Name of the sea-ice concentration variable inside the NetCDF files.

    # ── Sequence / spatial shape ─────────────────────────────────────────────────
    window_size:
        Number of consecutive past days used as model input (N input days).
        Reduce as a last-resort memory lever (minimum: 3).
    horizon:
        Forecast horizon in days (K target days).  Increase for multi-day targets.
    crop_h, crop_w:
        Optional centre-crop dimensions (rows, columns).  Set both to enable the
        spatial crop lever — e.g. crop_h=256, crop_w=256 for a 256×256 sub-region.
        None (default) keeps the full native grid.

    # ── Model architecture ───────────────────────────────────────────────────────
    base_filters:
        Number of filters in the first U-Net encoder level.  32 gives ~2.3 M params
        at depth=4.  Reduce to 16 to roughly halve feature-map memory.
    depth:
        Number of U-Net down/up-sampling stages.  4 is the standard; 3 saves memory.

    # ── Optimiser / training ─────────────────────────────────────────────────────
    batch_size:
        REQUIRED to be 1 for 6 GB local mode.  Increasing this will OOM.
    num_epochs:
        Maximum number of training epochs before stopping.
    lr:
        AdamW initial learning rate.
    weight_decay:
        AdamW weight-decay coefficient.
    patience:
        Early-stopping patience in epochs (stop after this many consecutive
        non-improving validation epochs).
    grad_clip_norm:
        Maximum gradient L2 norm for clip_grad_norm_.
    seed:
        Global random seed for full reproducibility.
    val_split:
        Fraction of samples reserved for chronological validation.

    # ── Memory-fit flags ─────────────────────────────────────────────────────────
    use_amp:
        Enable Automatic Mixed Precision (fp16 forward + fp32 master weights).
        Halves activation memory and exploits RTX 4050 Tensor Cores.
    use_grad_checkpoint:
        Enable gradient checkpointing on the U-Net bottleneck and all decoder
        up-blocks.  Trades re-computation for substantially lower peak activation
        memory during backward.

    # ── Dry-run memory gate ──────────────────────────────────────────────────────
    dry_run_only:
        If True, execute exactly one forward+backward pass, print peak memory,
        then exit without starting real training.  Useful for memory budgeting.
    dry_run_warn_mb:
        If peak dry-run memory exceeds this threshold (MB), print a warning and
        ask for confirmation before proceeding.  Default: 5_500 MB (leaves
        ~500 MB headroom on a 6 GB card).
    no_confirm:
        If True, skip the interactive confirmation prompt and proceed (or exit)
        automatically.  Useful for scripted / CI runs.

    # ── Checkpointing ────────────────────────────────────────────────────────────
    checkpoint_dir:
        Directory where per-epoch checkpoints, best_model.pt, last_model.pt, and
        training_history.json are saved.
    resume:
        If True, automatically detect and resume from the highest-numbered
        epoch_NNNN.pt checkpoint in checkpoint_dir.
    """

    # ── Data paths ──────────────────────────────────────────────────────────────
    data_path: str = "data/sea_ice/*.nc"
    mask_path: str = "data/land_mask.nc"
    variable_name: str = "siconc"

    # ── Sequence / spatial shape ─────────────────────────────────────────────────
    window_size: int = 7
    horizon: int = 1
    crop_h: int | None = None
    crop_w: int | None = None

    # ── Model architecture ───────────────────────────────────────────────────────
    base_filters: int = 32
    depth: int = 4

    # ── Optimiser / training ─────────────────────────────────────────────────────
    batch_size: int = 1
    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    grad_clip_norm: float = 1.0
    seed: int = 42
    val_split: float = 0.2

    # ── Memory-fit flags (all ON by default) ─────────────────────────────────────
    use_amp: bool = True
    use_grad_checkpoint: bool = True

    # ── Dry-run memory gate ──────────────────────────────────────────────────────
    dry_run_only: bool = False
    dry_run_warn_mb: float = 5_500.0
    no_confirm: bool = False

    # ── Checkpointing ────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints_local/"
    resume: bool = True

    # ── Derived helpers (not CLI args) ───────────────────────────────────────────

    def ckpt_dir_path(self) -> Path:
        """Return checkpoint_dir as a resolved Path object."""
        return Path(self.checkpoint_dir)

    def describe(self) -> str:
        """Return a compact human-readable summary of active settings."""
        crop_str = (
            f"{self.crop_h}×{self.crop_w}"
            if (self.crop_h and self.crop_w)
            else "full grid"
        )
        return (
            f"LocalTrainConfig("
            f"batch={self.batch_size}, "
            f"window={self.window_size}, "
            f"horizon={self.horizon}, "
            f"grid_crop={crop_str}, "
            f"base_filters={self.base_filters}, "
            f"depth={self.depth}, "
            f"amp={self.use_amp}, "
            f"grad_ckpt={self.use_grad_checkpoint}, "
            f"epochs={self.num_epochs}, "
            f"lr={self.lr:.2e}"
            f")"
        )
