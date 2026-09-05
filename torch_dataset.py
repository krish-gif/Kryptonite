"""PyTorch Dataset and Chronological DataLoader for Sea-Ice Forecasting.

Wraps preprocessed NumPy tensors from `SeaIceDataPipeline` into PyTorch `Dataset`
and `DataLoader` instances with strict chronological train/validation splitting
to prevent temporal data leakage across overlapping sliding windows.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


logger = logging.getLogger("SeaIceTorchDataset")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class SeaIceDataset(Dataset):
    """PyTorch Dataset wrapper for preprocessed sea-ice concentration arrays.

    Stores references to X and Y arrays without unnecessary copies.
    __getitem__ returns individual sample tensors of dtype torch.float32
    without a leading batch dimension (the DataLoader handles batch collation).

    Compatible with both:
    - ConvLSTM inputs: X shaped (N, Time_Steps, Channels, H, W)
    - U-Net inputs:    X shaped (N, Channels, H, W)
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Initialize SeaIceDataset.

        Args:
            X: Input sequences array of shape (Num_Samples, ...).
            Y: Target maps array of shape (Num_Samples, ...).

        Raises:
            TypeError: If X or Y are not NumPy ndarrays.
            ValueError: If lengths do not match or arrays are empty.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy.ndarray, got {type(X)}")
        if not isinstance(Y, np.ndarray):
            raise TypeError(f"Y must be a numpy.ndarray, got {type(Y)}")

        if len(X) != len(Y):
            raise ValueError(
                f"Sample count mismatch: X has {len(X)} samples, but Y has {len(Y)} samples."
            )

        if len(X) == 0:
            raise ValueError("SeaIceDataset cannot be initialized with 0 samples.")

        # Store arrays as references (zero-copy)
        # Ensure memory is C-contiguous so torch.from_numpy() operates zero-copy
        self.X = np.ascontiguousarray(X)
        self.Y = np.ascontiguousarray(Y)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a single training sample as float32 PyTorch tensors.

        Args:
            idx: Sample index (0 <= idx < len(self)).

        Returns:
            x: Input tensor for a single sample (no leading batch dimension).
            y: Target tensor for a single sample (no leading batch dimension).
        """
        x_np = self.X[idx]
        y_np = self.Y[idx]

        # Use torch.from_numpy() for zero-copy memory wrapping
        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        # Enforce float32 dtype
        if x_tensor.dtype != torch.float32:
            x_tensor = x_tensor.to(dtype=torch.float32)
        if y_tensor.dtype != torch.float32:
            y_tensor = y_tensor.to(dtype=torch.float32)

        return x_tensor, y_tensor


def chronological_split(
    X: np.ndarray,
    Y: np.ndarray,
    val_split: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split sliding-window arrays into train and validation sets.

    IMPORTANT: Never use random splitting on sliding-window data!
    Because sliding windows overlap (e.g. sample 0 uses days 0–6, sample 1 uses days 1–7),
    random shuffling leaks identical/near-duplicate maps between train and val.
    Chronological splitting places the first (1 - val_split) continuous period
    into training and the remaining tail into validation.

    Args:
        X: Sequence input array of shape (N, ...).
        Y: Target array of shape (N, ...).
        val_split: Fraction of samples reserved for validation (0.0 < val_split < 1.0).

    Returns:
        X_train: Training input array.
        Y_train: Training target array.
        X_val: Validation input array.
        Y_val: Validation target array.

    Raises:
        ValueError: If val_split is not in (0.0, 1.0) or results in an empty split.
    """
    if not (0.0 < val_split < 1.0):
        raise ValueError(
            f"val_split must be strictly between 0.0 and 1.0 (exclusive), got {val_split}"
        )

    n_samples = len(X)
    val_size = int(round(n_samples * val_split))

    if val_size < 1:
        raise ValueError(
            f"Validation split of {val_split:.2f} on {n_samples} total samples produces "
            f"0 validation samples. Increase val_split or provide more timesteps."
        )

    train_size = n_samples - val_size
    if train_size < 1:
        raise ValueError(
            f"Validation split of {val_split:.2f} on {n_samples} total samples produces "
            f"0 training samples. Reduce val_split."
        )

    # Strictly chronological slice: head -> train, tail -> val
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_val = X[train_size:]
    Y_val = Y[train_size:]

    logger.info(
        f"Chronological split completed: Total={n_samples} -> "
        f"Train={len(X_train)} ({len(X_train)/n_samples*100:.1f}%), "
        f"Val={len(X_val)} ({len(X_val)/n_samples*100:.1f}%)"
    )

    return X_train, Y_train, X_val, Y_val


def create_dataloaders(
    X: np.ndarray,
    Y: np.ndarray,
    val_split: float = 0.2,
    batch_size: int = 8,
    shuffle_train: bool = True,
    num_workers: int = 0,
    drop_last_train: bool = False,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Factory function to split data chronologically and build PyTorch DataLoaders.

    Args:
        X: Preprocessed input array (from SeaIceDataPipeline.format_for_model()).
        Y: Preprocessed target array (from SeaIceDataPipeline.format_for_model()).
        val_split: Fraction of samples for validation (default: 0.2).
        batch_size: Number of samples per batch (default: 8).
        shuffle_train: Whether to shuffle batch order *within* the training set (default: True).
                       Shuffling within the pre-isolated training period does not cause leakage.
        num_workers: Subprocesses for data loading (default: 0 for synchronous).
        drop_last_train: Whether to drop the last incomplete training batch (default: False).
        pin_memory: If True, copies Tensors to pinned CUDA memory before returning (default: False).

    Returns:
        train_loader: PyTorch DataLoader for training.
        val_loader: PyTorch DataLoader for validation (shuffle is always False).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    # 1. Chronological train/val split
    X_train, Y_train, X_val, Y_val = chronological_split(X, Y, val_split=val_split)

    # 2. Build Dataset instances
    train_dataset = SeaIceDataset(X_train, Y_train)
    val_dataset = SeaIceDataset(X_val, Y_val)

    # 3. Build DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=drop_last_train,
        pin_memory=pin_memory,
    )

    # Validation loader is NEVER shuffled to maintain deterministic, chronological evaluation
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    from pathlib import Path
    from data_pipeline import SeaIceDataPipeline

    base_dir = Path(__file__).parent / "data"
    sea_ice_pattern = str(base_dir / "sea_ice" / "*.nc")
    mask_file = str(base_dir / "land_mask.nc")

    print("=" * 75)
    print("Testing PyTorch Dataset & DataLoader Module")
    print("=" * 75)

    # 1. Run data pipeline to get formatted arrays
    pipeline = SeaIceDataPipeline(
        data_path=sea_ice_pattern,
        mask_path=mask_file,
    )
    pipeline.load().clean().normalize(max_value=100.0)

    # Test ConvLSTM format
    print("\n--- 1. ConvLSTM Pipeline Integration ---")
    X_conv, Y_conv = pipeline.format_for_model("convlstm", window_size=7, horizon=1)
    train_loader_conv, val_loader_conv = create_dataloaders(
        X_conv, Y_conv, val_split=0.2, batch_size=8, shuffle_train=True
    )

    print(f"Total samples:       {len(X_conv)}")
    print(f"Train dataset len:   {len(train_loader_conv.dataset)}")
    print(f"Val dataset len:     {len(val_loader_conv.dataset)}")
    assert len(train_loader_conv.dataset) + len(val_loader_conv.dataset) == len(X_conv)

    # Pull one batch
    batch_x, batch_y = next(iter(train_loader_conv))
    print(f"Train batch X shape: {batch_x.shape}, dtype: {batch_x.dtype}")
    print(f"Train batch Y shape: {batch_y.shape}, dtype: {batch_y.dtype}")
    assert batch_x.shape[0] == 8, f"Expected batch_size 8, got {batch_x.shape[0]}"
    assert batch_x.dtype == torch.float32

    # Full iteration check
    for _ in train_loader_conv:
        pass
    for _ in val_loader_conv:
        pass
    print("Full pass over train_loader and val_loader completed with zero shape errors!")

    # Test U-Net format
    print("\n--- 2. U-Net Pipeline Integration ---")
    # To swap in U-Net data: pass format_for_model('unet') into create_dataloaders; nothing else changes!
    X_unet, Y_unet = pipeline.format_for_model("unet", window_size=7, horizon=1)
    train_loader_unet, val_loader_unet = create_dataloaders(
        X_unet, Y_unet, val_split=0.2, batch_size=8, shuffle_train=True
    )

    batch_x_u, batch_y_u = next(iter(train_loader_unet))
    print(f"Train batch X shape: {batch_x_u.shape}, dtype: {batch_x_u.dtype}")
    print(f"Train batch Y shape: {batch_y_u.shape}, dtype: {batch_y_u.dtype}")
    assert batch_x_u.shape == (8, 7, 64, 64)
    assert batch_y_u.shape == (8, 1, 64, 64)

    for _ in train_loader_unet:
        pass
    for _ in val_loader_unet:
        pass
    print("Full pass over U-Net train_loader and val_loader completed with zero shape errors!")

    print("\n" + "=" * 75)
    print("PyTorch Dataset & DataLoader Verification Complete!")
    print("=" * 75)
