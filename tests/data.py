"""Sea-Ice Data Pipeline: NetCDF loading, masking, sliding-window Dataset, time-split DataLoaders.

Loads NSIDC G02202 v6 daily Antarctic NetCDF files, masks invalid pixels (NaN → 0),
normalises to [0, 1], and builds PyTorch sliding-window datasets split strictly by time.

The concentration variable `cdr_seaice_conc` is stored as a float fraction in [0, 1].
NaN pixels represent land, pole-hole, and missing data — all are set to 0.0.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset, Subset

from config import Config


logger = logging.getLogger("SeaIceData")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(levelname)s — %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  NetCDF Loading
# ═══════════════════════════════════════════════════════════════════════════════

def discover_netcdf_files(data_root: str) -> list[str]:
    """Find all Antarctic NetCDF files under *data_root*, sorted by filename (date order).

    Looks for subdirectories matching ``Antarctic_NetCDF_*`` and globs ``sic_pss25_*.nc``
    inside each. Returns an absolute-path list sorted lexicographically (which is also
    chronological because filenames embed ``YYYYMMDD``).

    Raises:
        FileNotFoundError: If no NetCDF files are found under *data_root*.
    """
    pattern = os.path.join(data_root, "Antarctic_NetCDF_*", "sic_pss25_*.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Antarctic NetCDF files found matching '{pattern}'. "
            f"Check that data_root='{data_root}' contains Antarctic_NetCDF_YYYY/ subdirs."
        )
    logger.info(f"Discovered {len(files)} NetCDF files in {data_root}")
    return files


def load_and_preprocess(
    cfg: Config,
) -> np.ndarray:
    """Load all NetCDF files, mask invalid pixels, normalise to [0, 1].

    Steps:
        1. Discover and sort all ``.nc`` files under ``cfg.data_root``.
        2. Open each file with xarray and extract ``cdr_seaice_conc`` (shape: 1×H×W).
        3. Concatenate along the time axis → (T, H, W).
        4. **Spatial consistency check**: assert all files share the same (H, W).
        5. Replace NaN (land / pole-hole / missing) with 0.0.
        6. Clip to [0, 1] for safety (handles any micro-overshoots).
        7. Optionally spatially downsample by ``cfg.downsample_factor``.

    Args:
        cfg: Pipeline configuration.

    Returns:
        data: float32 NumPy array of shape ``(T, H, W)`` with values in [0, 1].
    """
    files = discover_netcdf_files(cfg.data_root)

    arrays: list[np.ndarray] = []
    ref_shape: tuple[int, int] | None = None

    for i, fpath in enumerate(files):
        ds = xr.open_dataset(fpath)

        if cfg.concentration_var not in ds.data_vars:
            available = list(ds.data_vars.keys())
            ds.close()
            raise KeyError(
                f"Variable '{cfg.concentration_var}' not found in {fpath}. "
                f"Available: {available}"
            )

        da = ds[cfg.concentration_var]  # (time=1, y, x)
        arr = da.values.astype(np.float32)  # (1, H, W)
        ds.close()

        # Squeeze any singleton batch-like dims to get (1, H, W)
        while arr.ndim > 3:
            arr = arr.squeeze(0)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]  # (1, H, W)

        spatial = arr.shape[1:]
        if ref_shape is None:
            ref_shape = spatial
            logger.info(
                f"Reference spatial shape from first file: (y={ref_shape[0]}, x={ref_shape[1]})"
            )
        elif spatial != ref_shape:
            raise ValueError(
                f"Spatial shape mismatch: file {os.path.basename(fpath)} has {spatial}, "
                f"but first file established {ref_shape}."
            )

        arrays.append(arr)

    # Concatenate → (T, H, W)
    data = np.concatenate(arrays, axis=0)
    logger.info(f"Concatenated data shape: (T={data.shape[0]}, H={data.shape[1]}, W={data.shape[2]})")

    # Count NaN pixels before masking
    nan_count = int(np.isnan(data).sum())
    total_pixels = int(data.size)
    logger.info(
        f"NaN pixels (land/pole-hole/missing): {nan_count:,} / {total_pixels:,} "
        f"({nan_count / total_pixels * 100:.1f}%)"
    )

    # Mask: NaN → 0.0, clip to [0, 1]
    data = np.nan_to_num(data, nan=0.0)
    data = np.clip(data, 0.0, 1.0)

    # Verify no NaNs remain
    assert not np.isnan(data).any(), "Data still contains NaN after masking!"
    logger.info(f"Value range after masking: [{data.min():.4f}, {data.max():.4f}]")

    # Optional spatial downsampling (area-average pooling)
    if cfg.downsample_factor > 1:
        f = cfg.downsample_factor
        T, H, W = data.shape
        new_H = H // f
        new_W = W // f
        # Crop to exact multiple
        data = data[:, :new_H * f, :new_W * f]
        # Reshape and mean-pool
        data = data.reshape(T, new_H, f, new_W, f).mean(axis=(2, 4))
        logger.info(f"Downsampled {f}x: ({H}, {W}) → ({new_H}, {new_W})")

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Sliding-Window PyTorch Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class SeaIceWindowDataset(Dataset):
    """Sliding-window dataset for ConvLSTM encoder-decoder training.

    Given a continuous daily concentration cube of shape ``(T, H, W)``:
        - **input**:  ``data[idx : idx + N]``  → shape ``(N, 1, H, W)``  (channel dim added)
        - **target**: ``data[idx + N : idx + N + K]`` → shape ``(K, 1, H, W)``

    Total valid windows = ``T - N - K + 1``.

    Boundary proof (last valid idx = T - N - K):
        - Input ends at  ``(T - N - K) + N - 1 = T - K - 1``   ✓
        - Target ends at ``(T - N - K) + N + K - 1 = T - 1``   ✓  (last timestep)

    The dataset stores a reference to the full NumPy array (no copies).
    ``__getitem__`` slices and wraps into ``torch.float32`` tensors.
    """

    def __init__(self, data: np.ndarray, N: int, K: int) -> None:
        """Initialise the sliding-window dataset.

        Args:
            data: Full concentration cube, shape ``(T, H, W)``, float32, values in [0,1].
            N: Number of input (past) frames.
            K: Number of target (future) frames.

        Raises:
            ValueError: If data is too short for the chosen N + K.
        """
        if data.ndim != 3:
            raise ValueError(f"Expected 3D data (T, H, W), got shape {data.shape}")

        T = data.shape[0]
        min_len = N + K
        if T < min_len:
            raise ValueError(
                f"Data has {T} timesteps but N={N} + K={K} = {min_len} are required. "
                f"Provide more data or reduce N/K."
            )

        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.N = N
        self.K = K
        self.num_windows = T - N - K + 1

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (input, target) tensors for window *idx*.

        Returns:
            input_seq:  shape ``(N, 1, H, W)`` — past N frames with channel dim.
            target_seq: shape ``(K, 1, H, W)`` — next K frames with channel dim.
        """
        # Slice from the backing NumPy array
        x_np = self.data[idx : idx + self.N]       # (N, H, W)
        y_np = self.data[idx + self.N : idx + self.N + self.K]  # (K, H, W)

        # Add channel dim: (T, H, W) → (T, 1, H, W)
        x = torch.from_numpy(x_np).unsqueeze(1)  # (N, 1, H, W)
        y = torch.from_numpy(y_np).unsqueeze(1)  # (K, 1, H, W)

        return x, y


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Time-Based Splitting & DataLoader Construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    cfg: Config,
    data: np.ndarray,
) -> tuple[DataLoader, DataLoader, DataLoader, SeaIceWindowDataset]:
    """Build train / val / test DataLoaders from the full concentration cube.

    Splitting is strictly chronological: the first ``train_frac`` windows go to
    training, the next ``val_frac`` to validation, and the remainder to test.
    No shuffling across time boundaries.

    Args:
        cfg: Pipeline configuration.
        data: Preprocessed concentration cube, shape ``(T, H, W)``.

    Returns:
        train_loader: Training DataLoader (shuffled within training window indices).
        val_loader: Validation DataLoader (sequential).
        test_loader: Test DataLoader (sequential).
        full_dataset: The underlying ``SeaIceWindowDataset`` (useful for inspection).
    """
    full_dataset = SeaIceWindowDataset(data, N=cfg.N, K=cfg.K)
    total = len(full_dataset)

    train_end = int(total * cfg.train_frac)
    val_end = train_end + int(total * cfg.val_frac)

    # Ensure at least 1 sample in each split
    if train_end < 1:
        train_end = 1
    if val_end <= train_end:
        val_end = train_end + 1
    if val_end >= total:
        val_end = total - 1  # leave at least 1 test sample

    train_indices = list(range(0, train_end))
    val_indices = list(range(train_end, val_end))
    test_indices = list(range(val_end, total))

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)

    # ── Sanity-check printout ──
    sample_x, sample_y = full_dataset[0]
    print("\n" + "=" * 60)
    print("Data Pipeline Summary")
    print("=" * 60)
    print(f"  Total timesteps (T):    {data.shape[0]}")
    print(f"  Spatial grid (H × W):   {data.shape[1]} × {data.shape[2]}")
    print(f"  Input window (N):       {cfg.N}")
    print(f"  Forecast horizon (K):   {cfg.K}")
    print(f"  Total sliding windows:  {total}")
    print(f"  Train samples:          {len(train_indices)} ({len(train_indices)/total*100:.1f}%)")
    print(f"  Val samples:            {len(val_indices)} ({len(val_indices)/total*100:.1f}%)")
    print(f"  Test samples:           {len(test_indices)} ({len(test_indices)/total*100:.1f}%)")
    print(f"  Input tensor shape:     {tuple(sample_x.shape)}  (N, C, H, W)")
    print(f"  Target tensor shape:    {tuple(sample_y.shape)}  (K, C, H, W)")
    print(f"  Batch size:             {cfg.batch_size}")
    print(f"  Train batches/epoch:    {len(train_loader)}")
    print(f"  Val batches/epoch:      {len(val_loader)}")
    print(f"  Test batches:           {len(test_loader)}")
    print("=" * 60 + "\n")

    return train_loader, val_loader, test_loader, full_dataset


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Convenience one-call entry point
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(
    cfg: Config,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Full data preparation in one call: load → preprocess → split → DataLoaders.

    Args:
        cfg: Pipeline configuration.

    Returns:
        train_loader, val_loader, test_loader, raw_data_cube (for inspection).
    """
    data = load_and_preprocess(cfg)
    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg, data)
    return train_loader, val_loader, test_loader, data


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = Config()
    print(cfg.summary())
    train_loader, val_loader, test_loader, data = prepare_data(cfg)

    # Pull one batch and print shapes
    bx, by = next(iter(train_loader))
    print(f"First train batch — X: {bx.shape}, Y: {by.shape}, dtype: {bx.dtype}")
