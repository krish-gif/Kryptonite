"""Sea-Ice Data Pipeline — Caching and Processing.

This script processes raw NSIDC NetCDF files into PyTorch tensors and
caches them to persistent storage so the data preprocessing only runs once.
"""

import os
import glob
import logging
import torch
import xarray as xr
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("DataCache")

def build_and_cache_dataset(
    data_root: str,
    cache_dir: str,
    N: int = 10,
    K: int = 3,
    train_frac: float = 0.8,
    val_frac: float = 0.1
):
    """Load, process, window, split chronologically, and cache data to disk."""
    
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Looking for NetCDF files in {data_root}")
    
    # 1. Load and concatenate files chronologically
    pattern = os.path.join(data_root, "**", "*.nc")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {data_root}")
        
    logger.info(f"Found {len(files)} files. Loading...")
    
    # Extract dates from filenames or metadata (here we rely on the time dim in the dataset)
    # Using a loop since dask is not installed for open_mfdataset
    datasets = []
    for f in files:
        datasets.append(xr.open_dataset(f))
    ds = xr.concat(datasets, dim='time')
    
    # The variable is 'cdr_seaice_conc'
    var_name = 'cdr_seaice_conc'
    if var_name not in ds.variables:
         raise ValueError(f"Variable '{var_name}' not found in dataset. Available: {list(ds.variables.keys())}")
            
    dates = ds.time.values
    logger.info(f"Loaded dataset: {len(dates)} timesteps from {dates[0]} to {dates[-1]}")
    logger.info(f"Spatial shape: (y={ds.y.size}, x={ds.x.size})")
    
    # 2. Mask and Normalize
    logger.info("Applying masking and normalization...")
    # Read into memory as numpy array
    data_array = ds[var_name].values  # shape (T, Y, X)
    
    # NaN represents land, missing data, pole hole. Convert to 0.0
    nan_mask = np.isnan(data_array)
    logger.info(f"Masking {nan_mask.sum():,} NaN pixels to 0.0")
    data_array[nan_mask] = 0.0
    
    # Clip to [0.0, 1.0] (the data is a fraction)
    data_array = np.clip(data_array, 0.0, 1.0)
    
    logger.info(f"Final data range: [{data_array.min():.4f}, {data_array.max():.4f}]")
    
    # 3. Build sliding windows
    T, H, W = data_array.shape
    num_windows = T - N - K + 1
    if num_windows <= 0:
        raise ValueError(f"Not enough timesteps ({T}) for N={N} and K={K}")
        
    logger.info(f"Building {num_windows} sliding windows (N={N}, K={K})...")
    
    # Use PyTorch for tensors
    tensor_data = torch.from_numpy(data_array).float()  # (T, H, W)
    
    X_all = torch.zeros((num_windows, N, 1, H, W), dtype=torch.float32)
    Y_all = torch.zeros((num_windows, K, 1, H, W), dtype=torch.float32)
    window_dates = []
    
    for i in range(num_windows):
        X_all[i, :, 0, :, :] = tensor_data[i : i + N]
        Y_all[i, :, 0, :, :] = tensor_data[i + N : i + N + K]
        # Track the start date of this window
        window_dates.append(dates[i])
        
    window_dates = np.array(window_dates)
    
    # 4. Strict chronological split
    train_end = int(num_windows * train_frac)
    val_end = int(num_windows * (train_frac + val_frac))
    
    X_train, Y_train = X_all[:train_end], Y_all[:train_end]
    train_dates = window_dates[:train_end]
    
    X_val, Y_val = X_all[train_end:val_end], Y_all[train_end:val_end]
    val_dates = window_dates[train_end:val_end]
    
    X_test, Y_test = X_all[val_end:], Y_all[val_end:]
    test_dates = window_dates[val_end:]
    
    # 5. Save to persistent cache
    logger.info(f"Caching to {cache_dir}...")
    torch.save(X_train, os.path.join(cache_dir, "X_train.pt"))
    torch.save(Y_train, os.path.join(cache_dir, "Y_train.pt"))
    torch.save(X_val, os.path.join(cache_dir, "X_val.pt"))
    torch.save(Y_val, os.path.join(cache_dir, "Y_val.pt"))
    torch.save(X_test, os.path.join(cache_dir, "X_test.pt"))
    torch.save(Y_test, os.path.join(cache_dir, "Y_test.pt"))
    
    # Save dates for reference
    np.save(os.path.join(cache_dir, "train_dates.npy"), train_dates)
    np.save(os.path.join(cache_dir, "val_dates.npy"), val_dates)
    np.save(os.path.join(cache_dir, "test_dates.npy"), test_dates)
    
    # 6. Final verification report
    print("\n" + "="*60)
    print("PIPELINE COMPLETION REPORT")
    print("="*60)
    print(f"Total timesteps processed: {T}")
    print(f"Total valid sliding windows: {num_windows}")
    print(f"Data value range (normalized): [{data_array.min():.4f}, {data_array.max():.4f}]")
    print("-" * 60)
    print("SPLIT DETAILS:")
    if len(train_dates) > 0:
        print(f"  Train : {X_train.shape[0]} sequences | Shape: {X_train.shape} | Dates: {str(train_dates[0])[:10]} to {str(train_dates[-1])[:10]}")
    if len(val_dates) > 0:
        print(f"  Val   : {X_val.shape[0]} sequences | Shape: {X_val.shape} | Dates: {str(val_dates[0])[:10]} to {str(val_dates[-1])[:10]}")
    if len(test_dates) > 0:
        print(f"  Test  : {X_test.shape[0]} sequences | Shape: {X_test.shape} | Dates: {str(test_dates[0])[:10]} to {str(test_dates[-1])[:10]}")
    print("="*60 + "\n")
    
if __name__ == "__main__":
    # In a local environment, this targets the relative path. 
    # In a Studio, update this path to the Studio's teamspace directory if needed.
    data_root_dir = "data/South" 
    cache_directory = "cache"
    
    build_and_cache_dataset(
        data_root=data_root_dir,
        cache_dir=cache_directory,
        N=10,
        K=3
    )
