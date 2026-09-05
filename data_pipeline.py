"""Sea-Ice NetCDF Data Preprocessing Pipeline for Deep Learning Forecasting.

This module provides a modular, production-ready pipeline (`SeaIceDataPipeline`)
for preprocessing raw satellite NetCDF sea-ice concentration data and static land masks
into windowed, normalized NumPy tensors formatted for ConvLSTM or U-Net architectures.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr


# Default logger configuration
def _get_default_logger() -> logging.Logger:
    logger = logging.getLogger("SeaIceDataPipeline")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class SeaIceDataPipeline:
    """End-to-end preprocessing pipeline for sea-ice concentration forecasting.

    Key capabilities:
    1. Loads NetCDF sea-ice datasets (single file, list of files, or glob pattern)
       and static land masks (.nc or .npy).
    2. Enforces spatial dimension matching between land mask and concentration maps.
    3. Cleans data by zeroing out land pixels and imputing missing satellite passes (NaNs)
       with 0.0, recording pixel statistics per timestep.
    4. Normalizes concentration to [0.0, 1.0] using a configurable maximum scale constant.
    5. Preallocates NumPy sequence buffers for sliding-window temporal training.
    6. Formats output tensors with strict shape assertions for ConvLSTM and U-Net models.

    Attributes:
        data_path: Path, glob pattern, or list of paths for sea-ice NetCDF file(s).
        mask_path: Path to static land mask NetCDF or NumPy file.
        variable_name: Name of sea-ice concentration variable in NetCDF (e.g. 'siconc').
        mask_variable_name: Optional explicit name of mask variable in NetCDF.
        logger: Logger instance for progress and cleaning telemetry.
        raw_data: Loaded raw sea-ice NumPy array shaped (Time, Height, Width).
        land_mask: Loaded 2D land mask NumPy array shaped (Height, Width).
        cleaned_data: Cleaned sea-ice NumPy array after masking and NaN imputation.
        normalized_data: Sea-ice array normalized to [0.0, 1.0].
        cleaning_stats: Dictionary containing telemetry on masked and imputed pixels.
    """

    def __init__(
        self,
        data_path: str | Path | list[str | Path],
        mask_path: str | Path,
        variable_name: str = "siconc",
        mask_variable_name: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize pipeline with dataset paths and configuration.

        Args:
            data_path: File path, glob pattern (e.g. 'data/*.nc'), or list of paths to NetCDF files.
            mask_path: Path to land mask file (.nc or .npy).
            variable_name: Variable identifier in the NetCDF dataset (default: 'siconc').
            mask_variable_name: Optional variable name in mask NetCDF (auto-detected if None).
            logger: Custom logger. If None, default pipeline logger is used.
        """
        self.data_path = data_path
        self.mask_path = Path(mask_path)
        self.variable_name = variable_name
        self.mask_variable_name = mask_variable_name
        self.logger = logger or _get_default_logger()

        # Pipeline internal states
        self.raw_data: np.ndarray | None = None
        self.land_mask: np.ndarray | None = None
        self.cleaned_data: np.ndarray | None = None
        self.normalized_data: np.ndarray | None = None
        self.cleaning_stats: dict[str, int | list[int]] = {}

        # Cached sliding-window sequences
        self._cached_X: np.ndarray | None = None
        self._cached_Y: np.ndarray | None = None
        self._cached_window_size: int | None = None
        self._cached_horizon: int | None = None

    def _resolve_data_files(self) -> list[Path]:
        """Resolve data_path into a validated list of existing NetCDF Path objects."""
        files: list[Path] = []

        if isinstance(self.data_path, (list, tuple)):
            for p in self.data_path:
                path_obj = Path(p)
                if not path_obj.is_file():
                    raise FileNotFoundError(f"Specified NetCDF file does not exist: {path_obj}")
                files.append(path_obj)
        elif isinstance(self.data_path, (str, Path)):
            path_str = str(self.data_path)
            # Check if glob pattern
            if any(char in path_str for char in ["*", "?", "["]):
                matched = [Path(p) for p in sorted(glob.glob(path_str))]
                if not matched:
                    raise FileNotFoundError(
                        f"Glob pattern '{path_str}' matched 0 NetCDF files."
                    )
                files.extend(matched)
            else:
                path_obj = Path(path_str)
                if not path_obj.exists():
                    raise FileNotFoundError(f"Specified NetCDF file does not exist: {path_obj}")
                if path_obj.is_dir():
                    matched = sorted(path_obj.glob("*.nc"))
                    if not matched:
                        raise FileNotFoundError(
                            f"Directory '{path_obj}' contains no NetCDF (*.nc) files."
                        )
                    files.extend(matched)
                else:
                    files.append(path_obj)
        else:
            raise TypeError(
                f"Unsupported data_path type: {type(self.data_path)}. Expected str, Path, or list."
            )

        return sorted(files)

    def load(self) -> SeaIceDataPipeline:
        """Load NetCDF sea-ice dataset and static land mask into memory.

        Extracts the ice concentration variable into a NumPy array shaped (Time, Height, Width)
        and verifies that the land mask has matching spatial dimensions (Height, Width).

        Returns:
            self: Enables fluent method chaining.

        Raises:
            FileNotFoundError: If data or mask files are missing.
            KeyError: If variable_name is not present in the NetCDF dataset.
            ValueError: If spatial dimensions do not match or array is not 3D.
        """
        # 1. Resolve and validate data files
        data_files = self._resolve_data_files()
        self.logger.info(
            f"Loading sea-ice data from {len(data_files)} file(s): "
            f"{[f.name for f in data_files]}"
        )

        # 2. Open sea-ice dataset via xarray
        if len(data_files) == 1:
            ds = xr.open_dataset(data_files[0])
        else:
            # Concatenate multiple NetCDF files along time without requiring dask
            datasets = [xr.open_dataset(f) for f in data_files]
            # If datasets share coords, sort by time coordinate if present
            time_dim = "time"
            for candidate in ["time", "Time", "date"]:
                if candidate in datasets[0].dims:
                    time_dim = candidate
                    break
            ds = xr.concat(datasets, dim=time_dim)

        if self.variable_name not in ds.data_vars:
            available = list(ds.data_vars.keys())
            raise KeyError(
                f"Variable '{self.variable_name}' was not found in dataset. "
                f"Available data variables: {available}"
            )

        # Extract ice concentration array
        ice_da = ds[self.variable_name]
        ice_array = ice_da.values.astype(np.float32)

        # Squeeze singleton dimensions if necessary (e.g. (Time, 1, Lat, Lon) -> (Time, Lat, Lon))
        if ice_array.ndim == 4 and ice_array.shape[1] == 1:
            ice_array = np.squeeze(ice_array, axis=1)

        if ice_array.ndim != 3:
            raise ValueError(
                f"Expected sea-ice concentration array to be 3D with shape (Time, Height, Width), "
                f"but got shape {ice_array.shape} (dimensions: {ice_da.dims})"
            )

        total_time, height, width = ice_array.shape
        self.logger.info(
            f"Extracted variable '{self.variable_name}' with shape: "
            f"Time={total_time}, Height={height}, Width={width}"
        )

        # 3. Load and validate land mask
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Land mask file not found at: {self.mask_path}")

        self.logger.info(f"Loading land mask from {self.mask_path}...")
        if self.mask_path.suffix.lower() == ".npy":
            mask_arr = np.load(self.mask_path).astype(np.float32)
        else:
            mask_ds = xr.open_dataset(self.mask_path)
            if self.mask_variable_name and self.mask_variable_name in mask_ds.data_vars:
                mask_var_name = self.mask_variable_name
            elif "mask" in mask_ds.data_vars:
                mask_var_name = "mask"
            elif "land_mask" in mask_ds.data_vars:
                mask_var_name = "land_mask"
            elif len(mask_ds.data_vars) > 0:
                mask_var_name = list(mask_ds.data_vars.keys())[0]
            else:
                raise KeyError(f"No data variables found in mask dataset: {self.mask_path}")

            mask_arr = mask_ds[mask_var_name].values.astype(np.float32)

        # Squeeze singleton dimensions in mask if present
        mask_arr = np.squeeze(mask_arr)

        if mask_arr.ndim != 2:
            raise ValueError(
                f"Land mask must be a 2D spatial grid (Height, Width), but got shape {mask_arr.shape}"
            )

        # Validate spatial dimension match
        if mask_arr.shape != (height, width):
            raise ValueError(
                f"Dimension mismatch between land mask shape {mask_arr.shape} and "
                f"sea-ice spatial dimensions {(height, width)}. They must match exactly."
            )

        self.raw_data = ice_array
        self.land_mask = mask_arr
        self.logger.info("Data and land mask loaded and validated successfully.")
        return self

    def clean(self) -> SeaIceDataPipeline:
        """Clean sea-ice data by applying land mask and imputing NaNs.

        Steps:
        1. Multiplies ice array by the land mask (land=0, ocean=1) to zero out land pixels.
        2. Replaces all NaN values (caused by satellite dropouts/cloud gaps) with 0.0.
        3. Logs detailed telemetry of masked and imputed pixels per timestep.

        Returns:
            self: Enables fluent method chaining.

        Raises:
            RuntimeError: If called before `load()`.
        """
        if self.raw_data is None or self.land_mask is None:
            raise RuntimeError("Pipeline data is not loaded. Call load() before clean().")

        total_time, height, width = self.raw_data.shape
        self.logger.info("Beginning data cleaning and land masking...")

        # 1. Zero out land pixels
        # land_mask is (H, W) where ocean=1, land=0.
        # Broadcast across time: (1, H, W) * (Time, H, W)
        land_mask_2d = (self.land_mask != 0.0).astype(np.float32)
        land_pixels_total = int(np.sum(land_mask_2d == 0.0))

        # Detect non-zero values on land in raw data before masking
        land_broadcast = land_mask_2d[np.newaxis, :, :] == 0.0
        non_zero_land_pixels_per_step = [
            int(np.count_nonzero(self.raw_data[t][land_broadcast[0]]))
            for t in range(total_time)
        ]
        total_land_masked = int(sum(non_zero_land_pixels_per_step))

        # Apply land mask
        masked_array = self.raw_data * land_mask_2d[np.newaxis, :, :]

        # 2. Identify NaNs (missing satellite passes) and impute with 0.0
        nan_mask = np.isnan(masked_array)
        nan_pixels_per_step = [int(np.sum(nan_mask[t])) for t in range(total_time)]
        total_nans_imputed = int(np.sum(nan_mask))

        cleaned_array = np.nan_to_num(masked_array, nan=0.0)

        # 3. Telemetry & logging
        total_pixels = total_time * height * width
        self.cleaning_stats = {
            "total_pixels": total_pixels,
            "static_land_pixels_per_grid": land_pixels_total,
            "total_land_pixels_masked": total_land_masked,
            "total_nans_imputed": total_nans_imputed,
            "nan_pixels_per_step": nan_pixels_per_step,
            "non_zero_land_pixels_per_step": non_zero_land_pixels_per_step,
        }

        self.logger.info(
            f"Data cleaning complete: {total_nans_imputed:,} NaNs imputed to 0.0 "
            f"({(total_nans_imputed / total_pixels) * 100:.2f}% of total pixels)."
        )
        self.logger.info(
            f"Land masking complete: {total_land_masked:,} non-zero land pixels zeroed out."
        )

        # Check for any anomalies (e.g., if a timestep is completely missing)
        for t, n_nan in enumerate(nan_pixels_per_step):
            if n_nan > (height * width * 0.5):
                self.logger.warning(
                    f"Timestep {t} has high missing data rate: {n_nan} / {height * width} "
                    f"({(n_nan / (height * width)) * 100:.1f}%) NaNs imputed."
                )

        self.cleaned_data = cleaned_array
        return self

    def normalize(self, max_value: float = 100.0) -> SeaIceDataPipeline:
        """Scale sea-ice concentration values to [0.0, 1.0].

        Divides the cleaned array by `max_value` (e.g., 100.0 for percentage-based concentration)
        and clips values strictly to [0.0, 1.0] to guarantee range bounds against any sensor spikes.

        Args:
            max_value: Configurable maximum valid concentration scale (default: 100.0).

        Returns:
            self: Enables fluent method chaining.

        Raises:
            RuntimeError: If called before `clean()`.
            ValueError: If `max_value` <= 0.
        """
        if self.cleaned_data is None:
            raise RuntimeError("Data has not been cleaned. Call clean() before normalize().")

        if max_value <= 0.0:
            raise ValueError(f"max_value must be strictly positive, got {max_value}")

        self.logger.info(f"Normalizing concentration values by max_value={max_value}...")
        scaled = self.cleaned_data / np.float32(max_value)
        # Clip to ensure numerical safety against minor overshoots
        normalized = np.clip(scaled, 0.0, 1.0).astype(np.float32)

        # Range assertions
        min_val = float(np.min(normalized))
        max_val = float(np.max(normalized))
        if min_val < 0.0 or max_val > 1.0:
            raise ValueError(
                f"Normalized data out of expected bounds [0.0, 1.0]. Got [{min_val}, {max_val}]"
            )

        if np.isnan(normalized).any():
            raise ValueError("Normalized array contains unexpected NaNs.")

        self.logger.info(
            f"Normalization complete. Array range: [{min_val:.4f}, {max_val:.4f}]."
        )
        self.normalized_data = normalized
        return self

    def create_sequences(
        self,
        window_size: int = 7,
        horizon: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate sliding-window training sequences with preallocated NumPy buffers.

        Given a continuous sequence of daily concentration maps of length T:
        - window_size (W): number of consecutive past days provided as input X.
        - horizon (H): forecasting step ahead into the future (H=1 predicts day W+1).

        For any valid sample index i:
            X[i] = data[i : i + W]                   (shape: W, Height, Width)
            Y[i] = data[i + W + H - 1]               (shape: Height, Width)

        Number of valid samples:
            num_samples = T - W - H + 1

        Boundary condition check:
            Maximum index i = num_samples - 1 = T - W - H.
            Target index for last sample = (T - W - H) + W + H - 1 = T - 1 (the final timestep).
            No out-of-bounds slicing or off-by-one errors can occur.

        Args:
            window_size: Number of past time steps per input sequence (default: 7).
            horizon: Forecast horizon steps ahead (default: 1, next day).

        Returns:
            X: Input sequences array of shape (Num_Samples, window_size, Height, Width).
            Y: Target maps array of shape (Num_Samples, Height, Width).

        Raises:
            RuntimeError: If called before `normalize()`.
            ValueError: If window_size or horizon < 1, or if sequence is too short.
        """
        if self.normalized_data is None:
            raise RuntimeError(
                "Data has not been normalized. Call normalize() before create_sequences()."
            )

        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        total_timesteps, height, width = self.normalized_data.shape
        required_timesteps = window_size + horizon

        if total_timesteps < required_timesteps:
            raise ValueError(
                f"Total timesteps ({total_timesteps}) is smaller than the minimum required "
                f"timesteps for window_size={window_size} and horizon={horizon} "
                f"({required_timesteps}). Cannot generate sequences."
            )

        num_samples = total_timesteps - window_size - horizon + 1

        self.logger.info(
            f"Generating sliding-window sequences: total_timesteps={total_timesteps}, "
            f"window_size={window_size}, horizon={horizon} -> num_samples={num_samples}."
        )

        # Preallocate NumPy arrays upfront for memory efficiency on long series
        X = np.empty((num_samples, window_size, height, width), dtype=np.float32)
        Y = np.empty((num_samples, height, width), dtype=np.float32)

        data = self.normalized_data

        # Explicit sliding-window loop:
        # Sample index i:
        #   X[i]: days from i up to i + window_size - 1 (length window_size)
        #   Y[i]: forecast target day at index i + window_size + horizon - 1
        # Example with window_size=7, horizon=1:
        #   i=0: X uses [0..6] (7 days), predicts day 7 (the 8th day)
        #   i=1: X uses [1..7] (7 days), predicts day 8 (the 9th day)
        for i in range(num_samples):
            X[i] = data[i : i + window_size]
            Y[i] = data[i + window_size + horizon - 1]

        self._cached_X = X
        self._cached_Y = Y
        self._cached_window_size = window_size
        self._cached_horizon = horizon

        return X, Y

    def format_for_model(
        self,
        model_type: Literal["convlstm", "unet"] = "convlstm",
        window_size: int = 7,
        horizon: int = 1,
        squeeze_target_time: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Format sliding-window sequences into architecture-specific tensor shapes.

        Supported model types:
        1. "convlstm":
           - X: (Batch, Time_Steps, Channels, Height, Width) where Channels = 1.
           - Y: (Batch, 1, 1, Height, Width) by default, or (Batch, 1, Height, Width)
                if squeeze_target_time=True.
        2. "unet":
           - X: (Batch, Channels, Height, Width) where Channels = Time_Steps
                (past daily maps are stacked along the channel axis).
           - Y: (Batch, 1, Height, Width) representing single-channel forecast map.

        Args:
            model_type: Architecture type: 'convlstm' or 'unet'.
            window_size: Number of past time steps per input sequence (default: 7).
            horizon: Forecast horizon steps ahead (default: 1).
            squeeze_target_time: If True, ConvLSTM target Y drops the time dimension
                                 and is shaped (Batch, 1, H, W). Default is False.

        Returns:
            X_model: Formatted model input tensor.
            Y_model: Formatted model target tensor.

        Raises:
            ValueError: If model_type is unsupported or if output shape fails strict assertions.
        """
        # Generate or reuse cached sequences
        if (
            self._cached_X is None
            or self._cached_Y is None
            or self._cached_window_size != window_size
            or self._cached_horizon != horizon
        ):
            X, Y = self.create_sequences(window_size=window_size, horizon=horizon)
        else:
            X, Y = self._cached_X, self._cached_Y

        num_samples, w_size, height, width = X.shape
        model_key = model_type.strip().lower()

        if model_key == "convlstm":
            # ConvLSTM expects 5D tensors: (Batch, Time_Steps, Channels, Height, Width)
            # Add singleton channel dimension to X: (N, W, 1, H, W)
            X_model = np.expand_dims(X, axis=2)

            if squeeze_target_time:
                # 4D target: (Batch, Channels, Height, Width) with Channels = 1
                Y_model = np.expand_dims(Y, axis=1)
                expected_Y_shape = (num_samples, 1, height, width)
            else:
                # 5D target: (Batch, Time_Steps, Channels, Height, Width) with Time_Steps=1, Channels=1
                Y_model = np.expand_dims(Y, axis=(1, 2))
                expected_Y_shape = (num_samples, 1, 1, height, width)

            expected_X_shape = (num_samples, window_size, 1, height, width)

            # Strict shape assertions
            if X_model.shape != expected_X_shape:
                raise ValueError(
                    f"ConvLSTM input shape assertion failed! "
                    f"Expected shape: {expected_X_shape}, but got: {X_model.shape}"
                )
            if Y_model.shape != expected_Y_shape:
                raise ValueError(
                    f"ConvLSTM target shape assertion failed! "
                    f"Expected shape: {expected_Y_shape}, but got: {Y_model.shape}"
                )

        elif model_key == "unet":
            # U-Net 2D expects 4D tensors: (Batch, Channels, Height, Width)
            # The input window's Time_Steps are treated as input Channels
            X_model = X  # Already shape (num_samples, window_size, height, width)

            # Add singleton channel dimension to target Y: (Batch, 1, Height, Width)
            Y_model = np.expand_dims(Y, axis=1)

            expected_X_shape = (num_samples, window_size, height, width)
            expected_Y_shape = (num_samples, 1, height, width)

            # Strict shape assertions
            if X_model.shape != expected_X_shape:
                raise ValueError(
                    f"U-Net input shape assertion failed! "
                    f"Expected shape: {expected_X_shape}, but got: {X_model.shape}"
                )
            if Y_model.shape != expected_Y_shape:
                raise ValueError(
                    f"U-Net target shape assertion failed! "
                    f"Expected shape: {expected_Y_shape}, but got: {Y_model.shape}"
                )

        else:
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. Valid options are 'convlstm' or 'unet'."
            )

        self.logger.info(
            f"Successfully formatted tensors for '{model_type}': "
            f"X_train={X_model.shape}, Y_train={Y_model.shape}"
        )
        return X_model, Y_model

    def run_pipeline(
        self,
        model_type: Literal["convlstm", "unet"] = "convlstm",
        max_value: float = 100.0,
        window_size: int = 7,
        horizon: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience method to execute the entire preprocessing pipeline end-to-end.

        Args:
            model_type: Target architecture ('convlstm' or 'unet').
            max_value: Maximum concentration scale for normalization (default: 100.0).
            window_size: Consecutive input days (default: 7).
            horizon: Prediction horizon steps ahead (default: 1).

        Returns:
            Tuple of (X_train, Y_train) formatted and ready for training.
        """
        return (
            self.load()
            .clean()
            .normalize(max_value=max_value)
            .format_for_model(model_type=model_type, window_size=window_size, horizon=horizon)
        )


if __name__ == "__main__":
    # Self-contained demonstration on sample synthetic data
    base_dir = Path(__file__).parent / "data"
    sea_ice_pattern = str(base_dir / "sea_ice" / "*.nc")
    mask_file = str(base_dir / "land_mask.nc")

    print("=" * 70)
    print("Running SeaIceDataPipeline End-to-End Demonstration")
    print("=" * 70)

    pipeline = SeaIceDataPipeline(
        data_path=sea_ice_pattern,
        mask_path=mask_file,
        variable_name="siconc",
    )

    # 1. Load, Clean, Normalize
    pipeline.load()
    pipeline.clean()
    pipeline.normalize(max_value=100.0)

    # 2. Format for ConvLSTM
    print("\n--- ConvLSTM Model Formatting ---")
    X_convlstm, Y_convlstm = pipeline.format_for_model(
        model_type="convlstm", window_size=7, horizon=1
    )
    print(f"X_convlstm shape: {X_convlstm.shape} (Batch, Time_Steps, Channels, H, W)")
    print(f"Y_convlstm shape: {Y_convlstm.shape} (Batch, Time_Steps, Channels, H, W)")
    print(f"ConvLSTM range:   [{X_convlstm.min():.4f}, {X_convlstm.max():.4f}]")
    print(f"ConvLSTM NaNs:    {np.isnan(X_convlstm).sum()} (X), {np.isnan(Y_convlstm).sum()} (Y)")

    # 3. Format for U-Net
    print("\n--- U-Net Model Formatting ---")
    X_unet, Y_unet = pipeline.format_for_model(
        model_type="unet", window_size=7, horizon=1
    )
    print(f"X_unet shape:     {X_unet.shape} (Batch, Channels/Time_Steps, H, W)")
    print(f"Y_unet shape:     {Y_unet.shape} (Batch, Channels, H, W)")
    print(f"U-Net range:      [{X_unet.min():.4f}, {X_unet.max():.4f}]")
    print(f"U-Net NaNs:       {np.isnan(X_unet).sum()} (X), {np.isnan(Y_unet).sum()} (Y)")

    print("\n" + "=" * 70)
    print("Pipeline Verification Successful!")
    print("=" * 70)
