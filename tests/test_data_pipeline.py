"""Comprehensive pytest test suite for SeaIceDataPipeline."""

from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from data_pipeline import SeaIceDataPipeline


@pytest.fixture
def sample_dataset_fixtures(tmp_path: Path):
    """Create lightweight, deterministic NetCDF datasets for testing."""
    height, width = 16, 16
    total_days = 20

    lats = np.linspace(60.0, 90.0, height, dtype=np.float32)
    lons = np.linspace(-180.0, 180.0, width, dtype=np.float32)
    time_coords = np.arange(total_days)

    # 1. Land mask: top half ocean (1), bottom half land (0)
    land_mask = np.ones((height, width), dtype=np.float32)
    land_mask[height // 2 :, :] = 0.0

    mask_nc_path = tmp_path / "test_land_mask.nc"
    mask_ds = xr.Dataset(
        data_vars={"mask": (["lat", "lon"], land_mask)},
        coords={"lat": lats, "lon": lons},
    )
    mask_ds.to_netcdf(mask_nc_path)

    mask_npy_path = tmp_path / "test_land_mask.npy"
    np.save(mask_npy_path, land_mask)

    # 2. Sea-ice data: time series where pixel value at time t is (t * 5.0)
    # This creates a deterministic progression: day 0 has 0, day 1 has 5, day 2 has 10, etc.
    raw_ice = np.zeros((total_days, height, width), dtype=np.float32)
    for t in range(total_days):
        raw_ice[t, :, :] = float(t * 5.0)

    # Inject known NaNs into ocean region on day 3 at (0, 0) and (0, 1)
    raw_ice[3, 0, 0] = np.nan
    raw_ice[3, 0, 1] = np.nan

    # Inject non-zero noise into land region on day 2 to test masking
    raw_ice[2, height - 1, width - 1] = 99.0

    data_nc_path = tmp_path / "test_siconc.nc"
    data_ds = xr.Dataset(
        data_vars={"siconc": (["time", "lat", "lon"], raw_ice)},
        coords={"time": time_coords, "lat": lats, "lon": lons},
    )
    data_ds.to_netcdf(data_nc_path)

    return {
        "data_path": data_nc_path,
        "mask_nc_path": mask_nc_path,
        "mask_npy_path": mask_npy_path,
        "height": height,
        "width": width,
        "total_days": total_days,
        "land_mask": land_mask,
        "raw_ice": raw_ice,
    }


def test_missing_data_file_raises(tmp_path: Path):
    """Pipeline must raise FileNotFoundError if data path does not exist."""
    pipeline = SeaIceDataPipeline(
        data_path=tmp_path / "non_existent.nc",
        mask_path=tmp_path / "mask.nc",
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        pipeline.load()


def test_empty_glob_pattern_raises(tmp_path: Path):
    """Pipeline must raise FileNotFoundError if glob matches 0 files."""
    pipeline = SeaIceDataPipeline(
        data_path=str(tmp_path / "*.nc"),
        mask_path=tmp_path / "mask.nc",
    )
    with pytest.raises(FileNotFoundError, match="matched 0 NetCDF files"):
        pipeline.load()


def test_missing_mask_file_raises(sample_dataset_fixtures):
    """Pipeline must raise FileNotFoundError if land mask path does not exist."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=Path("non_existent_mask.nc"),
    )
    with pytest.raises(FileNotFoundError, match="Land mask file not found"):
        pipeline.load()


def test_variable_not_found_raises(sample_dataset_fixtures):
    """Pipeline must raise KeyError if requested variable is not in the dataset."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
        variable_name="non_existent_var",
    )
    with pytest.raises(KeyError, match="Variable 'non_existent_var' was not found"):
        pipeline.load()


def test_spatial_dimension_mismatch_raises(sample_dataset_fixtures, tmp_path: Path):
    """Pipeline must raise ValueError if land mask spatial dimensions do not match ice data."""
    # Create mismatched 8x8 mask when data is 16x16
    mismatched_mask = np.ones((8, 8), dtype=np.float32)
    mismatched_path = tmp_path / "mismatched_mask.npy"
    np.save(mismatched_path, mismatched_mask)

    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=mismatched_path,
    )
    with pytest.raises(ValueError, match="Dimension mismatch"):
        pipeline.load()


def test_load_with_numpy_mask(sample_dataset_fixtures):
    """Pipeline must load successfully when mask is a .npy file."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_npy_path"],
    )
    pipeline.load()
    assert pipeline.raw_data is not None
    assert pipeline.land_mask is not None
    assert pipeline.land_mask.shape == (16, 16)


def test_clean_masks_land_and_imputes_nans(sample_dataset_fixtures):
    """Cleaning step must zero out land pixels, replace NaNs with 0.0, and record stats."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
    )
    pipeline.load().clean()

    cleaned = pipeline.cleaned_data
    assert cleaned is not None
    assert not np.isnan(cleaned).any(), "Cleaned data still contains NaNs!"

    # Known NaNs were at day 3, (0, 0) and (0, 1) -> must now be 0.0
    assert cleaned[3, 0, 0] == 0.0
    assert cleaned[3, 0, 1] == 0.0

    # Land region (lower half) must be strictly 0.0 across all timesteps
    h = sample_dataset_fixtures["height"]
    assert np.all(cleaned[:, h // 2 :, :] == 0.0)

    # Verify telemetry stats
    assert pipeline.cleaning_stats["total_nans_imputed"] == 2
    assert pipeline.cleaning_stats["total_land_pixels_masked"] >= 1


def test_normalize_scale_and_bounds(sample_dataset_fixtures):
    """Normalization must scale data to [0.0, 1.0] using max_value constant."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
    )
    pipeline.load().clean().normalize(max_value=100.0)

    norm = pipeline.normalized_data
    assert norm is not None
    assert np.min(norm) >= 0.0
    assert np.max(norm) <= 1.0
    assert not np.isnan(norm).any()

    # Test invalid max_value
    with pytest.raises(ValueError, match="max_value must be strictly positive"):
        pipeline.normalize(max_value=0.0)


def test_sliding_window_logic_and_boundary_conditions(sample_dataset_fixtures):
    """Verify exact sliding-window slices and lack of off-by-one errors."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
    )
    pipeline.load().clean().normalize(max_value=100.0)

    total_days = sample_dataset_fixtures["total_days"]  # 20
    window_size = 7
    horizon = 1

    # Expected num samples: 20 - 7 - 1 + 1 = 13
    expected_samples = total_days - window_size - horizon + 1
    X, Y = pipeline.create_sequences(window_size=window_size, horizon=horizon)

    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert X.shape == (expected_samples, window_size, 16, 16)
    assert Y.shape == (expected_samples, 16, 16)

    # In our deterministic data fixture:
    # Ocean pixel (0, 5) at day t has raw value t * 5.0, normalized by 100 -> (t * 0.05)
    # Sample 0:
    #   Input X[0] must contain days [0, 1, 2, 3, 4, 5, 6]
    #   Target Y[0] must be day 7 (value = 7 * 0.05 = 0.35)
    for day_offset in range(window_size):
        expected_val = day_offset * 0.05
        # Skip day 3 index 0,0 since that was NaN, check (0, 5)
        np.testing.assert_allclose(X[0, day_offset, 0, 5], expected_val, atol=1e-5)

    np.testing.assert_allclose(Y[0, 0, 5], 7 * 0.05, atol=1e-5)

    # Last sample (index expected_samples - 1 = 12):
    #   Input X[12] must contain days [12, 13, 14, 15, 16, 17, 18]
    #   Target Y[12] must be day 19 (the final day, value = 19 * 0.05 = 0.95)
    last_idx = expected_samples - 1
    for day_offset in range(window_size):
        expected_val = (last_idx + day_offset) * 0.05
        np.testing.assert_allclose(X[last_idx, day_offset, 0, 5], expected_val, atol=1e-5)

    np.testing.assert_allclose(Y[last_idx, 0, 5], 19 * 0.05, atol=1e-5)


def test_sliding_window_custom_horizon(sample_dataset_fixtures):
    """Verify sequence generation with multi-day horizon (e.g., horizon=3)."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
    )
    pipeline.load().clean().normalize(max_value=100.0)

    total_days = 20
    window_size = 5
    horizon = 3

    # Expected num samples: 20 - 5 - 3 + 1 = 13
    expected_samples = total_days - window_size - horizon + 1
    X, Y = pipeline.create_sequences(window_size=window_size, horizon=horizon)

    assert X.shape == (expected_samples, 5, 16, 16)
    assert Y.shape == (expected_samples, 16, 16)

    # Sample 0: X uses [0, 1, 2, 3, 4], target Y is day (0 + 5 + 3 - 1) = 7
    np.testing.assert_allclose(Y[0, 0, 5], 7 * 0.05, atol=1e-5)


def test_sliding_window_insufficient_length_raises(sample_dataset_fixtures):
    """Pipeline must raise ValueError if timesteps < window_size + horizon."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
    )
    pipeline.load().clean().normalize()

    with pytest.raises(ValueError, match="smaller than the minimum required"):
        pipeline.create_sequences(window_size=15, horizon=10)  # 15+10 = 25 > 20


def test_convlstm_format_and_assertions(sample_dataset_fixtures):
    """ConvLSTM output formatting must produce 5D tensors with Channel=1."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
    )
    pipeline.load().clean().normalize()

    # Standard 5D seq2seq format
    X_conv, Y_conv = pipeline.format_for_model(
        model_type="convlstm", window_size=7, horizon=1, squeeze_target_time=False
    )
    assert X_conv.shape == (13, 7, 1, 16, 16)
    assert Y_conv.shape == (13, 1, 1, 16, 16)

    # Squeezed target format
    X_conv2, Y_conv2 = pipeline.format_for_model(
        model_type="convlstm", window_size=7, horizon=1, squeeze_target_time=True
    )
    assert X_conv2.shape == (13, 7, 1, 16, 16)
    assert Y_conv2.shape == (13, 1, 16, 16)


def test_unet_format_and_assertions(sample_dataset_fixtures):
    """U-Net output formatting must stack time steps into Channels."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
    )
    pipeline.load().clean().normalize()

    X_unet, Y_unet = pipeline.format_for_model(
        model_type="unet", window_size=7, horizon=1
    )
    # In U-Net, Channels = window_size = 7
    assert X_unet.shape == (13, 7, 16, 16)
    # Target has 1 output channel
    assert Y_unet.shape == (13, 1, 16, 16)


def test_invalid_model_type_raises(sample_dataset_fixtures):
    """Pipeline must reject unsupported model types."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
    )
    pipeline.load().clean().normalize()

    with pytest.raises(ValueError, match="Unsupported model_type: 'resnet'"):
        pipeline.format_for_model(model_type="resnet")


def test_run_pipeline_end_to_end(sample_dataset_fixtures):
    """run_pipeline() must execute all stages fluently in one call."""
    pipeline = SeaIceDataPipeline(
        data_path=sample_dataset_fixtures["data_path"],
        mask_path=sample_dataset_fixtures["mask_nc_path"],
    )
    X, Y = pipeline.run_pipeline(
        model_type="convlstm", max_value=100.0, window_size=5, horizon=1
    )
    assert X.shape == (15, 5, 1, 16, 16)
    assert Y.shape == (15, 1, 1, 16, 16)
    assert not np.isnan(X).any()
    assert not np.isnan(Y).any()
