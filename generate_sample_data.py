"""Synthetic NetCDF dataset generator for Sea-Ice Data Pipeline testing and verification.

Generates:
1. data/sea_ice/siconc_2024_part1.nc (timesteps 0 to 29)
2. data/sea_ice/siconc_2024_part2.nc (timesteps 30 to 59)
3. data/land_mask.nc (spatial grid: land = 0, ocean = 1)
4. data/land_mask.npy (NumPy array format of the same mask)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def generate_synthetic_data(
    output_dir: str | Path = "data",
    height: int = 64,
    width: int = 64,
    total_days: int = 60,
    nan_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate synthetic sea ice NetCDF files and matching land mask.

    Args:
        output_dir: Root directory for output files.
        height: Spatial grid height (latitude dimension).
        width: Spatial grid width (longitude dimension).
        total_days: Total consecutive daily timesteps to generate.
        nan_ratio: Fraction of ocean pixels randomly set to NaN (missing satellite data).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sea_ice_dir, land_mask_path).
    """
    rng = np.random.default_rng(seed)
    base_dir = Path(output_dir)
    sea_ice_dir = base_dir / "sea_ice"
    sea_ice_dir.mkdir(parents=True, exist_ok=True)

    # 1. Coordinates
    lats = np.linspace(60.0, 90.0, height, dtype=np.float32)
    lons = np.linspace(-180.0, 180.0, width, dtype=np.float32)
    dates = pd.date_range("2024-01-01", periods=total_days, freq="D")

    # 2. Static Land Mask (land = 0, ocean = 1)
    # Create land features: e.g. Greenland/coastal corners
    land_mask = np.ones((height, width), dtype=np.float32)
    # Circular continent / island in the lower-left corner
    yy, xx = np.ogrid[:height, :width]
    island1 = ((yy - 12) ** 2 + (xx - 14) ** 2) < (10**2)
    island2 = ((yy - 8) ** 2 + (xx - 48) ** 2) < (8**2)
    coast = yy < 5  # southern lower latitude strip as land
    land_mask[island1 | island2 | coast] = 0.0

    # Save land_mask.nc
    mask_ds = xr.Dataset(
        data_vars={"mask": (["lat", "lon"], land_mask)},
        coords={"lat": lats, "lon": lons},
        attrs={"description": "Static land mask: 0 = land, 1 = ocean"},
    )
    land_mask_nc_path = base_dir / "land_mask.nc"
    mask_ds.to_netcdf(land_mask_nc_path)

    # Also save land_mask.npy
    land_mask_npy_path = base_dir / "land_mask.npy"
    np.save(land_mask_npy_path, land_mask)

    # 3. Dynamic Sea-Ice Concentration (Time, Height, Width)
    # Concentrations in percentage: 0 to 100%
    # Gradients with latitude + seasonal growth/melt oscillation + random fluctuations
    time_indices = np.arange(total_days)[:, None, None]
    lat_factor = (lats[None, :, None] - 60.0) / 30.0  # 0 at 60N, 1 at 90N
    seasonal = np.sin(2 * np.pi * time_indices / 365.0) * 15.0

    raw_ice = lat_factor * 85.0 + seasonal + rng.normal(0, 5, (total_days, height, width))
    raw_ice = np.clip(raw_ice, 0.0, 100.0).astype(np.float32)

    # Land pixels can have arbitrary non-zero values in raw satellite data
    raw_ice[:, land_mask == 0.0] = rng.uniform(10.0, 90.0, size=(total_days, int((land_mask == 0.0).sum())))

    # Introduce random NaNs (missing passes / cloud obscuration) over ocean pixels
    ocean_mask_3d = np.broadcast_to(land_mask == 1.0, (total_days, height, width))
    nan_selector = (rng.random((total_days, height, width)) < nan_ratio) & ocean_mask_3d
    raw_ice[nan_selector] = np.nan

    # 4. Split into two NetCDF files to demonstrate multi-file / glob support
    half = total_days // 2

    ds1 = xr.Dataset(
        data_vars={"siconc": (["time", "lat", "lon"], raw_ice[:half])},
        coords={"time": dates[:half], "lat": lats, "lon": lons},
        attrs={"title": "Synthetic Sea-Ice Concentration 2024 (Part 1)", "units": "%"},
    )
    part1_path = sea_ice_dir / "siconc_2024_part1.nc"
    ds1.to_netcdf(part1_path)

    ds2 = xr.Dataset(
        data_vars={"siconc": (["time", "lat", "lon"], raw_ice[half:])},
        coords={"time": dates[half:], "lat": lats, "lon": lons},
        attrs={"title": "Synthetic Sea-Ice Concentration 2024 (Part 2)", "units": "%"},
    )
    part2_path = sea_ice_dir / "siconc_2024_part2.nc"
    ds2.to_netcdf(part2_path)

    print(f"Generated synthetic dataset in '{base_dir.resolve()}':")
    print(f"  - Land mask: {land_mask_nc_path} (shape {land_mask.shape})")
    print(f"  - Land mask npy: {land_mask_npy_path}")
    print(f"  - Sea ice part 1: {part1_path} (shape {raw_ice[:half].shape})")
    print(f"  - Sea ice part 2: {part2_path} (shape {raw_ice[half:].shape})")
    print(f"  - Total days: {total_days}, NaN count: {np.isnan(raw_ice).sum()}")

    return sea_ice_dir, land_mask_nc_path


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    generate_synthetic_data(output_dir=current_dir / "data")
