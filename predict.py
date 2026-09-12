"""Sea-Ice Batch Inference & Forecast Export Script.

Runs the trained SeaIceUNet model on the most recent observation window,
produces 3 lead-day forecast PNG images (colorized ice concentration maps),
and writes a forecast.json manifest consumed by the React dashboard.

Output files (written to --output-dir, default: frontend/public/):
    forecast.json        – data contract manifest
    forecast_day1.png    – lead-day 1 prediction
    forecast_day2.png    – lead-day 2 prediction
    forecast_day3.png    – lead-day 3 prediction

Data Contract (forecast.json):
{
  "forecast_date": "YYYY-MM-DD",         # date of the last observation used
  "generated_at": "ISO-8601",            # wall-clock time this script ran
  "grid": {
    "bounds": [minLon, minLat, maxLon, maxLat],  # WGS-84 degrees
    "width": <int>,
    "height": <int>
  },
  "predictions": [
    { "lead_day": 1, "date": "YYYY-MM-DD", "image_url": "forecast_day1.png" },
    ...
  ],
  "extent_history": [
    { "date": "YYYY-MM-DD", "extent_km2": <float> }   # last 30 days of obs
  ]
}

Coordinate system note
----------------------
The synthetic data uses WGS-84 lat/lon directly, so no reprojection is
required for the demo case.  For real NSIDC polar-stereographic data
(EPSG:4326 grid corners stored in native EPSG:3413), enable the pyproj
reprojection block tagged "REAL DATA CRS" below.

MapLibre's ImageSource.coordinates expects WGS-84 [lon, lat] corners:
    [[TL], [TR], [BR], [BL]]
These are derived from grid.bounds by mapHelpers.js on the frontend.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# ── Optional heavy imports (gracefully degrade for --dry-run without PyTorch) ──

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    import xarray as xr
    _XR_AVAILABLE = True
except ImportError:
    _XR_AVAILABLE = False


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="[%(asctime)s] [predict] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("predict")


# ── Colormap helpers ───────────────────────────────────────────────────────────

def _build_ice_colormap():
    """Return a matplotlib colormap suitable for sea-ice concentration.

    Gradient: transparent (0%) → deep-blue → cyan → white (100%).
    Falls back to 'Blues' if matplotlib is unavailable.
    """
    if not _MPL_AVAILABLE:
        return None
    colors_stops = [
        (0.00, (0.04, 0.10, 0.20, 0.0)),   # fully transparent at 0%
        (0.10, (0.05, 0.20, 0.45, 0.6)),
        (0.30, (0.08, 0.35, 0.65, 0.8)),
        (0.55, (0.15, 0.55, 0.85, 0.9)),
        (0.75, (0.40, 0.75, 0.95, 0.95)),
        (1.00, (0.92, 0.97, 1.00, 1.0)),   # near-white at 100%
    ]
    positions = [s[0] for s in colors_stops]
    rgba = [s[1] for s in colors_stops]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sea_ice", list(zip(positions, rgba)), N=256
    )
    return cmap


def _array_to_png(
    concentration_01: np.ndarray,
    land_mask: np.ndarray | None,
    out_path: Path,
    cmap=None,
) -> None:
    """Convert a [0,1] concentration array to a colourised RGBA PNG.

    Args:
        concentration_01: 2-D array, values in [0.0, 1.0].
        land_mask: 2-D binary array (ocean=1, land=0). Land pixels become
                   fully transparent in the PNG so the basemap shows through.
        out_path: Destination file path.
        cmap: matplotlib colormap or None (falls back to PIL gradient).
    """
    h, w = concentration_01.shape

    if _MPL_AVAILABLE and cmap is not None:
        # Use matplotlib colormap → RGBA (float32, 0-1)
        rgba_float = cmap(concentration_01)          # (H, W, 4)
        rgba_uint8 = (rgba_float * 255).astype(np.uint8)
    else:
        # Minimal fallback: blue gradient without matplotlib
        r = (concentration_01 * 150).astype(np.uint8)
        g = (concentration_01 * 200 + 55).astype(np.uint8)
        b = np.full((h, w), 220, dtype=np.uint8)
        a = (concentration_01 * 230 + 25).astype(np.uint8)
        rgba_uint8 = np.stack([r, g, b, a], axis=-1)

    # Zero out alpha on land so basemap shows through
    if land_mask is not None:
        land_pixels = land_mask == 0.0
        rgba_uint8[land_pixels, 3] = 0

    if _PIL_AVAILABLE:
        img = Image.fromarray(rgba_uint8, mode="RGBA")
        img.save(out_path)
    else:
        # Absolute last-resort: raw RGBA bytes → BMP-like dump is complex,
        # so we write a 1×1 placeholder and warn.
        logger.warning(
            "Pillow not installed. Writing placeholder PNG. "
            "Install Pillow: pip install Pillow"
        )
        # Write a minimal 1×1 transparent PNG (89 bytes, valid PNG header)
        _write_minimal_png(out_path)

    logger.info(f"Written PNG → {out_path} ({w}×{h})")


def _write_minimal_png(path: Path) -> None:
    """Write the smallest possible valid transparent RGBA PNG (1×1 px)."""
    import zlib, struct

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
    raw = b"\x00\x00\x00\x00\x00"          # filter byte + 1 RGBA pixel (transparent)
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    path.write_bytes(header + ihdr + idat + iend)


# ── Grid bounds helpers ────────────────────────────────────────────────────────

def _derive_bounds_wgs84(
    lats: np.ndarray,
    lons: np.ndarray,
) -> list[float]:
    """Return [minLon, minLat, maxLon, maxLat] in WGS-84 degrees.

    For synthetic data that already carries WGS-84 lat/lon coordinates, this
    is simply the array min/max.

    REAL DATA CRS block (polar stereographic → WGS-84):
    If your NetCDF uses native polar stereographic x/y coordinates, uncomment
    the pyproj block below and pass the native CRS string.

    Example (NSIDC EASE-Grid / EPSG:3413):
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:3413", "EPSG:4326", always_xy=True)
        lons_wgs84, lats_wgs84 = transformer.transform(lons, lats)
        return [float(lons_wgs84.min()), float(lats_wgs84.min()),
                float(lons_wgs84.max()), float(lats_wgs84.max())]
    """
    return [
        float(lons.min()),
        float(lats.min()),
        float(lons.max()),
        float(lats.max()),
    ]


# ── Extent computation ─────────────────────────────────────────────────────────

def _compute_extent_km2(
    concentration_01: np.ndarray,
    land_mask: np.ndarray,
    lats: np.ndarray,
    threshold: float = 0.15,
) -> float:
    """Estimate total sea-ice extent in km².

    Standard definition: ocean pixels where concentration ≥ 15%.
    Cell area is approximated using the latitude of each row (equal-angle grid).

    Args:
        concentration_01: (H, W) array in [0, 1].
        land_mask: (H, W) binary mask (ocean=1, land=0).
        lats: 1-D latitude array of length H (degrees North).
        threshold: ice-covered threshold (default 0.15 = 15%).

    Returns:
        Estimated extent in km².
    """
    h, w = concentration_01.shape

    # Cell width in degrees (assume equal spacing)
    if w > 1:
        dlon = abs(360.0 / w)   # approximate for global longitude span
    else:
        dlon = 1.0

    if h > 1:
        dlat = abs((lats[-1] - lats[0]) / (h - 1))
    else:
        dlat = 1.0

    # Area per cell at each latitude (km²)
    R_km = 6371.0
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)
    lat_rad = np.deg2rad(lats)                    # (H,)
    cell_area_km2 = (R_km ** 2) * dlat_rad * dlon_rad * np.cos(lat_rad)  # (H,)

    # Broadcast to (H, W)
    cell_area_2d = np.tile(cell_area_km2[:, np.newaxis], (1, w))

    ice_pixels = (concentration_01 >= threshold) & (land_mask == 1.0)
    return float(np.sum(cell_area_2d[ice_pixels]))


# ── Synthetic fallback (no checkpoint available) ───────────────────────────────

def _generate_synthetic_forecast(
    lats: np.ndarray,
    lons: np.ndarray,
    land_mask: np.ndarray,
    window_data_01: np.ndarray,
    lead_days: int = 3,
) -> list[np.ndarray]:
    """Produce plausible synthetic forecasts by extrapolating recent trend.

    Used when no trained checkpoint is available (demo / CI mode).
    Each lead-day slightly melts the most recent frame by a random amount,
    preserving the land mask.

    Returns:
        List of ``lead_days`` arrays shaped (H, W) in [0.0, 1.0].
    """
    rng = np.random.default_rng(seed=42)
    last_frame = window_data_01[-1].copy()          # most recent observation
    preds = []
    current = last_frame
    for d in range(lead_days):
        noise = rng.normal(0, 0.01, current.shape).astype(np.float32)
        melt = rng.uniform(0.005, 0.015)            # gentle melt trend
        nxt = np.clip(current - melt + noise, 0.0, 1.0)
        nxt[land_mask == 0.0] = 0.0
        preds.append(nxt)
        current = nxt
    return preds


# ── Main export logic ──────────────────────────────────────────────────────────

def run_predict(
    data_dir: Path,
    checkpoint_path: Path | None,
    output_dir: Path,
    window_size: int = 7,
    lead_days: int = 3,
    history_days: int = 30,
    dry_run: bool = False,
) -> dict:
    """Load data, run inference, write PNGs + forecast.json.

    Args:
        data_dir: Directory containing sea_ice/*.nc files and land_mask.nc.
        checkpoint_path: Path to model .pt checkpoint, or None for synthetic.
        output_dir: Directory where forecast.json and PNGs will be written.
        window_size: Number of observation days fed to the model (default 7).
        lead_days: Number of forecast days to produce (default 3).
        history_days: Days of history to include in extent_history (default 30).
        dry_run: If True, skip writing files and just return the manifest dict.

    Returns:
        The forecast manifest dictionary (same as forecast.json content).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info("Loading sea-ice data via SeaIceDataPipeline...")

    if not _XR_AVAILABLE:
        raise ImportError("xarray is required. Run: pip install xarray netCDF4")

    # Import pipeline from repo root
    repo_root = Path(__file__).parent
    sys.path.insert(0, str(repo_root))
    from data_pipeline import SeaIceDataPipeline

    sea_ice_pattern = str(data_dir / "sea_ice" / "*.nc")
    mask_file = str(data_dir / "land_mask.nc")

    pipeline = SeaIceDataPipeline(
        data_path=sea_ice_pattern,
        mask_path=mask_file,
        variable_name="siconc",
    )
    pipeline.load().clean().normalize(max_value=100.0)

    norm_data = pipeline.normalized_data       # (T, H, W) in [0, 1]
    land_mask = pipeline.land_mask             # (H, W)
    T, H, W = norm_data.shape

    # Open each NetCDF individually (no dask required) then concatenate
    import xarray as xr
    nc_files = sorted((data_dir / "sea_ice").glob("*.nc"))
    datasets = [xr.open_dataset(f) for f in nc_files]
    ds = xr.concat(datasets, dim="time") if len(datasets) > 1 else datasets[0]
    lats = ds["lat"].values.astype(np.float32)   # (H,)
    lons = ds["lon"].values.astype(np.float32)   # (W,)
    time_coords = ds["time"].values               # numpy datetime64 array

    # ── 2. Derive grid bounds ─────────────────────────────────────────────────
    bounds_wgs84 = _derive_bounds_wgs84(lats, lons)
    logger.info(f"Grid bounds (WGS-84): {bounds_wgs84}")

    # ── 3. Prepare observation window for inference ───────────────────────────
    if T < window_size:
        raise ValueError(
            f"Not enough timesteps ({T}) for window_size={window_size}. "
            "Generate more synthetic data or use a smaller window."
        )

    window_01 = norm_data[-window_size:]       # (window_size, H, W)

    # ── 4. Run model inference ────────────────────────────────────────────────
    if checkpoint_path is not None and checkpoint_path.exists() and _TORCH_AVAILABLE:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        from unet_model import SeaIceUNet

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Support both raw state-dict and trainer checkpoint dicts
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        land_mask_tensor = torch.from_numpy(land_mask)
        model = SeaIceUNet(
            in_channels=window_size,
            out_channels=1,
            base_filters=32,
            depth=4,
            land_mask=land_mask_tensor,
        )
        model.load_state_dict(state_dict)
        model.eval()

        predictions_01: list[np.ndarray] = []
        current_window = window_01.copy()

        with torch.no_grad():
            for d in range(lead_days):
                x_tensor = torch.from_numpy(current_window[np.newaxis])  # (1, W, H, W)
                pred = model(x_tensor).squeeze().numpy()                  # (H, W)
                pred = np.clip(pred, 0.0, 1.0)
                predictions_01.append(pred)
                # Roll window forward: drop oldest, append prediction
                current_window = np.concatenate(
                    [current_window[1:], pred[np.newaxis]], axis=0
                )

        logger.info("Model inference complete.")
    else:
        if checkpoint_path is not None:
            logger.warning(
                f"Checkpoint not found at {checkpoint_path} — "
                "falling back to synthetic forecasts."
            )
        else:
            logger.info("No checkpoint specified — using synthetic forecast mode.")

        predictions_01 = _generate_synthetic_forecast(
            lats, lons, land_mask, window_01, lead_days=lead_days
        )

    # ── 5. Render PNG images ──────────────────────────────────────────────────
    cmap = _build_ice_colormap()
    pred_entries = []
    last_obs_date = str(time_coords[-1])[:10]           # "YYYY-MM-DD"
    last_obs_dt = datetime.strptime(last_obs_date, "%Y-%m-%d")

    for d, pred_01 in enumerate(predictions_01, start=1):
        png_name = f"forecast_day{d}.png"
        png_path = output_dir / png_name
        forecast_date_str = (last_obs_dt + timedelta(days=d)).strftime("%Y-%m-%d")

        if not dry_run:
            _array_to_png(pred_01, land_mask, png_path, cmap=cmap)

        pred_entries.append({
            "lead_day": d,
            "date": forecast_date_str,
            "image_url": png_name,
        })

    # ── 6. Build extent_history ───────────────────────────────────────────────
    extent_history = []
    history_slice = norm_data[-history_days:]
    history_times = time_coords[-history_days:]

    for i, ts in enumerate(history_times):
        date_str = str(ts)[:10]
        km2 = _compute_extent_km2(history_slice[i], land_mask, lats)
        extent_history.append({"date": date_str, "extent_km2": round(km2, 1)})

    # ── 7. Assemble manifest ──────────────────────────────────────────────────
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "forecast_date": last_obs_date,
        "generated_at": now_iso,
        "grid": {
            "bounds": bounds_wgs84,
            "width": W,
            "height": H,
        },
        "predictions": pred_entries,
        "extent_history": extent_history,
    }

    if not dry_run:
        json_path = output_dir / "forecast.json"
        json_path.write_text(json.dumps(manifest, indent=2))
        logger.info(f"Written manifest → {json_path}")

    logger.info("predict.py complete.")
    return manifest


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sea-Ice Batch Inference: produce forecast.json + PNGs."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data",
        help="Directory containing sea_ice/*.nc and land_mask.nc (default: ./data)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model .pt checkpoint. If omitted, synthetic forecasts are used.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "frontend" / "public",
        help="Output directory for forecast.json and PNGs (default: frontend/public/)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Number of past observation days fed to the model (default: 7).",
    )
    p.add_argument(
        "--lead-days",
        type=int,
        default=3,
        help="Number of forecast lead days to produce (default: 3).",
    )
    p.add_argument(
        "--history-days",
        type=int,
        default=30,
        help="Days of observed history to include in extent_history (default: 30).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate pipeline without writing any output files.",
    )
    # Auto-detect best available checkpoint when no explicit path given
    p.add_argument(
        "--auto-checkpoint",
        action="store_true",
        help="Search checkpoints_es/ then checkpoints_test/ for best_model.pt.",
    )
    return p.parse_args()


def _find_checkpoint(repo_root: Path) -> Path | None:
    """Search standard checkpoint directories for a saved model file."""
    candidates = [
        repo_root / "model_final.pt",
        repo_root / "model_final.onnx",
        repo_root / "checkpoints" / "model_final.pt",
        repo_root / "checkpoints_es" / "best_model.pt",
        repo_root / "checkpoints_test" / "best_model.pt",
        repo_root / "checkpoints_es" / "last_model.pt",
        repo_root / "checkpoints_test" / "last_model.pt",
    ]
    for c in candidates:
        if c.exists():
            logger.info(f"Auto-detected checkpoint: {c}")
            return c
    logger.info("No checkpoint found — synthetic forecast mode will be used.")
    return None


if __name__ == "__main__":
    args = _parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None and args.auto_checkpoint:
        checkpoint = _find_checkpoint(Path(__file__).parent)

    manifest = run_predict(
        data_dir=args.data_dir,
        checkpoint_path=checkpoint,
        output_dir=args.output_dir,
        window_size=args.window_size,
        lead_days=args.lead_days,
        history_days=args.history_days,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\n── Dry-run manifest (not written to disk) ──")
        print(json.dumps(manifest, indent=2))
