"""
environment_service.py - Live Environmental Data Service
=========================================================

Fetches real-time wind and ocean current data for the Antarctic routing grid.

Data Sources (in priority order):
  1. Open-Meteo Marine API (instant, free, no key required)
  2. Fallback: Synthetic Southern Ocean climatology model

Physics Constants ported from:
  src/iceberg_model/physics/constants.py (Bigg et al. 1997)

Ekman Transport equations from:
  src/iceberg_model/physics/coriolis.py
  src/iceberg_model/physics/sea_ice_drag.py
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import httpx
import numpy as np

log = logging.getLogger("kryptonite.environment")

# ── Physical Constants (from scratch project physics/constants.py) ────────
EARTH_ROTATION_RATE = 7.2921159e-5    # rad/s
RHO_SEAWATER        = 1027.0          # kg/m³
RHO_AIR             = 1.225           # kg/m³
C_DA                = 1.3             # Air drag coefficient (Bigg et al. 1997)
C_DW                = 1.5             # Ocean drag coefficient
C_DSI               = 1.0             # Sea-ice drag coefficient
GRAVITY             = 9.80665         # m/s²
KNOTS_PER_MS        = 1.94384         # knots per m/s


@dataclass
class EnvironmentalFields:
    """Environmental vector fields over the 128x128 Antarctic grid."""
    wind_u: np.ndarray       # Zonal wind (m/s)
    wind_v: np.ndarray       # Meridional wind (m/s)
    ocean_u: np.ndarray      # Zonal ocean current (m/s)
    ocean_v: np.ndarray      # Meridional ocean current (m/s)
    drift_u: np.ndarray      # Ekman ice drift U (m/s)
    drift_v: np.ndarray      # Ekman ice drift V (m/s)
    source: str              # "open_meteo" | "synthetic" | "era5"
    grid_h: int = 128
    grid_w: int = 128


def coriolis_parameter(latitude_deg: float) -> float:
    """f = 2 * Ω * sin(φ). Negative in Southern Hemisphere (deflects LEFT)."""
    return 2.0 * EARTH_ROTATION_RATE * math.sin(math.radians(latitude_deg))


def compute_ekman_drift(wind_u: np.ndarray, wind_v: np.ndarray,
                         sic_grid: np.ndarray, lat_center: float = -66.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Ekman Transport: ice drifts at ~2% of wind speed, deflected 25° LEFT
    in the Southern Hemisphere due to the Coriolis effect.
    
    Physics from: scratch project physics/coriolis.py + sea_ice_drag.py
    """
    # Deflection angle: -25° (leftward in SH)
    theta = math.radians(-25.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    
    # Ekman surface current: 2% of wind speed, rotated
    drift_u = 0.02 * (wind_u * cos_t - wind_v * sin_t)
    drift_v = 0.02 * (wind_u * sin_t + wind_v * cos_t)
    
    # Scale by sea-ice concentration — only pack ice transmits wind stress
    sic_mask = np.clip(sic_grid, 0.0, 1.0)
    drift_u *= sic_mask
    drift_v *= sic_mask
    
    return drift_u, drift_v


def wind_drag_force_knots(wind_u: float, wind_v: float,
                           ship_u: float, ship_v: float) -> float:
    """
    Compute wind drag effect on ship speed (simplified from forces.py).
    F_air = 0.5 * ρ_air * C_Da * A_sail * |V_rel|² * cos(θ)
    
    Returns speed delta in knots (positive = tailwind boost, negative = headwind penalty).
    """
    # Relative wind vector
    rel_u = wind_u - ship_u
    rel_v = wind_v - ship_v
    rel_speed = math.hypot(rel_u, rel_v)
    
    if rel_speed < 0.1:
        return 0.0
    
    # Dot product of relative wind with ship heading (normalized)
    ship_speed = math.hypot(ship_u, ship_v)
    if ship_speed < 0.01:
        return 0.0
    
    cos_angle = (rel_u * ship_u + rel_v * ship_v) / (rel_speed * ship_speed)
    
    # Simplified sail force → speed effect
    # Using typical sail area ~500m² for Antarctic supply vessel
    A_sail = 500.0
    F_magnitude = 0.5 * RHO_AIR * C_DA * A_sail * rel_speed**2
    
    # Convert force to speed delta (F = m*a, typical vessel mass ~10000 tonnes)
    vessel_mass = 10_000_000.0  # kg
    accel = F_magnitude / vessel_mass  # m/s²
    
    # Effect over ~1 hour transit per cell
    speed_delta_ms = accel * 3600.0 * cos_angle * 0.01  # dampened
    return speed_delta_ms * KNOTS_PER_MS


def ocean_drag_effect(ocean_u: float, ocean_v: float,
                       ship_u: float, ship_v: float) -> float:
    """
    Hydrodynamic drag/assist from ocean currents (from ocean_drag.py).
    Returns speed delta in knots.
    """
    # Dot product of current with ship heading
    ship_speed = math.hypot(ship_u, ship_v)
    if ship_speed < 0.01:
        return 0.0
    
    # Current component along ship heading
    current_along = (ocean_u * ship_u + ocean_v * ship_v) / ship_speed
    
    # Current assist/resist (direct velocity addition, dampened by drag coefficient)
    return current_along * KNOTS_PER_MS * 0.8  # 80% coupling efficiency


async def fetch_open_meteo_wind(lat_min: float, lat_max: float,
                                  lon_min: float, lon_max: float,
                                  grid_h: int = 128, grid_w: int = 128) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Fetch live wind data from Open-Meteo Marine API.
    Returns (wind_u, wind_v) arrays interpolated to grid, or None on failure.
    """
    # Sample 9 points across the grid
    sample_lats = np.linspace(lat_min, lat_max, 3)
    sample_lons = np.linspace(lon_min, lon_max, 3)
    
    lat_str = ",".join(f"{lat:.2f}" for lat in sample_lats for _ in sample_lons)
    lon_str = ",".join(f"{lon:.2f}" for _ in sample_lats for lon in sample_lons)
    
    url = (
        f"https://marine-api.open-meteo.com/v1/marine?"
        f"latitude={lat_str}&longitude={lon_str}"
        f"&current=wind_wave_direction,wind_wave_height"
        f"&hourly=wind_wave_height,wind_wave_direction"
        f"&forecast_days=1"
    )
    
    # Also get wind from regular weather API
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat_str}&longitude={lon_str}"
        f"&current=wind_speed_10m,wind_direction_10m"
        f"&forecast_days=1"
    )
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(weather_url)
            resp.raise_for_status()
            data = resp.json()
        
        # Parse multi-location response
        wind_speeds = []
        wind_dirs = []
        
        if isinstance(data, list):
            for loc in data:
                current = loc.get("current", {})
                ws = current.get("wind_speed_10m", 0.0)
                wd = current.get("wind_direction_10m", 270.0)
                wind_speeds.append(ws / 3.6)  # km/h → m/s
                wind_dirs.append(wd)
        else:
            current = data.get("current", {})
            ws = current.get("wind_speed_10m", 0.0)
            wd = current.get("wind_direction_10m", 270.0)
            wind_speeds = [ws / 3.6] * 9
            wind_dirs = [wd] * 9
        
        # Interpolate 3x3 sample points to full grid
        wind_speeds_grid = np.array(wind_speeds).reshape(3, 3)
        wind_dirs_grid = np.array(wind_dirs).reshape(3, 3)
        
        from scipy.ndimage import zoom
        factor_h = grid_h / 3
        factor_w = grid_w / 3
        
        ws_full = zoom(wind_speeds_grid, (factor_h, factor_w), order=1)
        wd_full = zoom(wind_dirs_grid, (factor_h, factor_w), order=1)
        
        # Convert speed + direction to U, V components
        wd_rad = np.radians(wd_full)
        wind_u = -ws_full * np.sin(wd_rad)  # Meteorological convention
        wind_v = -ws_full * np.cos(wd_rad)
        
        # Convert to knots for the pathfinder
        wind_u *= KNOTS_PER_MS
        wind_v *= KNOTS_PER_MS
        
        log.info("Open-Meteo: fetched live wind data (mean speed: %.1f kts)", 
                 np.mean(np.hypot(wind_u, wind_v)))
        return wind_u[:grid_h, :grid_w], wind_v[:grid_h, :grid_w]
        
    except Exception as e:
        log.warning("Open-Meteo fetch failed: %s — falling back to synthetic", e)
        return None


def generate_synthetic_fields(grid_h: int = 128, grid_w: int = 128,
                                sic_grid: np.ndarray | None = None) -> EnvironmentalFields:
    """
    Generate synthetic but physically rigorous Southern Ocean environment.
    Based on climatological patterns: persistent westerlies ("Roaring Forties"),
    Antarctic Circumpolar Current, and Ekman-derived ice drift.
    """
    # Wind: Dominant westerlies with cyclonic perturbations
    wind_u = np.full((grid_h, grid_w), 25.0)  # ~25 kt base westerly
    wind_v = np.full((grid_h, grid_w), -5.0)  # Slight northerly component
    
    for r in range(grid_h):
        for c in range(grid_w):
            # Latitude-dependent intensification (stronger near -65°S)
            lat_factor = 1.0 + 0.3 * math.sin(math.pi * r / grid_h)
            wind_u[r, c] *= lat_factor
            wind_u[r, c] += math.sin(c / 10.0) * 8.0
            wind_v[r, c] += math.cos(r / 10.0) * 6.0
    
    # Ocean currents: Antarctic Circumpolar Current (~0.1-0.5 m/s eastward)
    ocean_u = np.full((grid_h, grid_w), 0.3) * KNOTS_PER_MS  # ~0.3 m/s eastward → knots
    ocean_v = np.full((grid_h, grid_w), -0.05) * KNOTS_PER_MS
    
    for r in range(grid_h):
        for c in range(grid_w):
            ocean_u[r, c] += math.sin(r / 20.0) * 0.15 * KNOTS_PER_MS
            ocean_v[r, c] += math.cos(c / 15.0) * 0.1 * KNOTS_PER_MS
    
    # Ekman drift (only where ice exists)
    if sic_grid is not None:
        drift_u, drift_v = compute_ekman_drift(wind_u, wind_v, sic_grid)
    else:
        theta = math.radians(-25.0)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        drift_u = 0.02 * (wind_u * cos_t - wind_v * sin_t)
        drift_v = 0.02 * (wind_u * sin_t + wind_v * cos_t)
    
    return EnvironmentalFields(
        wind_u=wind_u, wind_v=wind_v,
        ocean_u=ocean_u, ocean_v=ocean_v,
        drift_u=drift_u, drift_v=drift_v,
        source="synthetic_climatology",
        grid_h=grid_h, grid_w=grid_w,
    )


async def fetch_environment(grid_h: int = 128, grid_w: int = 128,
                              lat_min: float = -72.0, lat_max: float = -60.0,
                              lon_min: float = 10.0, lon_max: float = 80.0,
                              sic_grid: np.ndarray | None = None) -> EnvironmentalFields:
    """
    Primary entry point: try live Open-Meteo, fall back to synthetic.
    """
    live_wind = await fetch_open_meteo_wind(lat_min, lat_max, lon_min, lon_max, grid_h, grid_w)
    
    if live_wind is not None:
        wind_u, wind_v = live_wind
        
        # Ocean currents: synthetic ACC (Open-Meteo doesn't provide currents)
        ocean_u = np.full((grid_h, grid_w), 0.3) * KNOTS_PER_MS
        ocean_v = np.full((grid_h, grid_w), -0.05) * KNOTS_PER_MS
        for r in range(grid_h):
            for c in range(grid_w):
                ocean_u[r, c] += math.sin(r / 20.0) * 0.15 * KNOTS_PER_MS
                ocean_v[r, c] += math.cos(c / 15.0) * 0.1 * KNOTS_PER_MS
        
        # Compute Ekman drift from live wind
        if sic_grid is not None:
            drift_u, drift_v = compute_ekman_drift(wind_u, wind_v, sic_grid)
        else:
            theta = math.radians(-25.0)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            drift_u = 0.02 * (wind_u * cos_t - wind_v * sin_t)
            drift_v = 0.02 * (wind_u * sin_t + wind_v * cos_t)
        
        return EnvironmentalFields(
            wind_u=wind_u, wind_v=wind_v,
            ocean_u=ocean_u, ocean_v=ocean_v,
            drift_u=drift_u, drift_v=drift_v,
            source="open_meteo_live",
            grid_h=grid_h, grid_w=grid_w,
        )
    
    return generate_synthetic_fields(grid_h, grid_w, sic_grid)
