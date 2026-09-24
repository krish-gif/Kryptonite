"""
main.py - Kryptonite Navigation Engine (v2.2)
==============================================

FastAPI backend serving the PyTorch ConvLSTM sea-ice forecasts and
geospatial A* routing engine over the Antarctic grid.

Features:
 - Loads actual PyTorch ConvLSTM model (`trained/model_final.pt`).
 - Dual-Stage Routing (Open Ocean + Ice Transit).
 - Hourly Spatiotemporal Telemetry.
 - GeoJSON Ice Grid Endpoint (`/api/ice-grid`).
"""

import sys
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

# Load .env BEFORE any other imports that might need env vars
from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import pathfinder
import environment_service

# ── Configuration ────────────────────────────────────────────────────────
MODEL_PATH = Path("trained/model_final.pt")
DATA_DIR   = Path("data/South")

# Grid bounds mapping to the 128x128 array
GRID_H   = 128
GRID_W   = 128
LAT_MIN  = -72.0
LAT_MAX  = -60.0
LON_MIN  =  10.0
LON_MAX  =  80.0
ICE_THRESH = 0.15

# Global application state
_state = {
    "model": None,
    "last_obs": None,
    "last_forecast": None,
    "forecast_date": None,
    "env_fields": None,
    "env_source": "none",
}

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("kryptonite")

# ── PyTorch Model Loading & Inference ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the TorchScript model into global state on startup."""
    if not MODEL_PATH.exists():
        log.warning("No model found at %s. Inference will fail.", MODEL_PATH)
    else:
        log.info("Loading PyTorch model from %s...", MODEL_PATH)
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = torch.jit.load(str(MODEL_PATH), map_location=device)
            model.eval()
            _state["model"] = model
            log.info("Model loaded successfully on %s.", device)
        except Exception as e:
            log.error("Failed to load model: %s", e)
    
    # Fetch live environmental data and inject into pathfinder
    log.info("Fetching environmental data (wind, ocean currents, Ekman drift)...")
    try:
        env = await environment_service.fetch_environment(
            grid_h=GRID_H, grid_w=GRID_W,
            lat_min=LAT_MIN, lat_max=LAT_MAX,
            lon_min=LON_MIN, lon_max=LON_MAX,
        )
        pathfinder.set_environment(
            env.wind_u, env.wind_v,
            env.ocean_u, env.ocean_v,
            env.drift_u, env.drift_v,
            source=env.source,
        )
        _state["env_fields"] = env
        _state["env_source"] = env.source
        log.info("Environment loaded: source=%s, mean_wind=%.1f kts",
                 env.source, float(np.mean(np.hypot(env.wind_u, env.wind_v))))
    except Exception as e:
        log.warning("Failed to fetch live environment: %s — using synthetic fallback", e)
        env = environment_service.generate_synthetic_fields(GRID_H, GRID_W)
        pathfinder.set_environment(
            env.wind_u, env.wind_v,
            env.ocean_u, env.ocean_v,
            env.drift_u, env.drift_v,
            source=env.source,
        )
        _state["env_fields"] = env
        _state["env_source"] = env.source
    
    yield
    _state["model"] = None


def _synthetic_obs() -> np.ndarray:
    """Generate synthetic 5-day observation sequence if no real data."""
    # (T=5, H=128, W=128)
    obs = np.zeros((5, GRID_H, GRID_W), dtype=np.float32)
    for t in range(5):
        # A synthetic ice block growing from the south
        ice_limit = GRID_H - 20 - (t * 5)
        obs[t, ice_limit:, :] = 0.8
    return obs


def _run_inference(model: torch.jit.ScriptModule, obs_5d: np.ndarray) -> np.ndarray:
    """Run model, return (3, 128, 128) float32 SIC."""
    device = next(model.parameters()).device
    # Add batch and channel dims: (B=1, T=5, C=1, H, W)
    x = torch.from_numpy(obs_5d).unsqueeze(0).unsqueeze(2).to(device)
    with torch.no_grad():
        out = model(x)  # shape (1, 3, 1, H, W)
    
    # Extract the 3-day forecast sequence: (3, H, W)
    forecast = out.squeeze(0).squeeze(1).cpu().numpy()
    return np.clip(forecast, 0.0, 1.0)


def _get_or_build_forecast() -> tuple[np.ndarray, np.ndarray]:
    """Return (obs, forecast), running inference if not cached."""
    if _state["last_forecast"] is not None:
        return _state["last_obs"], _state["last_forecast"]

    if _state["model"] is None:
        raise RuntimeError("PyTorch model is not loaded.")

    obs = _synthetic_obs()
    forecast = _run_inference(_state["model"], obs)

    _state["last_obs"]       = obs
    _state["last_forecast"]  = forecast
    _state["forecast_date"]  = datetime.now(timezone.utc)

    log.info("Forecast computed — ice extent +72h: %.1f%%",
             float((forecast[2] > ICE_THRESH).mean() * 100))
    return obs, forecast


# ── FastAPI Application ──────────────────────────────────────────────────

app = FastAPI(
    title="Kryptonite Engine",
    version="2.2.0",
    description="Dual-Stage Ice Pathfinding & Inference API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ──────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    start_coords: tuple[float, float] = Field(
        ...,
        description="Origin [lat, lon]. Can be global (e.g. Cape Town).",
        example=[-33.9, 18.4]
    )
    end_coords: tuple[float, float] = Field(
        ...,
        description="Destination [lat, lon]. Must be Maitri or Bharati bounds.",
        example=[-70.76, 11.44]
    )
    forecast_window: int = Field(
        72,
        description="Which forecast step to route against (+24h, +48h, +72h)."
    )
    safety_weight: float = Field(
        1.0,
        description="Buffer dilation strength (0.0 to 10.0).",
        ge=0.0,
        le=10.0,
    )


# ── Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["System"])
async def health():
    """Liveness check and credentials status."""
    creds = {
        "copernicus_cds": bool(os.environ.get("CDSAPI_KEY")),
        "nasa_earthdata": bool(os.environ.get("EARTHDATA_TOKEN") or (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"))),
        "copernicus_marine": bool(os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") and os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD")),
        "carto": bool(os.environ.get("CARTO_API_KEY")),
    }
    return {
        "status":        "ok",
        "model_loaded":  _state["model"] is not None,
        "forecast_date": _state["forecast_date"].isoformat() if _state["forecast_date"] else None,
        "env_source":    _state["env_source"],
        "credentials":   creds,
        "grid": {
            "height": GRID_H, "width": GRID_W,
            "lat_range": [LAT_MIN, LAT_MAX],
            "lon_range": [LON_MIN, LON_MAX],
        },
    }

@app.get("/api/ice-grid", tags=["Forecast"])
async def get_ice_grid(horizon: int = 24):
    """
    Return the predicted ice concentration grid as a GeoJSON FeatureCollection
    of colored polygons for direct MapLibre rendering.
    """
    if _state["model"] is None:
        raise HTTPException(503, "Model not loaded.")
        
    try:
        _, forecast = _get_or_build_forecast()
    except Exception as exc:
        raise HTTPException(500, f"Forecast error: {exc}")
        
    step = {24: 0, 48: 1, 72: 2}.get(horizon, 0)
    if forecast.ndim == 3:
        step = min(step, forecast.shape[0] - 1)
        grid = forecast[step]
    else:
        grid = forecast
        
    features = []
    CELL_LAT = (LAT_MIN - LAT_MAX) / GRID_H
    CELL_LON = (LON_MAX - LON_MIN) / GRID_W
    
    for r in range(GRID_H):
        for c in range(GRID_W):
            sic = float(grid[r, c])
            if sic < 0.05: continue
            
            latN = LAT_MAX + r * CELL_LAT
            lonW = LON_MIN + c * CELL_LON
            latS = latN + CELL_LAT
            lonE = lonW + CELL_LON
            
            features.append({
                "type": "Feature",
                "properties": {"sic": sic, "horizon": horizon},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lonW, latN], [lonE, latN],
                        [lonE, latS], [lonW, latS],
                        [lonW, latN]
                    ]]
                }
            })
            
    return {"type": "FeatureCollection", "features": features}


@app.get("/api/forecast", tags=["Forecast"])
async def get_forecast():
    """Return raw forecast matrices and extent metrics."""
    if _state["model"] is None:
        raise HTTPException(503, "PyTorch model not loaded.")
    try:
        _, forecast = _get_or_build_forecast()
    except Exception as exc:
        raise HTTPException(500, str(exc))
    
    # Calculate sea ice extent (km2) for >15% threshold
    # PSS25 cell = 25km x 25km = 625 km2
    extent_km2 = {}
    for i, h in enumerate([24, 48, 72]):
        step_grid = forecast[min(i, forecast.shape[0] - 1)]
        ice_cells = int((step_grid > ICE_THRESH).sum())
        extent_km2[f"day{i+1}"] = ice_cells * 625
        
    return {
        "forecast_date": _state["forecast_date"].isoformat(),
        "extent_km2": extent_km2,
        "grid_meta": {
            "bounds": [[LAT_MAX, LON_MIN], [LAT_MIN, LON_MAX]],
            "width": GRID_W, "height": GRID_H
        }
    }


@app.post("/api/calculate-route", tags=["Routing"])
async def calculate_route(req: RouteRequest):
    """
    Run dual-stage routing + A* pathfinding.
    """
    if _state["model"] is None:
        raise HTTPException(503, "PyTorch model not loaded.")
        
    # Validate destination is in Antarctic bounds
    dest_lat, dest_lon = req.end_coords
    if not (LAT_MIN <= dest_lat <= LAT_MAX):
        raise HTTPException(422, f"Destination lat {dest_lat} outside Antarctic grid ({LAT_MIN} to {LAT_MAX})")
    if not (LON_MIN <= dest_lon <= LON_MAX):
        raise HTTPException(422, f"Destination lon {dest_lon} outside grid ({LON_MIN} to {LON_MAX})")
        
    start_lat, start_lon = req.start_coords
    if not (-90.0 <= start_lat <= 90.0) or not (-180.0 <= start_lon <= 180.0):
        raise HTTPException(422, "Invalid global coordinates.")

    try:
        _, forecast = _get_or_build_forecast()
    except Exception as exc:
        raise HTTPException(500, f"Forecast error: {exc}")

    try:
        res = pathfinder.compute_route(
            ice_grid=forecast,
            start_latlon=(start_lat, start_lon),
            end_latlon=(dest_lat, dest_lon),
            forecast_window=req.forecast_window,
            safety_weight=req.safety_weight,
        )
    except Exception as exc:
        log.exception("Pathfinding error")
        raise HTTPException(500, f"Pathfinder error: {exc}")

    if res["geojson_path"] is None:
        return res

    return res

@app.get("/api/environmental-vectors", tags=["Environment"])
async def get_environmental_vectors():
    """
    Return vector arrows (MultiLineStrings) for Wind and Ekman Ice Drift
    sampled across the Antarctic operational area as GeoJSON FeatureCollection.
    """
    env = _state.get("env_fields")
    if env is None:
        raise HTTPException(503, "Environmental fields not loaded.")
        
    features = []
    sample_lats = np.linspace(LAT_MAX - 0.5, LAT_MIN + 0.5, 10)
    sample_lons = np.linspace(LON_MIN + 2.0, LON_MAX - 2.0, 14)
    
    import math
    for lat in sample_lats:
        for lon in sample_lons:
            r, c = pathfinder.latlon_to_rowcol(lat, lon)
            w_u, w_v, o_u, o_v, d_u, d_v = pathfinder._get_env(r, c)
            
            # 1. Wind arrow
            w_speed = math.hypot(w_u, w_v)
            if w_speed > 2.0:
                angle_rad = math.atan2(w_v, w_u)
                scale = 0.035
                dx = (w_u / w_speed) * min(1.2, w_speed * scale)
                dy = (w_v / w_speed) * min(0.7, w_speed * scale * 0.6)
                
                tip_x = lon + dx
                tip_y = lat + dy
                barb_len = 0.28 * math.hypot(dx, dy)
                b1_x = tip_x + barb_len * math.cos(angle_rad + math.pi - 0.4)
                b1_y = tip_y + barb_len * math.sin(angle_rad + math.pi - 0.4) * 0.6
                b2_x = tip_x + barb_len * math.cos(angle_rad + math.pi + 0.4)
                b2_y = tip_y + barb_len * math.sin(angle_rad + math.pi + 0.4) * 0.6
                
                features.append({
                    "type": "Feature",
                    "properties": {
                        "kind": "wind",
                        "speed_knots": round(w_speed, 1),
                        "direction": round(math.degrees(math.atan2(w_v, w_u)) % 360, 1),
                        "color": "#38bdf8",
                    },
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": [
                            [[lon, lat], [tip_x, tip_y]],
                            [[b1_x, b1_y], [tip_x, tip_y]],
                            [[b2_x, b2_y], [tip_x, tip_y]],
                        ]
                    }
                })
                
            # 2. Drift arrow (where drift > 0.05 kts)
            d_speed = math.hypot(d_u, d_v)
            if d_speed > 0.05:
                d_angle = math.atan2(d_v, d_u)
                d_dx = (d_u / d_speed) * min(0.9, d_speed * 1.8)
                d_dy = (d_v / d_speed) * min(0.5, d_speed * 1.8 * 0.6)
                
                d_tip_x = lon + d_dx
                d_tip_y = lat + d_dy
                d_barb_len = 0.25 * math.hypot(d_dx, d_dy)
                db1_x = d_tip_x + d_barb_len * math.cos(d_angle + math.pi - 0.4)
                db1_y = d_tip_y + d_barb_len * math.sin(d_angle + math.pi - 0.4) * 0.6
                db2_x = d_tip_x + d_barb_len * math.cos(d_angle + math.pi + 0.4)
                db2_y = d_tip_y + d_barb_len * math.sin(d_angle + math.pi + 0.4) * 0.6
                
                features.append({
                    "type": "Feature",
                    "properties": {
                        "kind": "drift",
                        "speed_knots": round(d_speed, 2),
                        "direction": round(math.degrees(math.atan2(d_v, d_u)) % 360, 1),
                        "color": "#a855f7",
                    },
                    "geometry": {
                        "type": "MultiLineString",
                        "coordinates": [
                            [[lon, lat], [d_tip_x, d_tip_y]],
                            [[db1_x, db1_y], [d_tip_x, d_tip_y]],
                            [[db2_x, db2_y], [d_tip_x, d_tip_y]],
                        ]
                    }
                })
                
    return {
        "type": "FeatureCollection",
        "features": features,
        "summary": {
            "source": env.source,
            "feature_count": len(features)
        }
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
