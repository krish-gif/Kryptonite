"""
pathfinder.py - Geographic A* Pathfinding over Antarctic Sea-Ice Cost Maps
============================================================================

Dual-Stage Routing:
  - Stage 1: Open Ocean (e.g. Cape Town -> -60S entry point) via Great Circle.
  - Stage 2: A* Ice Evasion (-60S -> Station) via kinematic A* over PyTorch grids.

Physics Engine (integrated from iceberg_model/physics/):
  - Wind drag: F_air = 0.5 * ρ_air * C_Da * A_sail * |V_rel|² (Bigg et al. 1997)
  - Ocean drag: Hydrodynamic current assist/resist via dot product
  - Ekman drift: 2% wind speed, 25° left deflection (Coriolis, Southern Hemisphere)
  - Ice resistance: Exponential penalty scaled by SIC^1.5
  - V_eff = V_base - Ice_Resistance + Wind_Boost + Ocean_Assist + Drift_Boost
"""

from __future__ import annotations

import heapq
import math
from collections import deque

import numpy as np
import scipy.interpolate as si
import pyproj

# ── Grid bounds (must match Colab training crop) ─────────────────────────
GRID_H   = 128
GRID_W   = 128
LAT_MIN  = -72.0
LAT_MAX  = -60.0
LON_MIN  =  10.0
LON_MAX  =  80.0

# ── Granular Hazard Thresholds ───────────────────────────────────────────
HAZARD_MARGINAL = 0.15   # 0.00-0.15: Open Water
HAZARD_PACK     = 0.40   # 0.16-0.40: Marginal Ice
HAZARD_FAST     = 0.70   # 0.41-0.70: Pack Ice; >0.70: Fast Ice (Impassable)

# ── 8-directional movement ───────────────────────────────────────────────
_DIRS = [
    (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
    (-1, -1, math.sqrt(2)), (-1,  1, math.sqrt(2)),
    ( 1, -1, math.sqrt(2)), ( 1,  1, math.sqrt(2)),
]

# ── Rigorous CRS Mapping (WGS84 -> EPSG:3412) ─────────────────────────────
transformer_to_xy = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3412", always_xy=True)
transformer_to_ll = pyproj.Transformer.from_crs("EPSG:3412", "EPSG:4326", always_xy=True)

# Calculate bounding box in projected space
X_CORNERS = [transformer_to_xy.transform(lon, lat) for lon, lat in [
    (LON_MIN, LAT_MIN), (LON_MAX, LAT_MIN), (LON_MIN, LAT_MAX), (LON_MAX, LAT_MAX)
]]
X_MIN = min(x for x, y in X_CORNERS)
X_MAX = max(x for x, y in X_CORNERS)
Y_MIN = min(y for x, y in X_CORNERS)
Y_MAX = max(y for x, y in X_CORNERS)

def latlon_to_rowcol(lat: float, lon: float) -> tuple[int, int]:
    x, y = transformer_to_xy.transform(lon, lat)
    y_frac = (Y_MAX - y) / (Y_MAX - Y_MIN)
    x_frac = (x - X_MIN) / (X_MAX - X_MIN)
    row = max(0, min(GRID_H - 1, int(round(y_frac * (GRID_H - 1)))))
    col = max(0, min(GRID_W - 1, int(round(x_frac * (GRID_W - 1)))))
    return row, col

def rowcol_to_latlon(row: int, col: int) -> tuple[float, float]:
    y = Y_MAX - (row / (GRID_H - 1)) * (Y_MAX - Y_MIN)
    x = X_MIN + (col / (GRID_W - 1)) * (X_MAX - X_MIN)
    lon, lat = transformer_to_ll.transform(x, y)
    return lat, lon

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ── Environmental Fields (injected from environment_service.py) ───────────
# These module-level references are set by set_environment() before routing.
_ENV_WIND_U  = None
_ENV_WIND_V  = None
_ENV_OCEAN_U = None
_ENV_OCEAN_V = None
_ENV_DRIFT_U = None
_ENV_DRIFT_V = None
_ENV_SOURCE  = "none"

def set_environment(wind_u, wind_v, ocean_u, ocean_v, drift_u, drift_v, source="injected"):
    """Inject live or synthetic environmental vector fields into the pathfinder."""
    global _ENV_WIND_U, _ENV_WIND_V, _ENV_OCEAN_U, _ENV_OCEAN_V
    global _ENV_DRIFT_U, _ENV_DRIFT_V, _ENV_SOURCE
    _ENV_WIND_U  = wind_u
    _ENV_WIND_V  = wind_v
    _ENV_OCEAN_U = ocean_u
    _ENV_OCEAN_V = ocean_v
    _ENV_DRIFT_U = drift_u
    _ENV_DRIFT_V = drift_v
    _ENV_SOURCE  = source

def _get_env(r, c):
    """Safely read environment vectors at grid cell (r, c)."""
    if _ENV_WIND_U is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    H, W = _ENV_WIND_U.shape
    r = max(0, min(H-1, int(r)))
    c = max(0, min(W-1, int(c)))
    return (
        float(_ENV_WIND_U[r, c]), float(_ENV_WIND_V[r, c]),
        float(_ENV_OCEAN_U[r, c]), float(_ENV_OCEAN_V[r, c]),
        float(_ENV_DRIFT_U[r, c]), float(_ENV_DRIFT_V[r, c]),
    )

def find_nearest_navigable_cell(ice_grid: np.ndarray, row: int, col: int, max_radius: int = 30) -> tuple[int, int] | None:
    H, W = ice_grid.shape
    if ice_grid[row, col] <= HAZARD_FAST: return (row, col)
    visited, queue = {(row, col)}, deque([(row, col)])
    while queue:
        r, c = queue.popleft()
        for dr, dc, _ in _DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in visited or not (0 <= nr < H and 0 <= nc < W): continue
            if abs(nr - row) > max_radius or abs(nc - col) > max_radius: continue
            visited.add((nr, nc))
            if ice_grid[nr, nc] <= HAZARD_FAST: return (nr, nc)
            queue.append((nr, nc))
    return None

def _heuristic(nr, nc, gr, gc):
    # Theoretical min time: direct distance at max possible assisted speed (e.g. 25 kts)
    n_lat, n_lon = rowcol_to_latlon(nr, nc)
    g_lat, g_lon = rowcol_to_latlon(gr, gc)
    dist_nm = haversine(n_lat, n_lon, g_lat, g_lon) / 1.852
    return dist_nm / 25.0

def astar(ice_grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], safety_weight: float):
    H, W = ice_grid.shape
    sr, sc = start
    gr, gc = goal

    snapped_start = find_nearest_navigable_cell(ice_grid, sr, sc)
    if not snapped_start: return None, (sr, sc), (gr, gc)
    sr, sc = snapped_start

    snapped_goal = find_nearest_navigable_cell(ice_grid, gr, gc)
    if not snapped_goal: return None, snapped_start, (gr, gc)
    gr, gc = snapped_goal

    if (sr, sc) == (gr, gc): return [(sr, sc)], snapped_start, snapped_goal

    g = np.full((H, W), np.inf, dtype=np.float64)
    g[sr, sc] = 0.0
    came_from = {}
    open_set = [(0.0, sr, sc)]
    closed_set = set()

    while open_set:
        f, cr, cc = heapq.heappop(open_set)
        if (cr, cc) == (gr, gc):
            path, cur = [(gr, gc)], (gr, gc)
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path, snapped_start, snapped_goal

        if (cr, cc) in closed_set: continue
        closed_set.add((cr, cc))

        for dr, dc, mv in _DIRS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < H and 0 <= nc < W): continue
            if (nr, nc) in closed_set: continue
            
            sic = ice_grid[nr, nc]
            if sic > HAZARD_FAST: continue
            
            # Ship heading unit vector (grid space, Y inverted)
            mag = math.hypot(dc, -dr)
            if mag == 0: continue
            ship_u = dc / mag
            ship_v = -dr / mag
            
            # Read injected environment at this cell
            w_u, w_v, o_u, o_v, d_u, d_v = _get_env(nr, nc)
            
            # Wind boost via dot product (tailwind positive, headwind negative)
            wind_boost = (w_u * ship_u + w_v * ship_v) * 0.05
            
            # Ocean current assist (direct velocity projection)
            ocean_assist = (o_u * ship_u + o_v * ship_v) * 0.8
            
            # Ice drift only applies if SIC > 15%
            drift_boost = 0.0
            if sic > 0.15:
                drift_boost = (d_u * ship_u + d_v * ship_v) * 2.0
            
            # Ice resistance: exponential penalty (from sea_ice_drag.py physics)
            ice_res = 14.0 * (sic ** 1.5) * (safety_weight + 0.5)
            
            # Effective speed equation
            v_eff = 14.0 - ice_res + wind_boost + ocean_assist + drift_boost
            
            # If combined forces make transit impossible, block this node
            if v_eff < 1.0: continue
            
            # Physical distance in NM (~7NM per cell at this resolution)
            dist_nm = mv * 7.0 
            time_cost = dist_nm / v_eff
            
            tg = g[cr, cc] + time_cost
            if tg < g[nr, nc]:
                g[nr, nc] = tg
                came_from[(nr, nc)] = (cr, cc)
                h = _heuristic(nr, nc, gr, gc)
                heapq.heappush(open_set, (tg + h, nr, nc))

    return None, snapped_start, snapped_goal

def build_great_circle_waypoints(lat1, lon1, lat2, lon2, num_points=20):
    waypoints = []
    for i in range(num_points):
        f = i / float(num_points - 1)
        lat = lat1 + (lat2 - lat1) * f
        lon = lon1 + (lon2 - lon1) * f
        waypoints.append([lon, lat])
    return waypoints

def compute_route(
    ice_grid: np.ndarray,
    start_latlon: tuple[float, float],
    end_latlon:   tuple[float, float],
    forecast_window: int = 72,
    safety_weight: float = 1.0,
) -> dict:
    step_idx = {24: 0, 48: 1, 72: 2}.get(forecast_window, 2)
    if ice_grid.ndim == 3:
        step = min(step_idx, ice_grid.shape[0] - 1)
        grid_2d = ice_grid[step].astype(np.float64)
    else:
        grid_2d = ice_grid.astype(np.float64)
    grid_2d = np.clip(grid_2d, 0.0, 1.0)
    
    slat, slon = start_latlon
    elat, elon = end_latlon

    # Dual Stage Logic
    stage1_coords = []
    if slat > LAT_MAX:
        entry_lat = LAT_MAX
        entry_lon = elon
        dist_s1 = haversine(slat, slon, entry_lat, entry_lon)
        num_pts = max(2, int(dist_s1 / 50))
        stage1_coords = build_great_circle_waypoints(slat, slon, entry_lat, entry_lon, num_pts)
        astar_start = (entry_lat, entry_lon)
    else:
        astar_start = start_latlon

    # Stage 2: A* through ice using physical kinematics
    start_rc = latlon_to_rowcol(*astar_start)
    end_rc   = latlon_to_rowcol(*end_latlon)

    path_rc, snapped_s, snapped_e = astar(grid_2d, start_rc, end_rc, safety_weight)

    snap_s_latlon = rowcol_to_latlon(*snapped_s)
    snap_e_latlon = rowcol_to_latlon(*snapped_e)
    snapped_start = [round(snap_s_latlon[1], 5), round(snap_s_latlon[0], 5)]
    snapped_end   = [round(snap_e_latlon[1], 5), round(snap_e_latlon[0], 5)]

    if path_rc is None:
        return {
            "geojson_path": None,
            "route_coords": [],
            "distance_km": 0.0,
            "risk_status": "NO_ROUTE",
            "waypoint_count": 0,
            "snapped_start_coords": snapped_start,
            "snapped_end_coords": snapped_end,
            "telemetry": [],
            "route_summary": None,
        }

    # Remove consecutive duplicates for spline interpolation
    unique_rc = []
    for p in path_rc:
        if not unique_rc or p != unique_rc[-1]:
            unique_rc.append(p)
            
    # Subsample & Smooth Stage 2 A* path using B-splines
    if len(unique_rc) > 5:
        rc_arr = np.array(unique_rc)
        r = rc_arr[:, 0]
        c = rc_arr[:, 1]
        # s controls the smoothness. k=3 is cubic spline.
        try:
            tck, u = si.splprep([r, c], s=10.0, k=3)
            # Generate highly detailed sub-grid points
            u_new = np.linspace(0, 1.0, max(150, len(unique_rc) * 2))
            r_new, c_new = si.splev(u_new, tck)
            sampled = list(zip(r_new, c_new))
        except Exception:
            # Fallback if spline fails
            sampled = unique_rc[::max(1, len(unique_rc)//100)]
    else:
        sampled = unique_rc

    stage2_coords = [ [round(rowcol_to_latlon(r, c)[1], 5), round(rowcol_to_latlon(r, c)[0], 5)] for r, c in sampled ]

    # Stitch Stage 1 and Stage 2
    if stage1_coords and stage2_coords:
        if haversine(stage1_coords[-1][1], stage1_coords[-1][0], stage2_coords[0][1], stage2_coords[0][0]) < 5.0:
            stage2_coords = stage2_coords[1:]
    
    full_coords = stage1_coords + stage2_coords

    # Generate Spatiotemporal Telemetry
    telemetry = []
    accumulated_km = 0.0
    accumulated_hours = 0.0
    
    total_open_water_km = 0.0
    total_marginal_ice_km = 0.0
    total_pack_ice_km = 0.0
    max_ice_encountered = 0.0
    critical_hurdles_count = 0
    max_crosswind = 0.0
    all_drift_speeds = []
    
    if ice_grid.ndim == 2:
        ice_grid_3d = np.array([ice_grid, ice_grid, ice_grid])
    else:
        ice_grid_3d = ice_grid
    num_steps = ice_grid_3d.shape[0]

    for i, (lon, lat) in enumerate(full_coords):
        if i > 0:
            prev_lon, prev_lat = full_coords[i-1]
            segment_km = haversine(prev_lat, prev_lon, lat, lon)
            accumulated_km += segment_km
        else:
            prev_lon, prev_lat = lon, lat
            segment_km = 0.0

        is_stage1 = lat > LAT_MAX
        
        # Default kinematics for stage 1 (Open Ocean, no ice model)
        wind_speed = 0.0
        wind_dir = 0.0
        ocean_speed = 0.0
        ocean_dir = 0.0
        drift_speed = 0.0
        drift_angle = 0.0
        w_u, w_v = 0.0, 0.0
        o_u, o_v = 0.0, 0.0
        d_u, d_v = 0.0, 0.0
        speed_knots = 14.0
        hazard_type = "Open Water"
        sic = 0.0
        
        if is_stage1:
            total_open_water_km += segment_km
        else:
            r, c = latlon_to_rowcol(lat, lon)
            step_idx = min(int(accumulated_hours // 24), num_steps - 1)
            sic = float(ice_grid_3d[step_idx, r, c])
            max_ice_encountered = max(max_ice_encountered, sic)
            
            # Read environment vectors at this cell
            w_u, w_v, o_u, o_v, d_u, d_v = _get_env(r, c)
            wind_speed = math.hypot(w_u, w_v)
            wind_dir = math.degrees(math.atan2(w_v, w_u)) % 360
            ocean_speed = math.hypot(o_u, o_v)
            ocean_dir = math.degrees(math.atan2(o_v, o_u)) % 360
            drift_speed = math.hypot(d_u, d_v)
            drift_angle = math.degrees(math.atan2(d_v, d_u)) % 360
            
            # Ship heading from path delta
            dy = lat - prev_lat
            dx = lon - prev_lon
            hmag = math.hypot(dx, dy)
            ship_u, ship_v = (dx/hmag, dy/hmag) if hmag > 0 else (1.0, 0.0)
            
            # Reconstruct V_eff with full physics
            wind_boost = (w_u * ship_u + w_v * ship_v) * 0.05
            ocean_assist = (o_u * ship_u + o_v * ship_v) * 0.8
            drift_boost = (d_u * ship_u + d_v * ship_v) * 2.0 if sic > 0.15 else 0.0
            ice_res = 14.0 * (sic ** 1.5) * (safety_weight + 0.5)
            v_eff = max(0.0, 14.0 - ice_res + wind_boost + ocean_assist + drift_boost)
            speed_knots = v_eff
            
            # Track max crosswind
            cross = abs(w_u * (-ship_v) + w_v * ship_u)
            max_crosswind = max(max_crosswind, cross)
            all_drift_speeds.append(drift_speed)
            
            if sic <= HAZARD_MARGINAL:
                hazard_type = "Open Water"
                total_open_water_km += segment_km
            elif sic <= HAZARD_PACK:
                hazard_type = "Marginal Ice Zone"
                total_marginal_ice_km += segment_km
            elif sic <= HAZARD_FAST:
                hazard_type = "Pack Ice"
                total_pack_ice_km += segment_km
                if segment_km > 0: critical_hurdles_count += 1
            else:
                hazard_type = "Fast Ice / Land"
                if segment_km > 0: critical_hurdles_count += 1

        if i > 0 and speed_knots > 0:
            accumulated_hours += segment_km / (speed_knots * 1.852)

        telemetry.append({
            "step": i,
            "coordinates": [round(lon, 4), round(lat, 4)],
            "ice_concentration": round(sic, 3),
            "hazard_type": hazard_type,
            "speed_knots": round(speed_knots, 1),
            "wind_speed_knots": round(wind_speed, 1),
            "wind_direction": round(wind_dir, 1),
            "ice_drift_vector": [round(d_u, 2), round(d_v, 2)] if not is_stage1 else [0, 0],
            "drift_velocity": round(drift_speed, 2) if not is_stage1 else 0.0,
            "drift_angle": round(drift_angle, 1) if not is_stage1 else 0.0,
            "ocean_current_knots": round(ocean_speed, 2) if not is_stage1 else 0.0,
            "ocean_direction": round(ocean_dir, 1) if not is_stage1 else 0.0,
            "calculated_ship_speed": round(speed_knots, 1),
            "effective_ship_speed": round(speed_knots, 1),
            "distance_km": round(accumulated_km, 1),
            "eta_hours": round(accumulated_hours, 1),
        })

    risk = "LOW" if max_ice_encountered <= HAZARD_MARGINAL else ("MODERATE" if max_ice_encountered <= HAZARD_PACK else "HIGH")

    route_summary = {
        "total_open_water_km": round(total_open_water_km, 1),
        "total_marginal_ice_km": round(total_marginal_ice_km, 1),
        "total_pack_ice_km": round(total_pack_ice_km, 1),
        "max_ice_concentration_encountered": round(max_ice_encountered, 3),
        "critical_hurdles_count": critical_hurdles_count,
    }
    
    environmental_summary = {
        "data_source": _ENV_SOURCE,
        "max_crosswind_knots": round(max_crosswind, 1),
        "avg_drift_velocity": round(float(np.mean(all_drift_speeds)) if all_drift_speeds else 0.0, 3),
        "max_drift_velocity": round(float(max(all_drift_speeds)) if all_drift_speeds else 0.0, 3),
    }

    props = {
        "distance_km":          round(accumulated_km, 1),
        "risk_status":          risk,
        "waypoint_count":       len(full_coords),
        "forecast_window_h":    forecast_window,
        "safety_weight":        safety_weight,
        "snapped_start_coords": snapped_start,
        "snapped_end_coords":   snapped_end,
        "waypoints":            telemetry,
        "route_summary":        route_summary,
        "environmental_summary": environmental_summary,
    }

    return {
        "geojson_path": {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": full_coords,
            },
            "properties": props,
        },
        "route_coords":           full_coords,
        "distance_km":            round(accumulated_km, 1),
        "risk_status":            risk,
        "waypoint_count":         len(full_coords),
        "snapped_start_coords":   snapped_start,
        "snapped_end_coords":     snapped_end,
        "telemetry":              telemetry,
        "route_summary":          route_summary,
        "environmental_summary":  environmental_summary,
    }
