/**
 * api.js — Kryptonite Navigation Engine API Client
 *
 * All FastAPI calls go through here. Centralised so endpoint URLs
 * and payload shape are defined in exactly one place.
 */

import axios from 'axios'

const BASE = 'http://localhost:8000'

/** @typedef {{ lat: number, lon: number, label: string }} Waypoint */

/**
 * Calculate optimal maritime route via A* + ConvLSTM safety corridors.
 *
 * @param {{
 *   start:        Waypoint,
 *   end:          Waypoint,
 *   forecastWindow: 0 | 24 | 48 | 72,
 *   safetyWeight: number,   // 0 = max fuel efficiency, 100 = max safety
 * }} payload
 * @returns {Promise<RouteResult>}
 */
export async function calculateRoute(payload, signal) {
  const { data } = await axios.post(
    `${BASE}/api/calculate-route`,
    {
      start:           payload.start,
      end:             payload.end,
      forecast_window: payload.forecastWindow,
      safety_weight:   payload.safetyWeight / 100,   // normalise 0–1 for backend
    },
    { signal, timeout: 30_000 },
  )
  return data
}

/**
 * Fetch current ice concentration grid (GeoJSON FeatureCollection).
 * Used to render the dynamic safety-corridor polygons.
 */
export async function fetchIceGrid(forecastWindow = 0, signal) {
  const { data } = await axios.get(`${BASE}/api/ice-grid`, {
    params: { window: forecastWindow },
    signal,
    timeout: 15_000,
  })
  return data
}

/**
 * Fetch iceberg positions as a GeoJSON FeatureCollection of Points.
 */
export async function fetchIcebergs(signal) {
  const { data } = await axios.get(`${BASE}/api/icebergs`, {
    signal,
    timeout: 10_000,
  })
  return data
}
