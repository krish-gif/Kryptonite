/**
 * api.js — Kryptonite Navigation Engine · FastAPI Client
 */
import axios from 'axios'

const BASE = 'http://localhost:8000'

export async function getIceGrid(horizon = 24, signal) {
  const { data } = await axios.get(`${BASE}/api/ice-grid`, {
    params: { horizon },
    signal,
    timeout: 30_000,
  })
  return data // GeoJSON FeatureCollection
}

export async function getForecast(signal) {
  const { data } = await axios.get(`${BASE}/api/forecast`, {
    signal,
    timeout: 30_000,
  })
  return data
}

export async function calculateRoute(params, signal) {
  const fw = [24, 48, 72].includes(params.forecastWindow) ? params.forecastWindow : 72
  const safetyWeight = Number(((params.safetyWeight / 100) * 10).toFixed(2))

  const payload = {
    start_coords:    [params.start.lat, params.start.lon],
    end_coords:      [params.end.lat,   params.end.lon],
    forecast_window: fw,
    safety_weight:   safetyWeight,
  }

  const { data } = await axios.post(`${BASE}/api/calculate-route`, payload, {
    signal,
    timeout: 45_000,
  })
  return data
}

export async function getHealth(signal) {
  const { data } = await axios.get(`${BASE}/api/health`, {
    signal,
    timeout: 5_000,
  })
  return data
}

export async function getEnvironmentalVectors(signal) {
  const { data } = await axios.get(`${BASE}/api/environmental-vectors`, {
    signal,
    timeout: 15_000,
  })
  return data // GeoJSON FeatureCollection of vector arrows
}
