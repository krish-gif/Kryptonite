/**
 * useNavigationEngine — async hook for A* route calculation.
 *
 * Sends POST /api/calculate-route, manages loading / error / result state,
 * and exposes a mock fallback so the UI is fully testable without a live backend.
 *
 * Returns:
 *   calculate(params)  – call to trigger a new route calculation
 *   result             – { route, icebergs, corridors, telemetry } | null
 *   loading            – true while the request is in flight
 *   error              – Error | null
 *   clearResult        – reset state
 */

import { useState, useCallback, useRef } from 'react'
import { calculateRoute } from '../api'

// ── Mock response when backend is unreachable ────────────────────
function buildMockResult(params) {
  // Generate a rough great-circle line from Cape Town → Maitri
  const steps = 18
  const startLon = params.start.lon, startLat = params.start.lat
  const endLon   = params.end.lon,   endLat   = params.end.lat
  const coords   = Array.from({ length: steps }, (_, i) => {
    const t = i / (steps - 1)
    return [
      startLon + (endLon - startLon) * t + Math.sin(t * Math.PI) * 6,
      startLat + (endLat - startLat) * t,
    ]
  })

  // Synthetic icebergs scattered near the route
  const icebergs = {
    type: 'FeatureCollection',
    features: [
      { type: 'Feature', geometry: { type: 'Point', coordinates: [18, -63] }, properties: { id: 'A68', size_km: 38 } },
      { type: 'Feature', geometry: { type: 'Point', coordinates: [24, -67] }, properties: { id: 'B17', size_km: 12 } },
      { type: 'Feature', geometry: { type: 'Point', coordinates: [12, -58] }, properties: { id: 'C09', size_km: 6  } },
      { type: 'Feature', geometry: { type: 'Point', coordinates: [28, -70] }, properties: { id: 'D22', size_km: 21 } },
    ],
  }

  // Safety corridor polygon around the densest ice zone
  const corridors = {
    type: 'FeatureCollection',
    features: [
      {
        type: 'Feature',
        properties: { risk: 'HIGH', concentration: 0.87 },
        geometry: {
          type: 'Polygon',
          coordinates: [[[14,-64],[22,-64],[26,-68],[22,-72],[14,-72],[10,-68],[14,-64]]],
        },
      },
      {
        type: 'Feature',
        properties: { risk: 'MODERATE', concentration: 0.52 },
        geometry: {
          type: 'Polygon',
          coordinates: [[[6,-55],[16,-55],[18,-60],[16,-65],[6,-65],[4,-60],[6,-55]]],
        },
      },
    ],
  }

  const safetyFactor = params.safetyWeight / 100
  const distNm       = 2_800 + Math.round((1 - safetyFactor) * 200)   // shorter = riskier
  const fuelTons     = +(distNm * 0.041).toFixed(1)
  const riskIndex    = +(safetyFactor < 0.4 ? 0.72 - safetyFactor * 0.6 : 0.28 - safetyFactor * 0.1).toFixed(2)

  return {
    route: { type: 'Feature', geometry: { type: 'LineString', coordinates: coords }, properties: {} },
    icebergs,
    corridors,
    telemetry: {
      distance_nm:  distNm,
      fuel_tons:    fuelTons,
      risk_index:   Math.max(0.05, riskIndex),
      eta_hours:    Math.round(distNm / 16),
      waypoints:    coords.length,
      status:       riskIndex > 0.55 ? 'REROUTING AROUND PACK ICE' : 'PATH CLEAR',
      status_level: riskIndex > 0.55 ? 'reroute' : riskIndex > 0.3 ? 'caution' : 'clear',
    },
    mock: true,
  }
}

// ── Hook ─────────────────────────────────────────────────────────
export function useNavigationEngine() {
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState(null)
  const abortRef = useRef(null)

  const calculate = useCallback(async (params) => {
    // Cancel any in-flight request
    if (abortRef.current) abortRef.current.abort()
    abortRef.current = new AbortController()

    setLoading(true)
    setError(null)

    try {
      const data = await calculateRoute(params, abortRef.current.signal)
      setResult(data)
    } catch (err) {
      if (err.name === 'CanceledError' || err.name === 'AbortError') return

      // Backend unreachable — fall back to mock so the UI can demo
      console.warn('[NavigationEngine] Backend unreachable, using mock data:', err.message)
      setResult(buildMockResult(params))
      setError({
        message: `Backend offline — showing simulated route (${err.message})`,
        isMock:  true,
      })
    } finally {
      setLoading(false)
    }
  }, [])

  const clearResult = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  return { calculate, result, loading, error, clearResult }
}
