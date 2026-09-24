import { useState, useCallback, useRef, useEffect } from 'react'
import { calculateRoute, getForecast, getIceGrid, getEnvironmentalVectors } from '../api'

function transformResponse(raw, params) {
  const route = raw.geojson_path ?? null
  const telemetry = {
    distance_km:      raw.distance_km,
    distance_nm:      raw.distance_km ? +(raw.distance_km / 1.852).toFixed(0) : 0,
    risk_status:      raw.risk_status,
    waypoints_count:  raw.waypoint_count,
    eta_hours:        raw.telemetry?.length ? raw.telemetry[raw.telemetry.length - 1].eta_hours : 0,
    safety_weight:    params.safetyWeight,
    forecast_window:  params.forecastWindow,
    status: raw.risk_status === 'NO_ROUTE'
      ? 'NO NAVIGABLE PATH FOUND'
      : raw.risk_status === 'HIGH'
        ? 'HIGH RISK — MARGINAL ICE ZONE'
        : raw.risk_status === 'MODERATE'
          ? 'CAUTION — BUFFER ZONE CROSSING'
          : 'PATH CLEAR — OPEN WATER',
    status_level: raw.risk_status === 'HIGH' ? 'reroute'
                : raw.risk_status === 'MODERATE' ? 'caution'
                : raw.risk_status === 'NO_ROUTE' ? 'noroute'
                : 'clear',
    snapped_start: raw.snapped_start_coords,
    snapped_end:   raw.snapped_end_coords,
    waypoints:     (raw.telemetry ?? []).map(wp => ({
      ...wp,
      navigation_mode: wp.hazard_type ? wp.hazard_type.toUpperCase().replace(/[\s\/]+/g, '_') : 'OPEN_WATER',
      sic: wp.ice_concentration !== undefined ? wp.ice_concentration : (wp.sic ?? 0)
    })),
    route_summary: raw.route_summary,
    environmental_summary: raw.environmental_summary ?? null,
  }

  return { route, telemetry }
}

export function useNavigationEngine() {
  const [result,   setResult]   = useState(null)
  const [forecast, setForecast] = useState(null)
  const [iceGrid,  setIceGrid]  = useState(null)
  const [envVectors, setEnvVectors] = useState(null)
  const [loading,  setLoading]  = useState(false)
  const [fcLoading,setFcLoading]= useState(false)
  const [error,    setError]    = useState(null)
  const abortRef   = useRef(null)
  const fcAbortRef = useRef(null)

  // Fetch forecast stats and environmental vectors on mount
  useEffect(() => {
    const ctrl = new AbortController()
    fcAbortRef.current = ctrl
    setFcLoading(true)
    getForecast(ctrl.signal)
      .then(data => { setForecast(data); setFcLoading(false) })
      .catch(err => {
        if (err.name === 'CanceledError' || err.name === 'AbortError') return
        setFcLoading(false)
      })

    getEnvironmentalVectors(ctrl.signal)
      .then(data => setEnvVectors(data))
      .catch(err => {
        if (err.name !== 'CanceledError') console.error("Env vectors fetch failed:", err)
      })

    return () => ctrl.abort()
  }, [])

  const fetchEnvVectors = useCallback(async () => {
    try {
      const data = await getEnvironmentalVectors()
      setEnvVectors(data)
    } catch (err) {
      console.error("Env vectors fetch failed:", err)
    }
  }, [])

  const fetchIceGrid = useCallback(async (horizon) => {
    const ctrl = new AbortController()
    try {
      const data = await getIceGrid(horizon, ctrl.signal)
      setIceGrid(data)
    } catch (err) {
      if (err.name !== 'CanceledError') console.error("Ice Grid fetch failed:", err)
    }
  }, [])

  const calculate = useCallback(async (params) => {
    if (abortRef.current) abortRef.current.abort()
    abortRef.current = new AbortController()

    setLoading(true)
    setError(null)

    try {
      // Calculate 3 concurrent safety corridors
      const [aggRaw, optRaw, cauRaw] = await Promise.all([
        calculateRoute({ ...params, safetyWeight: 0.1 }, abortRef.current.signal),
        calculateRoute({ ...params, safetyWeight: 1.0 }, abortRef.current.signal),
        calculateRoute({ ...params, safetyWeight: 4.0 }, abortRef.current.signal),
      ])

      const aggressive = transformResponse(aggRaw, { ...params, safetyWeight: 0.1 })
      const optimal    = transformResponse(optRaw, { ...params, safetyWeight: 1.0 })
      const cautious   = transformResponse(cauRaw, { ...params, safetyWeight: 4.0 })

      setResult({
        routes: { aggressive, optimal, cautious },
        activeRoute: 'optimal',
      })
    } catch (err) {
      if (err.name === 'CanceledError' || err.name === 'AbortError') return
      console.error('[NavigationEngine] Route computation failed:', err)
      setError({
        message: `Backend API Error: ${err.message}. Ensure PyTorch backend is running.`,
      })
      setResult(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const setActiveRoute = useCallback((key) => {
    setResult(prev => prev ? { ...prev, activeRoute: key } : null)
  }, [])

  const clearResult = useCallback(() => {
    setResult(null)
    setError(null)
  }, [])

  return { calculate, result, forecast, iceGrid, envVectors, fetchEnvVectors, fetchIceGrid, loading, fcLoading, error, clearResult, setActiveRoute }
}
