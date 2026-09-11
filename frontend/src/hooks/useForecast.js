/**
 * useForecast — fetches forecast.json, tracks loading/error state,
 * and flags stale data (generated_at older than STALE_THRESHOLD_HOURS).
 *
 * Returns:
 *   data       – parsed forecast manifest (or last known good data)
 *   loading    – true while the initial fetch is in flight
 *   error      – Error object if fetch failed, null otherwise
 *   isStale    – true if generated_at > 36 h ago OR fetch failed after
 *                a prior successful load
 */

import { useState, useEffect, useRef } from 'react'

const FORECAST_URL = '/forecast.json'
const STALE_THRESHOLD_HOURS = 36

export function useForecast() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)
  const [isStale, setIsStale] = useState(false)

  // Persist the last successfully fetched data so we can display it
  // even if a subsequent refresh fails.
  const lastGoodData = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function fetchForecast() {
      try {
        const res = await fetch(FORECAST_URL, {
          // Bypass service-worker / browser cache so the daily-updated
          // file is always fresh when the user opens the page.
          cache: 'no-store',
        })

        if (!res.ok) {
          throw new Error(`HTTP ${res.status} — could not load forecast.json`)
        }

        const json = await res.json()
        if (cancelled) return

        // Staleness check: compare generated_at against local wall clock
        const generatedAt = new Date(json.generated_at)
        const ageHours = (Date.now() - generatedAt.getTime()) / 3_600_000
        const stale = ageHours > STALE_THRESHOLD_HOURS

        lastGoodData.current = json
        setData(json)
        setError(null)
        setIsStale(stale)
      } catch (err) {
        if (cancelled) return
        // If we have prior data, show it with a staleness flag
        if (lastGoodData.current) {
          setData(lastGoodData.current)
          setIsStale(true)
        }
        setError(err)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    fetchForecast()
    return () => { cancelled = true }
  }, [])

  return { data, loading, error, isStale }
}
