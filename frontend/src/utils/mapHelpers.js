/**
 * mapHelpers.js — utilities for converting forecast.json grid metadata
 * into the coordinate arrays MapLibre expects for ImageSource.
 *
 * MapLibre ImageSource.coordinates format:
 *   [[top-left-lng, top-left-lat],
 *    [top-right-lng, top-right-lat],
 *    [bottom-right-lng, bottom-right-lat],
 *    [bottom-left-lng, bottom-left-lat]]
 *
 * All values are WGS-84 decimal degrees.
 */

/**
 * Convert a [minLon, minLat, maxLon, maxLat] bounds array into
 * the 4-corner coordinates array MapLibre's ImageSource requires.
 *
 * @param {[number, number, number, number]} bounds
 *   [minLon, minLat, maxLon, maxLat] in WGS-84 degrees
 * @returns {[[number,number],[number,number],[number,number],[number,number]]}
 */
export function boundsToImageCoords(bounds) {
  const [minLon, minLat, maxLon, maxLat] = bounds
  return [
    [minLon, maxLat], // top-left
    [maxLon, maxLat], // top-right
    [maxLon, minLat], // bottom-right
    [minLon, minLat], // bottom-left
  ]
}

/**
 * Compute the geographic center of a bounds array.
 *
 * @param {[number, number, number, number]} bounds
 * @returns {{ lng: number, lat: number }}
 */
export function boundsCenter(bounds) {
  const [minLon, minLat, maxLon, maxLat] = bounds
  return {
    lng: (minLon + maxLon) / 2,
    lat: (minLat + maxLat) / 2,
  }
}

/**
 * Format an ice extent value (km²) for display.
 * e.g. 12345678.9 → "12.35M km²"
 *
 * @param {number} km2
 * @returns {string}
 */
export function formatExtent(km2) {
  if (km2 >= 1_000_000) {
    return `${(km2 / 1_000_000).toFixed(2)}M`
  }
  if (km2 >= 1_000) {
    return `${(km2 / 1_000).toFixed(1)}K`
  }
  return km2.toLocaleString()
}

/**
 * Compute the day-over-day percent change between the last two
 * extent_history entries.
 *
 * @param {Array<{date: string, extent_km2: number}>} extentHistory
 * @returns {{ pct: number, direction: 'up' | 'down' | 'neutral' } | null}
 */
export function computeDeltaPct(extentHistory) {
  if (!extentHistory || extentHistory.length < 2) return null
  const prev = extentHistory[extentHistory.length - 2].extent_km2
  const curr = extentHistory[extentHistory.length - 1].extent_km2
  if (prev === 0) return null
  const pct = ((curr - prev) / prev) * 100
  return {
    pct: Math.abs(pct),
    direction: pct > 0.05 ? 'up' : pct < -0.05 ? 'down' : 'neutral',
  }
}
