/**
 * MapView — full-bleed MapLibre GL map with a raster ice forecast overlay.
 *
 * Map is initialised once via useRef + useEffect. The ice image layer is
 * updated reactively whenever `activeImageUrl` or `grid.bounds` changes,
 * without re-mounting the map or reloading the basemap tiles.
 *
 * Basemap: CARTO Dark Matter (free, no API key required).
 *
 * Props:
 *   grid          – { bounds: [minLon, minLat, maxLon, maxLat], width, height }
 *   activeImageUrl – URL of the PNG to overlay (relative to public/)
 *   selectedDay   – current lead day (0 = today/no overlay, 1-3 = forecast)
 *   predictions   – predictions array from forecast.json
 *   onChange      – lead-day selector callback
 */

import { useRef, useEffect, useCallback } from 'react'
import * as maplibregl from 'maplibre-gl'
import { boundsToImageCoords } from '../utils/mapHelpers'
import LeadDaySelector from './LeadDaySelector'
import ConcentrationLegend from './ConcentrationLegend'

const BASEMAP_STYLE =
  'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

const ICE_SOURCE_ID = 'sea-ice-source'
const ICE_LAYER_ID  = 'sea-ice-layer'

// Arctic-centered default view
const DEFAULT_CENTER = [0, 85]
const DEFAULT_ZOOM   = 2.2

export default function MapView({
  grid,
  activeImageUrl,
  selectedDay,
  predictions,
  onChange,
}) {
  const containerRef = useRef(null)
  const mapRef       = useRef(null)
  const mapReadyRef  = useRef(false) // true after 'load' fires

  // ── 1. Initialise map once ─────────────────────────────────
  useEffect(() => {
    if (mapRef.current) return // already initialised

    const map = new maplibregl.Map({
      container:  containerRef.current,
      style:      BASEMAP_STYLE,
      center:     DEFAULT_CENTER,
      zoom:       DEFAULT_ZOOM,
      minZoom:    1,
      maxZoom:    9,
      // Restrict pitch / bearing so the Arctic view stays flat
      pitch:      0,
      bearing:    0,
      attributionControl: false,
    })

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')

    map.on('load', () => {
      mapReadyRef.current = true
      // Trigger the layer update now that the style is loaded
      updateIceLayer(map, activeImageUrl, grid)
    })

    mapRef.current = map

    return () => {
      map.remove()
      mapRef.current = null
      mapReadyRef.current = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // intentionally empty — map must only init once

  // ── 2. Update ice layer when image or grid changes ─────────
  const updateIceLayer = useCallback((map, imageUrl, gridMeta) => {
    if (!map || !mapReadyRef.current) return

    // Remove existing source + layer
    if (map.getLayer(ICE_LAYER_ID))  map.removeLayer(ICE_LAYER_ID)
    if (map.getSource(ICE_SOURCE_ID)) map.removeSource(ICE_SOURCE_ID)

    if (!imageUrl || !gridMeta) return // nothing to render (e.g. selectedDay === 0)

    const coordinates = boundsToImageCoords(gridMeta.bounds)

    map.addSource(ICE_SOURCE_ID, {
      type: 'image',
      url:  imageUrl,
      coordinates,
    })

    map.addLayer({
      id:     ICE_LAYER_ID,
      type:   'raster',
      source: ICE_SOURCE_ID,
      paint: {
        'raster-opacity':      0.82,
        'raster-resampling':   'linear',
        'raster-fade-duration': 400,
      },
    })
  }, [])

  // React to prop changes after initial mount
  useEffect(() => {
    const map = mapRef.current
    if (!map || !mapReadyRef.current) return
    updateIceLayer(map, activeImageUrl, grid)
  }, [activeImageUrl, grid, updateIceLayer])

  // Also update when map finishes loading with already-set props
  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    const onLoad = () => {
      mapReadyRef.current = true
      updateIceLayer(map, activeImageUrl, grid)
    }
    if (map.isStyleLoaded()) {
      mapReadyRef.current = true
      updateIceLayer(map, activeImageUrl, grid)
    } else {
      map.once('load', onLoad)
      return () => map.off('load', onLoad)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // run once to register the event handler

  return (
    <div className="map-column">
      {/* MapLibre canvas container */}
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {/* Map overlays */}
      <LeadDaySelector
        selectedDay={selectedDay}
        predictions={predictions}
        onChange={onChange}
      />
      <ConcentrationLegend />

      <div className="map-attribution">
        © CARTO &nbsp;|&nbsp; © OpenStreetMap contributors
      </div>
    </div>
  )
}
