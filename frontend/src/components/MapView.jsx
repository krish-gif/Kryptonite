/**
 * MapView — Full-bleed MapLibre GL map for Kryptonite Antarctic Navigation System.
 *
 * Renders three dynamic GeoJSON layers:
 *   1. Route LineString (A* optimal path — sonar blue)
 *   2. Iceberg positions (orange point markers with size rings)
 *   3. Safety corridors (semi-transparent hazard polygons)
 *
 * Also renders a scanline CSS overlay for the tactical HUD aesthetic.
 */

import { useRef, useEffect, useCallback } from 'react'
import * as maplibregl from 'maplibre-gl'

const BASEMAP =
  'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

// Maitri Station, Antarctica
const CENTER_LON = 11.73
const CENTER_LAT = -25.0
const DEFAULT_ZOOM = 3.2

// Source/Layer IDs
const IDS = {
  ROUTE_SRC:     'kns-route-src',
  ROUTE_LAYER:   'kns-route-layer',
  ROUTE_GLOW:    'kns-route-glow',
  BERGS_SRC:     'kns-icebergs-src',
  BERGS_LAYER:   'kns-icebergs-layer',
  BERGS_RING:    'kns-icebergs-ring',
  CORR_SRC:      'kns-corridors-src',
  CORR_FILL:     'kns-corridors-fill',
  CORR_BORDER:   'kns-corridors-border',
}

function addOrUpdateSource(map, id, data) {
  if (map.getSource(id)) {
    map.getSource(id).setData(data)
  } else {
    map.addSource(id, { type: 'geojson', data })
  }
}

function removeLayerSafe(map, id) {
  if (map.getLayer(id)) map.removeLayer(id)
}

function removeSourceSafe(map, id) {
  if (map.getSource(id)) map.removeSource(id)
}

function initLayers(map) {
  // ── Safety Corridors ──────────────────────────────────
  map.addSource(IDS.CORR_SRC, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
  map.addLayer({
    id:     IDS.CORR_FILL,
    type:   'fill',
    source: IDS.CORR_SRC,
    paint: {
      'fill-color': [
        'match', ['get', 'risk'],
        'HIGH',     'rgba(255,107,0,0.18)',
        'MODERATE', 'rgba(255,215,0,0.12)',
        'rgba(0,212,255,0.07)',
      ],
    },
  })
  map.addLayer({
    id:     IDS.CORR_BORDER,
    type:   'line',
    source: IDS.CORR_SRC,
    paint: {
      'line-color': [
        'match', ['get', 'risk'],
        'HIGH',     '#ff6b00',
        'MODERATE', '#ffd700',
        '#00d4ff',
      ],
      'line-width':   1.5,
      'line-opacity': 0.7,
      'line-dasharray': [4, 3],
    },
  })

  // ── Icebergs ──────────────────────────────────────────
  map.addSource(IDS.BERGS_SRC, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
  map.addLayer({
    id:     IDS.BERGS_RING,
    type:   'circle',
    source: IDS.BERGS_SRC,
    paint: {
      'circle-color':        'transparent',
      'circle-stroke-color': '#ff6b00',
      'circle-stroke-width': 1.5,
      'circle-radius': ['interpolate', ['linear'], ['get', 'size_km'], 0, 12, 50, 28],
      'circle-opacity':       0.6,
    },
  })
  map.addLayer({
    id:     IDS.BERGS_LAYER,
    type:   'circle',
    source: IDS.BERGS_SRC,
    paint: {
      'circle-color':        '#ff8c33',
      'circle-radius':        5,
      'circle-stroke-color': '#fff',
      'circle-stroke-width':  1,
      'circle-blur':          0.2,
    },
  })

  // ── Route (glow + main line) ──────────────────────────
  map.addSource(IDS.ROUTE_SRC, { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
  map.addLayer({
    id:     IDS.ROUTE_GLOW,
    type:   'line',
    source: IDS.ROUTE_SRC,
    paint: {
      'line-color':   '#00d4ff',
      'line-width':    8,
      'line-opacity':  0.15,
      'line-blur':     4,
    },
  })
  map.addLayer({
    id:     IDS.ROUTE_LAYER,
    type:   'line',
    source: IDS.ROUTE_SRC,
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color':  '#00d4ff',
      'line-width':   2.5,
      'line-opacity': 0.95,
      'line-dasharray': [1, 0],
    },
  })
}

export default function MapView({ result, loading }) {
  const containerRef = useRef(null)
  const mapRef       = useRef(null)
  const readyRef     = useRef(false)
  const popupRef     = useRef(null)

  // ── Init map once ──────────────────────────────────────
  useEffect(() => {
    if (mapRef.current) return

    const map = new maplibregl.Map({
      container:  containerRef.current,
      style:      BASEMAP,
      center:     [CENTER_LON, CENTER_LAT],
      zoom:       DEFAULT_ZOOM,
      pitch:      0,
      bearing:    0,
      minZoom:    1,
      maxZoom:    12,
      attributionControl: false,
    })

    map.addControl(
      new maplibregl.NavigationControl({ showCompass: true }),
      'bottom-right',
    )
    map.addControl(new maplibregl.ScaleControl({ unit: 'nautical' }), 'bottom-left')

    map.on('load', () => {
      initLayers(map)
      readyRef.current = true
    })

    // Iceberg popup
    map.on('click', IDS.BERGS_LAYER, (e) => {
      const props = e.features[0].properties
      if (popupRef.current) popupRef.current.remove()
      popupRef.current = new maplibregl.Popup({ closeButton: false, offset: 12 })
        .setLngLat(e.features[0].geometry.coordinates)
        .setHTML(
          `<div style="line-height:1.6">
             <div style="color:#ff6b00;font-weight:700;margin-bottom:4px">⬡ ICEBERG ${props.id}</div>
             <div>SIZE <span style="color:#00d4ff">${props.size_km} km</span></div>
           </div>`,
        )
        .addTo(map)
    })
    map.on('mouseenter', IDS.BERGS_LAYER, () => (map.getCanvas().style.cursor = 'crosshair'))
    map.on('mouseleave', IDS.BERGS_LAYER, () => (map.getCanvas().style.cursor = ''))

    mapRef.current = map
    return () => { map.remove(); mapRef.current = null; readyRef.current = false }
  }, [])

  // ── Update layers when result changes ─────────────────
  const updateLayers = useCallback((result) => {
    const map = mapRef.current
    if (!map || !readyRef.current) return

    const emptyFC = { type: 'FeatureCollection', features: [] }

    if (!result) {
      addOrUpdateSource(map, IDS.ROUTE_SRC,  emptyFC)
      addOrUpdateSource(map, IDS.BERGS_SRC,  emptyFC)
      addOrUpdateSource(map, IDS.CORR_SRC,   emptyFC)
      return
    }

    // Route
    if (result.route) {
      addOrUpdateSource(map, IDS.ROUTE_SRC, result.route)
      // Fly to fit the route
      const coords = result.route.geometry.coordinates
      const lons   = coords.map(c => c[0])
      const lats   = coords.map(c => c[1])
      map.fitBounds(
        [[Math.min(...lons) - 2, Math.min(...lats) - 2],
         [Math.max(...lons) + 2, Math.max(...lats) + 2]],
        { padding: 80, duration: 1200 },
      )
    }

    // Icebergs
    addOrUpdateSource(map, IDS.BERGS_SRC, result.icebergs ?? emptyFC)

    // Safety corridors
    addOrUpdateSource(map, IDS.CORR_SRC, result.corridors ?? emptyFC)
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!map) return
    if (readyRef.current) {
      updateLayers(result)
    } else {
      map.once('load', () => updateLayers(result))
    }
  }, [result, updateLayers])

  return (
    <div className="relative w-full h-full">
      {/* MapLibre canvas */}
      <div ref={containerRef} className="w-full h-full" />

      {/* Scanline tactical overlay */}
      <div className="scanline-overlay absolute inset-0 pointer-events-none" />

      {/* Loading spinner */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-ocean-950/60 backdrop-blur-sm z-20">
          <div className="flex flex-col items-center gap-4">
            <div className="relative w-16 h-16">
              <div className="absolute inset-0 rounded-full border-2 border-sonar-500/20" />
              <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-sonar-500 animate-spin" />
              <div className="absolute inset-2 rounded-full border border-radar-500/30 animate-ping" />
            </div>
            <span className="font-mono text-xs tracking-widest text-sonar-400 animate-pulse">
              COMPUTING OPTIMAL ROUTE…
            </span>
          </div>
        </div>
      )}

      {/* Watermark */}
      <div className="absolute bottom-3 left-3 font-mono text-[10px] text-slate-600 pointer-events-none z-10">
        KNS v1.0 · © CARTO · © OSM
      </div>

      {/* Live data indicator */}
      <div className="absolute top-3 left-3 flex items-center gap-1.5 z-10
                      bg-ocean-900/80 border border-sonar-500/20 rounded px-2 py-1">
        <span className="w-1.5 h-1.5 rounded-full bg-radar-500 animate-pulse-radar" />
        <span className="font-mono text-[10px] tracking-widest text-slate-500">LIVE · ANTARCTIC SECTOR</span>
      </div>
    </div>
  )
}
