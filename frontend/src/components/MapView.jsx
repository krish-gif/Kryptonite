import { useRef, useEffect, useState, useCallback } from 'react'
import * as maplibregl from 'maplibre-gl'
import { Wind, Compass, Eye, EyeOff, Layers, Navigation, MapPin } from 'lucide-react'

const BASEMAP = {
  version: 8,
  sources: {
    'google-satellite': {
      type: 'raster',
      tiles: ['https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'],
      tileSize: 256,
      attribution: 'Map data © Google'
    }
  },
  layers: [{
    id: 'google-satellite-layer',
    type: 'raster',
    source: 'google-satellite',
    minzoom: 0,
    maxzoom: 22
  }]
}
const CENTER   = [35.0, -58.0]
const DEF_ZOOM = 2.8

const IDS = {
  ICE_SRC:           'kns-ice-src',
  ICE_FILL:          'kns-ice-fill',
  ICE_OUTLINE:       'kns-ice-outline',
  ENV_SRC:           'kns-env-src',
  WIND_LINE:         'kns-wind-line',
  DRIFT_LINE:        'kns-drift-line',
  ROUTE_SRC:         'kns-route-src',
  ROUTE_GLOW:        'kns-route-glow',
  ROUTE_LINE:        'kns-route-line',
  CORRIDOR_SRC:      'kns-corridor-src',
  CORRIDOR_FILL:     'kns-corridor-fill',
  CORRIDOR_OUTLINE:  'kns-corridor-outline',
  SNAP_SRC:          'kns-snap-src',
  SNAP_RING:         'kns-snap-ring',
  SNAP_LAYER:        'kns-snap-layer',
  STATION_LINE:      'kns-station-line',
  WP_SRC:            'kns-wp-src',
  WP_DOTS:           'kns-wp-dots',
  WP_PULSE:          'kns-wp-pulse',
  SELECTED_WP_SRC:   'kns-selected-wp-src',
  SELECTED_WP_RING:  'kns-selected-wp-ring',
  SELECTED_WP_DOT:   'kns-selected-wp-dot',
  CUSTOM_SRC:        'kns-custom-src',
  CUSTOM_LAYER:      'kns-custom-layer',
}

const EMPTY_FC = { type: 'FeatureCollection', features: [] }

function upsertSource(map, id, data) {
  if (map.getSource(id)) map.getSource(id).setData(data)
  else map.addSource(id, { type: 'geojson', data })
}

// Break a route into per-segment color features
function buildRouteSegments(routeFeature, type = 'optimal') {
  if (!routeFeature?.geometry?.coordinates) return []
  const coords    = routeFeature.geometry.coordinates
  const telemetry = routeFeature.properties?.waypoints || []

  if (telemetry.length === 0 || coords.length < 2) {
    const colors = { optimal: '#fbbf24', aggressive: '#f87171', cautious: '#34d399' }
    return [{ type: 'Feature', properties: { color: colors[type] || '#fbbf24', type, is_open_water: false }, geometry: routeFeature.geometry }]
  }

  return coords.slice(0, -1).map((coord, i) => {
    const wp  = telemetry[i + 1] || telemetry[i]
    const sic = wp?.ice_concentration ?? wp?.sic ?? 0
    let color = '#00d4ff'

    if (type === 'optimal') {
      if (sic > 0.40) color = '#f87171'
      else if (sic > 0.15) color = '#fbbf24'
      else color = '#00d4ff'
    } else if (type === 'aggressive') {
      color = sic > 0.15 ? '#ff6b6b' : '#f87171'
    } else if (type === 'cautious') {
      color = sic > 0.15 ? '#6ee7b7' : '#34d399'
    }

    return {
      type: 'Feature',
      properties: { color, type, is_open_water: sic <= 0.15 },
      geometry: { type: 'LineString', coordinates: [coord, coords[i + 1]] }
    }
  })
}

// Build safety corridor as a buffer polygon around route
function buildCorridorPolygon(routeFeature, type = 'optimal', bufferDeg = 0.3) {
  if (!routeFeature?.geometry?.coordinates) return null
  const coords = routeFeature.geometry.coordinates
  if (coords.length < 2) return null

  // Simple perpendicular buffer
  const upper = [], lower = []
  for (let i = 0; i < coords.length; i++) {
    const [cx, cy] = coords[i]
    let dx = 0, dy = 1
    if (i < coords.length - 1) {
      const [nx, ny] = coords[i + 1]
      const len = Math.hypot(nx - cx, ny - cy) || 1
      dx = -(ny - cy) / len
      dy = (nx - cx) / len
    }
    upper.push([cx + dx * bufferDeg, cy + dy * bufferDeg])
    lower.push([cx - dx * bufferDeg, cy - dy * bufferDeg])
  }

  const ring = [...upper, ...[...lower].reverse(), upper[0]]
  const colors = { optimal: 'rgba(251,191,36,0.08)', aggressive: 'rgba(248,113,113,0.06)', cautious: 'rgba(52,211,153,0.08)' }
  const outlines = { optimal: '#fbbf24', aggressive: '#f87171', cautious: '#34d399' }

  return {
    type: 'Feature',
    properties: {
      type,
      fill_color: colors[type] || colors.optimal,
      outline_color: outlines[type] || outlines.optimal,
    },
    geometry: { type: 'Polygon', coordinates: [ring] }
  }
}

// Build clickable waypoint dot GeoJSON from route telemetry
function buildWaypointDots(routeFeature) {
  if (!routeFeature?.geometry?.coordinates) return []
  const telemetry = routeFeature.properties?.waypoints || []
  const coords    = routeFeature.geometry.coordinates

  // Sample every 8th waypoint to avoid clutter
  return telemetry
    .filter((_, i) => i % 8 === 0 && i < coords.length)
    .map((wp, idx) => {
      const coord = coords[idx * 8] || wp.coordinates
      return {
        type: 'Feature',
        properties: {
          ...wp,
          sic: wp.ice_concentration ?? wp.sic ?? 0,
        },
        geometry: { type: 'Point', coordinates: Array.isArray(coord) ? coord : wp.coordinates }
      }
    })
}

function buildPopupHTML(props) {
  const sic = props.sic ?? props.ice_concentration ?? 0
  const barColor = sic > 0.4 ? '#f87171' : sic > 0.15 ? '#fbbf24' : '#00d4ff'
  const barW = Math.max(2, sic * 100).toFixed(0)
  const vEff = props.effective_ship_speed ?? props.speed_knots ?? 0
  const windSpd = props.wind_speed_knots ?? 0
  const windDir = props.wind_direction ?? 0
  const drift   = props.drift_velocity ?? 0
  const ocean   = props.ocean_current_knots ?? 0

  return `
    <div style="min-width:200px;line-height:1.5">
      <div style="font-weight:700;color:#00d4ff;font-size:11px;border-bottom:1px solid rgba(0,212,255,0.2);padding-bottom:4px;margin-bottom:6px">
        📍 T+${props.eta_hours ?? 0}h WAYPOINT
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;font-size:10px">
        <span style="color:#64748b">HAZARD</span>
        <span style="color:${barColor};font-weight:600">${props.hazard_type ?? '—'}</span>
        <span style="color:#64748b">SIC</span>
        <span style="color:${barColor}">${(sic * 100).toFixed(1)}%</span>
        <span style="color:#64748b">V_eff</span>
        <span style="color:#a78bfa">${vEff.toFixed(1)} kts</span>
        <span style="color:#64748b">WIND</span>
        <span style="color:#38bdf8">${windSpd.toFixed(1)} kts @ ${windDir.toFixed(0)}°</span>
        ${drift > 0.05 ? `<span style="color:#64748b">DRIFT</span><span style="color:#c084fc">${drift.toFixed(2)} kts</span>` : ''}
        ${ocean > 0.05 ? `<span style="color:#64748b">OCEAN</span><span style="color:#2dd4bf">${ocean.toFixed(2)} kts</span>` : ''}
        <span style="color:#64748b">DIST</span>
        <span style="color:#e2e8f0">${(props.distance_km ?? 0).toFixed(0)} km</span>
      </div>
      <div style="margin-top:6px;height:4px;background:rgba(15,30,50,0.8);border-radius:2px;overflow:hidden">
        <div style="width:${barW}%;height:100%;background:${barColor};border-radius:2px"></div>
      </div>
      <div style="margin-top:3px;font-size:9px;color:#475569">${props.coordinates ? `${props.coordinates[1].toFixed(4)}°S, ${props.coordinates[0].toFixed(4)}°E` : ''}</div>
    </div>
  `
}

function initLayers(map) {
  // 1. ICE GRID
  map.addSource(IDS.ICE_SRC, { type: 'geojson', data: EMPTY_FC })
  map.addLayer({
    id: IDS.ICE_FILL, type: 'fill', source: IDS.ICE_SRC,
    paint: {
      'fill-color': [
        'interpolate', ['linear'], ['get', 'sic'],
        0.00, 'rgba(0,212,255,0.00)',
        0.05, 'rgba(0,212,255,0.06)',
        0.15, 'rgba(0,212,255,0.18)',
        0.30, 'rgba(100,180,255,0.28)',
        0.40, 'rgba(251,191,36,0.40)',
        0.55, 'rgba(248,113,113,0.52)',
        0.70, 'rgba(239,68,68,0.65)',
        0.85, 'rgba(220,38,38,0.78)',
        1.00, 'rgba(255,255,255,0.88)',
      ],
      'fill-opacity': 0.9,
    },
  })
  map.addLayer({
    id: IDS.ICE_OUTLINE, type: 'line', source: IDS.ICE_SRC,
    filter: ['>=', ['get', 'sic'], 0.70],
    paint: {
      'line-color': 'rgba(239,68,68,0.4)',
      'line-width': 0.5,
      'line-opacity': 0.5,
    }
  })

  // 2. SAFETY CORRIDORS (filled buffers around each route variant)
  map.addSource(IDS.CORRIDOR_SRC, { type: 'geojson', data: EMPTY_FC })
  map.addLayer({
    id: IDS.CORRIDOR_FILL, type: 'fill', source: IDS.CORRIDOR_SRC,
    paint: {
      'fill-color': ['get', 'fill_color'],
      'fill-opacity': 0.6,
    }
  })
  map.addLayer({
    id: IDS.CORRIDOR_OUTLINE, type: 'line', source: IDS.CORRIDOR_SRC,
    paint: {
      'line-color': ['get', 'outline_color'],
      'line-width': 1.2,
      'line-opacity': 0.4,
      'line-dasharray': [4, 3],
    }
  })

  // 3. ENVIRONMENTAL VECTORS
  map.addSource(IDS.ENV_SRC, { type: 'geojson', data: EMPTY_FC })
  map.addLayer({
    id: IDS.WIND_LINE, type: 'line', source: IDS.ENV_SRC,
    filter: ['==', ['get', 'kind'], 'wind'],
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': '#38bdf8', 'line-width': 1.8, 'line-opacity': 0.75 }
  })
  map.addLayer({
    id: IDS.DRIFT_LINE, type: 'line', source: IDS.ENV_SRC,
    filter: ['==', ['get', 'kind'], 'drift'],
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': '#c084fc', 'line-width': 2.2, 'line-opacity': 0.85 }
  })

  // 4. ROUTE LINES (drawn above corridors)
  map.addSource(IDS.ROUTE_SRC, { type: 'geojson', data: EMPTY_FC })
  map.addLayer({
    id: IDS.ROUTE_GLOW, type: 'line', source: IDS.ROUTE_SRC,
    filter: ['all', ['!=', ['get', 'is_open_water'], true], ['==', ['get', 'type'], 'optimal']],
    paint: { 'line-color': ['get', 'color'], 'line-width': 14, 'line-opacity': 0.18, 'line-blur': 10 },
  })
  map.addLayer({
    id: IDS.ROUTE_LINE, type: 'line', source: IDS.ROUTE_SRC,
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': ['get', 'color'],
      'line-width': ['match', ['get', 'type'], 'optimal', 3.5, 1.8],
      'line-opacity': ['match', ['get', 'type'], 'optimal', 0.96, 0.38],
      'line-dasharray': ['case', ['==', ['get', 'is_open_water'], true], ['literal', [3, 3]], ['literal', [1]]]
    },
  })

  // 5. CLICKABLE WAYPOINT DOTS
  map.addSource(IDS.WP_SRC, { type: 'geojson', data: EMPTY_FC })
  map.addLayer({
    id: IDS.WP_PULSE, type: 'circle', source: IDS.WP_SRC,
    paint: {
      'circle-radius': 9,
      'circle-color': 'transparent',
      'circle-stroke-color': [
        'case',
        ['>', ['get', 'sic'], 0.4], '#f87171',
        ['>', ['get', 'sic'], 0.15], '#fbbf24',
        '#00d4ff'
      ],
      'circle-stroke-width': 1.5,
      'circle-stroke-opacity': 0.45,
    }
  })
  map.addLayer({
    id: IDS.WP_DOTS, type: 'circle', source: IDS.WP_SRC,
    paint: {
      'circle-radius': 4,
      'circle-color': [
        'case',
        ['>', ['get', 'sic'], 0.4], '#f87171',
        ['>', ['get', 'sic'], 0.15], '#fbbf24',
        '#00d4ff'
      ],
      'circle-stroke-color': 'rgba(255,255,255,0.9)',
      'circle-stroke-width': 1.5,
      'circle-opacity': 0.9,
    }
  })

  // 6. SNAP POINT MARKERS
  map.addSource(IDS.SNAP_SRC, { type: 'geojson', data: EMPTY_FC })
  map.addLayer({
    id: IDS.STATION_LINE, type: 'line', source: IDS.SNAP_SRC,
    filter: ['==', ['get', 'kind'], 'connection'],
    paint: { 'line-color': '#94a3b8', 'line-width': 1.5, 'line-dasharray': [2, 2], 'line-opacity': 0.7 }
  })
  map.addLayer({
    id: IDS.SNAP_RING, type: 'circle', source: IDS.SNAP_SRC,
    filter: ['!=', ['get', 'kind'], 'connection'],
    paint: {
      'circle-color': 'transparent',
      'circle-stroke-color': ['match', ['get', 'kind'], 'anchor', '#fbbf24', 'station', '#f87171', 'start', '#34d399', '#00d4ff'],
      'circle-stroke-width': 2.5,
      'circle-radius': 11,
      'circle-opacity': 0.8,
    }
  })
  map.addLayer({
    id: IDS.SNAP_LAYER, type: 'circle', source: IDS.SNAP_SRC,
    filter: ['!=', ['get', 'kind'], 'connection'],
    paint: {
      'circle-color': ['match', ['get', 'kind'], 'anchor', '#fbbf24', 'station', '#f87171', 'start', '#34d399', '#00d4ff'],
      'circle-radius': 5,
      'circle-stroke-color': '#fff',
      'circle-stroke-width': 1.5,
    }
  })

  // 7. SELECTED WAYPOINT HIGHLIGHT
  map.addSource(IDS.SELECTED_WP_SRC, { type: 'geojson', data: EMPTY_FC })
  map.addLayer({
    id: IDS.SELECTED_WP_RING, type: 'circle', source: IDS.SELECTED_WP_SRC,
    paint: {
      'circle-color': 'rgba(251,191,36,0.15)',
      'circle-stroke-color': '#fbbf24',
      'circle-stroke-width': 2.5,
      'circle-radius': 16,
    }
  })
  map.addLayer({
    id: IDS.SELECTED_WP_DOT, type: 'circle', source: IDS.SELECTED_WP_SRC,
    paint: { 'circle-color': '#fbbf24', 'circle-radius': 6, 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 2 }
  })

  // 8. CUSTOM CLICK POINTS
  map.addSource(IDS.CUSTOM_SRC, { type: 'geojson', data: EMPTY_FC })
  map.addLayer({
    id: IDS.CUSTOM_LAYER, type: 'circle', source: IDS.CUSTOM_SRC,
    paint: {
      'circle-radius': ['match', ['get', 'kind'], 'start', 8, 7],
      'circle-color': ['match', ['get', 'kind'], 'start', '#34d399', '#a78bfa'],
      'circle-stroke-color': '#fff',
      'circle-stroke-width': 2,
      'circle-opacity': 0.95,
    }
  })
}

export default function MapView({
  result,
  iceGrid,
  envVectors,
  selectedWaypoint = null,
  loading,
  onMapClick,
  customPoints = []
}) {
  const containerRef     = useRef(null)
  const mapRef           = useRef(null)
  const readyRef         = useRef(false)
  const popupRef         = useRef(null)
  const clickHandlerRef  = useRef(onMapClick)

  const [showIce,      setShowIce]      = useState(true)
  const [showWind,     setShowWind]     = useState(true)
  const [showDrift,    setShowDrift]    = useState(true)
  const [showCorridors,setShowCorridors]= useState(true)
  const [showWpDots,   setShowWpDots]   = useState(true)

  useEffect(() => { clickHandlerRef.current = onMapClick }, [onMapClick])

  // Init map
  useEffect(() => {
    if (mapRef.current) return
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: BASEMAP,
      center: CENTER,
      zoom: DEF_ZOOM,
    })
    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'bottom-right')

    // Map click for custom points (avoid clicking on WP dots)
    map.on('click', (e) => {
      const features = map.queryRenderedFeatures(e.point, { layers: [IDS.WP_DOTS] })
      if (features.length === 0 && clickHandlerRef.current) {
        clickHandlerRef.current(e.lngLat)
      }
    })

    map.on('load', () => {
      initLayers(map)
      readyRef.current = true
      if (iceGrid)     upsertSource(map, IDS.ICE_SRC, iceGrid)
      if (envVectors)  upsertSource(map, IDS.ENV_SRC, envVectors)
    })

    mapRef.current = map
    return () => { map.remove(); mapRef.current = null; readyRef.current = false }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Waypoint popup on click
  useEffect(() => {
    const map = mapRef.current
    if (!map) return

    const onWpClick = (e) => {
      const props = e.features[0].properties
      // Parse coordinates from GeoJSON properties (stored as JSON string sometimes)
      let coords = e.lngLat
      try {
        const rawCoords = typeof props.coordinates === 'string' ? JSON.parse(props.coordinates) : props.coordinates
        if (Array.isArray(rawCoords)) coords = { lng: rawCoords[0], lat: rawCoords[1] }
      } catch (_) {}

      if (popupRef.current) popupRef.current.remove()
      popupRef.current = new maplibregl.Popup({ closeButton: true, maxWidth: '260px', offset: 12 })
        .setLngLat([coords.lng, coords.lat])
        .setHTML(buildPopupHTML(props))
        .addTo(map)
    }

    const onWpEnter = () => { map.getCanvas().style.cursor = 'pointer' }
    const onWpLeave = () => { map.getCanvas().style.cursor = '' }

    if (readyRef.current) {
      map.on('click',       IDS.WP_DOTS, onWpClick)
      map.on('mouseenter',  IDS.WP_DOTS, onWpEnter)
      map.on('mouseleave',  IDS.WP_DOTS, onWpLeave)
    } else {
      map.on('load', () => {
        map.on('click',       IDS.WP_DOTS, onWpClick)
        map.on('mouseenter',  IDS.WP_DOTS, onWpEnter)
        map.on('mouseleave',  IDS.WP_DOTS, onWpLeave)
      })
    }
    return () => {
      if (map) {
        map.off('click',      IDS.WP_DOTS, onWpClick)
        map.off('mouseenter', IDS.WP_DOTS, onWpEnter)
        map.off('mouseleave', IDS.WP_DOTS, onWpLeave)
      }
    }
  }, [])

  // Render Ice Grid
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current) return
    upsertSource(map, IDS.ICE_SRC, iceGrid ?? EMPTY_FC)
  }, [iceGrid])

  // Render Environmental Vectors
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current) return
    upsertSource(map, IDS.ENV_SRC, envVectors ?? EMPTY_FC)
  }, [envVectors])

  // Layer Visibility
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current) return
    const vis = (id, show) => { if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', show ? 'visible' : 'none') }
    vis(IDS.ICE_FILL,         showIce)
    vis(IDS.ICE_OUTLINE,      showIce)
    vis(IDS.WIND_LINE,        showWind)
    vis(IDS.DRIFT_LINE,       showDrift)
    vis(IDS.CORRIDOR_FILL,    showCorridors)
    vis(IDS.CORRIDOR_OUTLINE, showCorridors)
    vis(IDS.WP_DOTS,          showWpDots)
    vis(IDS.WP_PULSE,         showWpDots)
  }, [showIce, showWind, showDrift, showCorridors, showWpDots])

  // Render Selected Waypoint Beacon
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current) return
    if (selectedWaypoint?.length === 2) {
      upsertSource(map, IDS.SELECTED_WP_SRC, {
        type: 'FeatureCollection',
        features: [{ type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: selectedWaypoint } }]
      })
    } else {
      upsertSource(map, IDS.SELECTED_WP_SRC, EMPTY_FC)
    }
  }, [selectedWaypoint])

  // Render Routes, Corridors, Waypoint Dots, Snap Markers
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current) return

    const snapFeatures = []

    if (!result?.routes) {
      upsertSource(map, IDS.ROUTE_SRC,    EMPTY_FC)
      upsertSource(map, IDS.CORRIDOR_SRC, EMPTY_FC)
      upsertSource(map, IDS.WP_SRC,       EMPTY_FC)
      upsertSource(map, IDS.SNAP_SRC,     EMPTY_FC)
      return
    }

    let allSegments  = []
    let allCorridors = []
    let allWpDots    = []
    let boundsCoords = []

    // Buffer sizes per corridor type (degrees ~= km at -65°lat)
    const buffers = { aggressive: 0.18, optimal: 0.28, cautious: 0.40 }

    Object.entries(result.routes).forEach(([type, { route }]) => {
      if (!route) return
      allSegments  = allSegments.concat(buildRouteSegments(route, type))

      const corridor = buildCorridorPolygon(route, type, buffers[type] ?? 0.28)
      if (corridor) allCorridors.push(corridor)

      if (type === 'optimal') {
        boundsCoords = route.geometry.coordinates
        allWpDots    = buildWaypointDots(route)
      }
    })

    upsertSource(map, IDS.ROUTE_SRC,    { type: 'FeatureCollection', features: allSegments })
    upsertSource(map, IDS.CORRIDOR_SRC, { type: 'FeatureCollection', features: allCorridors })
    upsertSource(map, IDS.WP_SRC,       { type: 'FeatureCollection', features: allWpDots })

    // Fit bounds to optimal route
    if (boundsCoords.length > 0) {
      const lons = boundsCoords.map(c => c[0])
      const lats = boundsCoords.map(c => c[1])
      map.fitBounds(
        [[Math.min(...lons) - 3, Math.min(...lats) - 2], [Math.max(...lons) + 3, Math.max(...lats) + 2]],
        { padding: 60, duration: 1400 }
      )
    }

    // Snap markers
    const optRoute = result.routes.optimal?.route
    if (optRoute?.properties?.snapped_start_coords) {
      snapFeatures.push({ type: 'Feature', properties: { kind: 'start' }, geometry: { type: 'Point', coordinates: optRoute.properties.snapped_start_coords } })
    }
    if (optRoute?.properties?.snapped_end_coords) {
      snapFeatures.push({ type: 'Feature', properties: { kind: 'anchor' }, geometry: { type: 'Point', coordinates: optRoute.properties.snapped_end_coords } })
    }

    upsertSource(map, IDS.SNAP_SRC, { type: 'FeatureCollection', features: snapFeatures })
  }, [result])

  // Custom click points
  useEffect(() => {
    const map = mapRef.current
    if (!map || !readyRef.current) return
    const features = customPoints.map((pt, idx) => ({
      type: 'Feature',
      properties: { kind: idx === 0 ? 'start' : 'end' },
      geometry: { type: 'Point', coordinates: pt }
    }))
    upsertSource(map, IDS.CUSTOM_SRC, { type: 'FeatureCollection', features })
  }, [customPoints])

  const LayerBtn = useCallback(({ active, onClick, children, title, color = 'cyan' }) => {
    const colors = {
      cyan:   active ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40' : 'text-slate-500',
      sky:    active ? 'bg-sky-500/20 text-sky-300 border-sky-500/40'   : 'text-slate-500',
      purple: active ? 'bg-purple-500/20 text-purple-300 border-purple-500/40' : 'text-slate-500',
      amber:  active ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'    : 'text-slate-500',
      violet: active ? 'bg-violet-500/20 text-violet-300 border-violet-500/40' : 'text-slate-500',
    }
    return (
      <button
        onClick={onClick}
        title={title}
        className={`flex items-center gap-1.5 px-2 py-1 rounded font-mono text-[10px] font-semibold tracking-wider border transition-all hover:text-slate-300 ${colors[color]} ${active ? 'border shadow-sm' : 'border-transparent'}`}
      >
        {children}
      </button>
    )
  }, [])

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />
      <div className="scanline-overlay absolute inset-0 pointer-events-none" />

      {/* Floating Layer Controls */}
      <div className="absolute top-3 left-3 z-10 flex items-center gap-1 p-1.5 rounded-xl bg-ocean-950/88 backdrop-blur-md border border-sonar-500/20 shadow-xl shadow-black/40">
        <div className="flex items-center gap-1 px-2 py-0.5 border-r border-slate-700/50 mr-0.5">
          <Layers size={12} className="text-sonar-400" />
          <span className="font-mono text-[10px] text-slate-400 font-bold uppercase tracking-wider">LAYERS</span>
        </div>

        <LayerBtn active={showIce} onClick={() => setShowIce(!showIce)} title="Sea Ice Concentration Grid" color="cyan">
          {showIce ? <Eye size={11} /> : <EyeOff size={11} />}
          <span>ICE GRID</span>
        </LayerBtn>

        <LayerBtn active={showCorridors} onClick={() => setShowCorridors(!showCorridors)} title="Safety Corridors" color="amber">
          <Navigation size={11} />
          <span>CORRIDORS</span>
        </LayerBtn>

        <LayerBtn active={showWind} onClick={() => setShowWind(!showWind)} title="Open-Meteo Wind Field" color="sky">
          <Wind size={11} />
          <span>WIND</span>
        </LayerBtn>

        <LayerBtn active={showDrift} onClick={() => setShowDrift(!showDrift)} title="Ekman Ice Drift Vectors" color="purple">
          <Compass size={11} />
          <span>DRIFT</span>
        </LayerBtn>

        <LayerBtn active={showWpDots} onClick={() => setShowWpDots(!showWpDots)} title="Waypoint Physics Dots" color="violet">
          <MapPin size={11} />
          <span>WAYPOINTS</span>
        </LayerBtn>
      </div>

      {/* Legend */}
      <div className="absolute bottom-10 left-3 z-10 flex flex-col gap-1 px-2.5 py-2 rounded-lg bg-ocean-950/85 backdrop-blur-md border border-sonar-500/15 shadow-lg text-[9px] font-mono">
        <span className="text-slate-500 tracking-widest uppercase mb-0.5">SIC Legend</span>
        {[
          ['Open Water', '#00d4ff', '0–15%'],
          ['Marginal Ice', '#fbbf24', '15–40%'],
          ['Pack Ice', '#f87171', '40–70%'],
          ['Fast Ice', 'rgba(255,255,255,0.9)', '70–100%'],
        ].map(([label, color, range]) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-sm flex-none" style={{ background: color, border: '1px solid rgba(255,255,255,0.15)' }} />
            <span className="text-slate-400">{label}</span>
            <span className="text-slate-600 ml-auto pl-2">{range}</span>
          </div>
        ))}
        <div className="h-px bg-slate-800/60 my-0.5" />
        <div className="flex items-center gap-2">
          <div className="w-5 h-px border-t border-dashed border-amber-400/70 flex-none" />
          <span className="text-slate-400">Safety Corridor</span>
        </div>
      </div>

      {/* Click hint */}
      <div className="absolute bottom-10 right-16 z-10 px-2.5 py-1.5 rounded-lg bg-ocean-950/80 backdrop-blur-sm border border-sonar-500/15 font-mono text-[9px] text-slate-500 tracking-wider">
        Click map to set waypoints · Click route dots for physics data
      </div>

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-ocean-950/65 backdrop-blur-sm z-20">
          <div className="flex flex-col items-center gap-5">
            <div className="relative w-20 h-20">
              <div className="absolute inset-0 rounded-full border-2 border-sonar-500/15" />
              <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-sonar-500 animate-spin" />
              <div className="absolute inset-2 rounded-full border border-radar-500/25 animate-ping" />
              <div className="absolute inset-5 rounded-full bg-sonar-500/10 animate-pulse" />
            </div>
            <div className="text-center">
              <span className="font-mono text-xs tracking-widest text-sonar-400 animate-pulse block">COMPUTING 3 ROUTE CORRIDORS…</span>
              <span className="font-mono text-[10px] text-slate-600 block mt-1">A* · Physics Engine · CRS Transform</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
