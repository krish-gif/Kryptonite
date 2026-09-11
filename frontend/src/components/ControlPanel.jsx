/**
 * ControlPanel — Captain's Command Sidebar (right panel).
 *
 * Contains:
 *   • Route Configuration (departure / destination inputs)
 *   • Forecast Window selector
 *   • Tactical Safety/Fuel slider
 *   • CALCULATE OPTIMAL ROUTE action button
 */

import { useState } from 'react'

const PRESETS = {
  departure: [
    { label: 'Cape Town, ZA',     lat: -33.9, lon: 18.4  },
    { label: 'Port Elizabeth, ZA', lat: -33.9, lon: 25.5  },
    { label: 'Hobart, AU',         lat: -42.9, lon: 147.3 },
  ],
  destination: [
    { label: 'Maitri Station',    lat: -70.8, lon: 11.7  },
    { label: 'Bharati Station',   lat: -69.4, lon: 76.2  },
    { label: 'McMurdo Station',   lat: -77.8, lon: 166.7 },
  ],
}

const FORECAST_OPTIONS = [
  { label: 'CURRENT',  value: 0  },
  { label: '+24H',     value: 24 },
  { label: '+48H',     value: 48 },
  { label: '+72H',     value: 72 },
]

// ── Small icon helpers ─────────────────────────────────────────
function IconAnchor() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="5" r="3"/><line x1="12" y1="8" x2="12" y2="21"/>
      <path d="M5 15l7 6 7-6"/>
    </svg>
  )
}
function IconTarget() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/>
      <line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/>
      <line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/>
    </svg>
  )
}
function IconClock() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
    </svg>
  )
}
function IconSliders() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/>
      <line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/>
      <line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/>
      <line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/>
      <line x1="17" y1="16" x2="23" y2="16"/>
    </svg>
  )
}

export default function ControlPanel({ onCalculate, loading }) {
  const [departure,       setDeparture]       = useState(PRESETS.departure[0])
  const [destination,     setDestination]     = useState(PRESETS.destination[0])
  const [forecastWindow,  setForecastWindow]  = useState(24)
  const [safetyWeight,    setSafetyWeight]    = useState(50)

  const handleSubmit = (e) => {
    e.preventDefault()
    onCalculate({
      start:          { lat: departure.lat,    lon: departure.lon,    label: departure.label },
      end:            { lat: destination.lat,  lon: destination.lon,  label: destination.label },
      forecastWindow,
      safetyWeight,
    })
  }

  // Safety weight label
  const weightLabel =
    safetyWeight < 30 ? 'FUEL OPTIMAL'
    : safetyWeight > 70 ? 'MAX SAFETY'
    : 'BALANCED'

  return (
    <form
      onSubmit={handleSubmit}
      className="flex flex-col gap-4 h-full overflow-y-auto"
      id="kns-control-panel"
    >

      {/* ── Route Configuration ──────────────────────────── */}
      <div className="panel-card p-4">
        <div className="section-header">
          <IconAnchor />
          Route Configuration
        </div>

        {/* Departure */}
        <div className="mb-3">
          <label className="telem-label block mb-1.5">Departure Point</label>
          <select
            className="nav-input"
            value={departure.label}
            onChange={(e) => {
              const p = PRESETS.departure.find(x => x.label === e.target.value)
              if (p) setDeparture(p)
            }}
          >
            {PRESETS.departure.map(p => (
              <option key={p.label} value={p.label}>{p.label}</option>
            ))}
          </select>
          <div className="mt-1 font-mono text-[10px] text-slate-600">
            {departure.lat.toFixed(2)}°S · {departure.lon.toFixed(2)}°E
          </div>
        </div>

        {/* Destination */}
        <div>
          <label className="telem-label block mb-1.5">Destination</label>
          <select
            className="nav-input"
            value={destination.label}
            onChange={(e) => {
              const p = PRESETS.destination.find(x => x.label === e.target.value)
              if (p) setDestination(p)
            }}
          >
            {PRESETS.destination.map(p => (
              <option key={p.label} value={p.label}>{p.label}</option>
            ))}
          </select>
          <div className="mt-1 font-mono text-[10px] text-slate-600">
            {Math.abs(destination.lat).toFixed(2)}°S · {destination.lon.toFixed(2)}°E
          </div>
        </div>
      </div>

      {/* ── Forecast Window ──────────────────────────────── */}
      <div className="panel-card p-4">
        <div className="section-header">
          <IconClock />
          ConvLSTM Forecast Window
        </div>
        <div className="grid grid-cols-4 gap-1.5">
          {FORECAST_OPTIONS.map(opt => (
            <button
              key={opt.value}
              type="button"
              onClick={() => setForecastWindow(opt.value)}
              className={`py-2 rounded font-mono text-xs font-semibold tracking-wider transition-all duration-200
                ${forecastWindow === opt.value
                  ? 'bg-sonar-500/20 border border-sonar-500/60 text-sonar-400 text-glow-sonar'
                  : 'bg-ocean-900 border border-ocean-600 text-slate-500 hover:border-sonar-500/30 hover:text-slate-400'
                }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── Tactical Slider ──────────────────────────────── */}
      <div className="panel-card p-4">
        <div className="section-header">
          <IconSliders />
          Algorithm Weight
        </div>

        <div className="flex justify-between items-center mb-2">
          <span className="font-mono text-[10px] text-hazard-400">⚡ FUEL EFF.</span>
          <span className={`font-mono text-xs font-semibold px-2 py-0.5 rounded tracking-wider
            ${safetyWeight > 70 ? 'text-radar-400 bg-radar-500/10' :
              safetyWeight < 30 ? 'text-hazard-400 bg-hazard-500/10' :
              'text-caution-400 bg-caution-500/10'}`}>
            {weightLabel}
          </span>
          <span className="font-mono text-[10px] text-radar-400">🛡 MAX SAFE</span>
        </div>

        <input
          type="range"
          min={0}
          max={100}
          value={safetyWeight}
          onChange={(e) => setSafetyWeight(Number(e.target.value))}
          className="tactical-slider"
          aria-label="Safety vs Fuel Efficiency weight"
        />

        <div className="mt-2 flex justify-center">
          <span className="font-mono text-xs text-slate-500">
            Safety weight: <span className="text-sonar-400">{safetyWeight}%</span>
          </span>
        </div>
      </div>

      {/* ── Action Button ─────────────────────────────────── */}
      <button
        type="submit"
        disabled={loading}
        className="btn-calculate"
        id="kns-calculate-btn"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeDasharray="31.4" strokeDashoffset="10"/>
            </svg>
            COMPUTING A* PATH…
          </span>
        ) : (
          '⌖ CALCULATE OPTIMAL ROUTE'
        )}
      </button>

      {/* ── Waypoints summary ─────────────────────────────── */}
      <div className="font-mono text-[10px] text-slate-600 text-center">
        <span className="text-sonar-500/60">{departure.label}</span>
        <span className="mx-2">→</span>
        <span className="text-radar-500/60">{destination.label}</span>
      </div>
    </form>
  )
}
