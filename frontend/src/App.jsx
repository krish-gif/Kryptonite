/**
 * App — root layout and state orchestration.
 *
 * Responsibilities:
 *   - Fetches forecast data via useForecast()
 *   - Manages selectedDay state (0 = today, 1-3 = lead days)
 *   - Derives the active image URL from selectedDay + predictions
 *   - Renders the full-bleed layout: header / banner / map / side panel
 */

import { useState } from 'react'
import { useForecast } from './hooks/useForecast'
import MapView    from './components/MapView'
import SidePanel  from './components/SidePanel'
import StaleBanner from './components/StaleBanner'

// ── Loading skeleton ───────────────────────────────────────────
function LoadingOverlay() {
  return (
    <div className="loading-overlay" role="status" aria-label="Loading forecast data">
      <div className="loading-spinner" aria-hidden="true" />
      <span className="loading-text">Loading Arctic forecast…</span>
    </div>
  )
}

// ── Snowflake icon for the logo ────────────────────────────────
function SnowflakeIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 2v20M2 12h20M5.636 5.636l12.728 12.728M18.364 5.636L5.636 18.364"
        stroke="white" strokeWidth="2" strokeLinecap="round"
      />
    </svg>
  )
}

// ── Derive the active image URL for the current lead day ───────
function getActiveImageUrl(selectedDay, predictions) {
  if (!predictions || selectedDay === 0) return null
  const pred = predictions.find(p => p.lead_day === selectedDay)
  return pred ? pred.image_url : null
}

// ── App ────────────────────────────────────────────────────────
export default function App() {
  const { data, loading, error, isStale } = useForecast()
  const [selectedDay, setSelectedDay] = useState(1)

  const activeImageUrl = getActiveImageUrl(selectedDay, data?.predictions)

  return (
    <div className="app">
      {/* ── Header ─────────────────────────────────────────── */}
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-icon" aria-hidden="true">
            <SnowflakeIcon />
          </div>
          <div>
            <h1 className="app-title">Arctic Ice Watch</h1>
            <div className="app-subtitle">Sea-Ice Concentration Forecast</div>
          </div>
        </div>

        <div className="app-header-spacer" />

        {data?.forecast_date && (
          <div style={{
            fontSize: '11px',
            color: 'var(--text-muted)',
            textAlign: 'right',
            lineHeight: 1.4,
          }}>
            <div style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>
              Forecast for
            </div>
            {data.forecast_date}
          </div>
        )}
      </header>

      {/* ── Stale / error banner (spans full width) ────────── */}
      <StaleBanner error={error} isStale={isStale} />

      {/* ── Main content ───────────────────────────────────── */}
      <main className="app-main">
        {/* Map column — always rendered so MapLibre can mount */}
        <div className="map-column" style={{ position: 'relative' }}>
          {loading && <LoadingOverlay />}
          <MapView
            grid={data?.grid}
            activeImageUrl={activeImageUrl}
            selectedDay={selectedDay}
            predictions={data?.predictions}
            onChange={setSelectedDay}
          />
        </div>

        {/* Side panel — metrics & chart */}
        <SidePanel
          data={data}
          isStale={isStale}
          error={error}
        />
      </main>
    </div>
  )
}
