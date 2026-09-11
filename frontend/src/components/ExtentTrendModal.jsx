/**
 * ExtentTrendModal — enlarged extent chart shown in a centered modal.
 *
 * Contains:
 *   1. A bigger version of the same Recharts LineChart (solid observed,
 *      dashed forecast, vertical reference line at the boundary).
 *   2. A forecast breakdown table below the chart: one row per lead day
 *      with date and predicted extent as a readable number.
 *
 * Props:
 *   merged         – unified chart data array (same as in-panel chart)
 *   forecastPoints – extrapolated forecast data [{date, forecast_km2}]
 *   boundaryDate   – the date string where observed → forecast
 *   yDomain        – [min, max] for the Y axis
 *   onClose        – callback to close the modal
 */

import { useEffect, useRef } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts'
import { formatExtent } from '../utils/mapHelpers'

// ── Custom tooltip (shared style with the inline chart) ──────
const ModalTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const entry = payload.find(p => p.value != null)
  if (!entry) return null

  return (
    <div className="ice-tooltip">
      <div className="ice-tooltip-label">{label}</div>
      <div className="ice-tooltip-value" style={{ color: entry.color }}>
        {formatExtent(entry.value)} km²
      </div>
      {payload[0]?.payload?.isForecast && (
        <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
          Forecast (extrapolated)
        </div>
      )}
    </div>
  )
}

// ── Format date for display ──────────────────────────────────
function formatDate(dateStr) {
  try {
    const d = new Date(dateStr + 'T00:00:00')
    return d.toLocaleDateString('en-US', {
      weekday: 'short',
      month:   'short',
      day:     'numeric',
    })
  } catch {
    return dateStr
  }
}

function formatDateShort(dateStr) {
  try {
    const d = new Date(dateStr + 'T00:00:00')
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  } catch {
    return dateStr
  }
}

export default function ExtentTrendModal({
  merged,
  forecastPoints,
  boundaryDate,
  yDomain,
  onClose,
}) {
  const closeButtonRef = useRef(null)
  const previousFocusRef = useRef(null)

  // ── Trap focus management + Escape key ──────────────────────
  useEffect(() => {
    previousFocusRef.current = document.activeElement
    // Focus the close button on open
    requestAnimationFrame(() => closeButtonRef.current?.focus())

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
      }
    }
    document.addEventListener('keydown', handleKeyDown)

    // Lock body scroll while modal is open
    document.body.style.overflow = 'hidden'

    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.body.style.overflow = ''
      // Return focus to the card that opened this
      previousFocusRef.current?.focus()
    }
  }, [onClose])

  // Prevent clicks inside the modal from closing it
  const handleContentClick = (e) => e.stopPropagation()

  const totalPoints = merged.length

  return (
    <div
      className="modal-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Expanded extent trend chart"
    >
      <div className="modal-container" onClick={handleContentClick}>
        {/* ── Header ──────────────────────────────────────── */}
        <div className="modal-header">
          <div>
            <h2 className="modal-title">Extent Trend</h2>
            <p className="modal-subtitle">
              30-day observed history with 3-day forecast
            </p>
          </div>
          <button
            ref={closeButtonRef}
            className="modal-close-btn"
            onClick={onClose}
            aria-label="Close modal"
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <path d="M4.5 4.5L13.5 13.5M13.5 4.5L4.5 13.5"
                stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* ── Enlarged chart ──────────────────────────────── */}
        <div className="modal-chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={merged}
              margin={{ top: 8, right: 16, left: 8, bottom: 4 }}
            >
              <CartesianGrid
                stroke="var(--border-subtle)"
                strokeDasharray="3 3"
                vertical={false}
              />
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickFormatter={(v, i) => {
                  const step = Math.max(1, Math.floor(totalPoints / 10))
                  return i % step === 0 ? formatDateShort(v) : ''
                }}
                tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
              />
              <YAxis
                domain={yDomain}
                tickLine={false}
                axisLine={false}
                tickFormatter={v => formatExtent(v)}
                tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
                width={58}
              />
              <Tooltip content={<ModalTooltip />} />

              <ReferenceLine
                x={boundaryDate}
                stroke="var(--accent-indigo-dim)"
                strokeWidth={1}
                strokeDasharray="4 4"
                label={{
                  value: 'Forecast →',
                  position: 'insideTopRight',
                  fill: 'var(--text-muted)',
                  fontSize: 10,
                  offset: 8,
                }}
              />

              {/* Observed — solid sky-blue */}
              <Line
                type="monotone"
                dataKey="observed_km2"
                stroke="var(--accent-blue)"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 5, fill: 'var(--accent-blue)', strokeWidth: 0 }}
                connectNulls={false}
                isAnimationActive={true}
                animationDuration={600}
              />

              {/* Forecast — dashed indigo */}
              <Line
                type="monotone"
                dataKey="forecast_km2"
                stroke="var(--accent-indigo)"
                strokeWidth={2.5}
                strokeDasharray="6 4"
                dot={(props) => {
                  const { cx, cy, payload } = props
                  if (!payload.forecast_km2 || payload.date === boundaryDate) return null
                  return (
                    <circle
                      key={payload.date}
                      cx={cx} cy={cy}
                      r={4}
                      fill="var(--accent-indigo)"
                      stroke="var(--bg-elevated)"
                      strokeWidth={2}
                    />
                  )
                }}
                activeDot={{ r: 5, fill: 'var(--accent-indigo)', strokeWidth: 0 }}
                connectNulls={false}
                isAnimationActive={true}
                animationDuration={600}
                animationBegin={150}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* ── Legend ───────────────────────────────────────── */}
        <div className="chart-legend" style={{ marginBottom: 'var(--space-5)' }}>
          <div className="chart-legend-item">
            <div className="chart-legend-line solid" />
            <span>Observed</span>
          </div>
          <div className="chart-legend-item">
            <div className="chart-legend-line dashed" />
            <span>Forecast</span>
          </div>
        </div>

        {/* ── Forecast breakdown table ────────────────────── */}
        <div className="modal-forecast-breakdown">
          <div className="modal-section-label">Forecast Breakdown</div>
          <div className="modal-forecast-table">
            {forecastPoints.map((fp, i) => (
              <div className="modal-forecast-row" key={fp.date}>
                <div className="modal-forecast-lead">
                  <span className="modal-forecast-dot" aria-hidden="true" />
                  Day +{i + 1}
                </div>
                <div className="modal-forecast-date">
                  {formatDate(fp.date)}
                </div>
                <div className="modal-forecast-value">
                  {fp.forecast_km2.toLocaleString(undefined, {
                    maximumFractionDigits: 0,
                  })}
                  <span className="modal-forecast-unit"> km²</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
