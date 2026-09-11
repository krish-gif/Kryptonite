/**
 * ExtentTrendChart — Recharts line chart showing:
 *   - Observed extent_history (solid sky-blue line)
 *   - 3 forecasted extent values appended as a dashed indigo continuation
 *
 * The last observed point is shared between both lines to create a
 * seamless visual join at the history/forecast boundary.
 *
 * Clicking anywhere on the card opens an expanded modal with a larger
 * chart and a per-lead-day forecast breakdown table.
 *
 * Props:
 *   extentHistory – [{ date, extent_km2 }] — last ~30 days observed
 *   predictions   – [{ lead_day, date, image_url }] — for date labelling;
 *                   extent values are synthetic (trend extrapolation) since
 *                   the model only outputs spatial maps, not scalar extents.
 *                   In a full pipeline, predict.py would compute these too.
 */

import { useState, useRef } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from 'recharts'
import { formatExtent } from '../utils/mapHelpers'
import ExtentTrendModal from './ExtentTrendModal'

// Extrapolate forecast extent values from the trend of the last 7 days
function extrapolateForecastExtents(history, predictions) {
  if (!history || history.length < 2 || !predictions) return []

  // Compute average daily change over last 7 data points
  const recent = history.slice(-7)
  const dailyChanges = recent.slice(1).map((d, i) => d.extent_km2 - recent[i].extent_km2)
  const avgChange = dailyChanges.reduce((a, b) => a + b, 0) / dailyChanges.length

  const last = history[history.length - 1]
  return predictions.map((p, i) => ({
    date: p.date,
    forecast_km2: Math.max(0, last.extent_km2 + avgChange * (i + 1)),
    isForecast: true,
  }))
}

const CustomTooltip = ({ active, payload, label }) => {
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

// Format date label on X-axis: "Jan 15" style
function formatDateLabel(dateStr) {
  try {
    const d = new Date(dateStr + 'T00:00:00')
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  } catch {
    return dateStr
  }
}

// Only show every N-th label to avoid crowding
function tickFormatter(value, index, total) {
  const step = Math.max(1, Math.floor(total / 6))
  return index % step === 0 ? formatDateLabel(value) : ''
}

// Expand icon SVG
function ExpandIcon() {
  return (
    <svg
      width="14" height="14"
      viewBox="0 0 14 14"
      fill="none"
      className="panel-card-expand-icon"
      aria-hidden="true"
    >
      <path
        d="M1.5 5V1.5H5M9 1.5H12.5V5M12.5 9V12.5H9M5 12.5H1.5V9"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export default function ExtentTrendChart({ extentHistory, predictions }) {
  const [modalOpen, setModalOpen] = useState(false)
  const cardRef = useRef(null)

  if (!extentHistory || extentHistory.length === 0) return null

  const forecastPoints = extrapolateForecastExtents(extentHistory, predictions)

  // Build a unified data array. The last observed point is the boundary.
  const lastObs = extentHistory[extentHistory.length - 1]
  const boundaryDate = lastObs.date

  // Observed data
  const observedData = extentHistory.map(d => ({
    date: d.date,
    observed_km2: d.extent_km2,
    forecast_km2: null,
  }))

  // Forecast data — include the last observed point so lines join seamlessly
  const forecastData = [
    { date: lastObs.date, observed_km2: null, forecast_km2: lastObs.extent_km2 },
    ...forecastPoints.map(f => ({
      date: f.date,
      observed_km2: null,
      forecast_km2: f.forecast_km2,
    })),
  ]

  // Merge by date
  const allDates = [
    ...observedData,
    ...forecastData.slice(1), // skip the repeated boundary date
  ]
  // Inject the boundary point as carrying both values
  const merged = allDates.map(d => {
    if (d.date === boundaryDate && forecastData.length > 0) {
      return { ...d, forecast_km2: lastObs.extent_km2 }
    }
    return d
  })

  // Compute Y-axis domain with a 10% margin
  const allValues = merged
    .flatMap(d => [d.observed_km2, d.forecast_km2])
    .filter(v => v != null)
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)
  const pad    = (maxVal - minVal) * 0.12 || 1000
  const yDomain = [
    Math.max(0, Math.floor((minVal - pad) / 1000) * 1000),
    Math.ceil((maxVal + pad) / 1000) * 1000,
  ]

  const totalPoints = merged.length

  const handleCardClick = () => setModalOpen(true)
  const handleModalClose = () => {
    setModalOpen(false)
    // Focus will return to the card via the modal's cleanup effect
  }

  return (
    <>
      <div
        ref={cardRef}
        className="panel-card panel-card-clickable"
        onClick={handleCardClick}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleCardClick() } }}
        role="button"
        tabIndex={0}
        aria-label="Extent Trend — click to expand"
        aria-haspopup="dialog"
      >
        <div className="panel-card-header">
          <div className="panel-card-label">Extent Trend</div>
          <ExpandIcon />
        </div>

        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={merged}
              margin={{ top: 4, right: 4, left: -18, bottom: 0 }}
            >
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickFormatter={(v, i) => tickFormatter(v, i, totalPoints)}
                tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
              />
              <YAxis
                domain={yDomain}
                tickLine={false}
                axisLine={false}
                tickFormatter={v => formatExtent(v)}
                tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
                width={52}
              />
              <Tooltip content={<CustomTooltip />} />

              {/* Vertical reference line at the history/forecast boundary */}
              <ReferenceLine
                x={boundaryDate}
                stroke="var(--border-subtle)"
                strokeDasharray="3 3"
              />

              {/* Observed history — solid sky-blue */}
              <Line
                type="monotone"
                dataKey="observed_km2"
                stroke="var(--accent-blue)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: 'var(--accent-blue)' }}
                connectNulls={false}
                isAnimationActive={true}
                animationDuration={800}
              />

              {/* Forecast continuation — dashed indigo */}
              <Line
                type="monotone"
                dataKey="forecast_km2"
                stroke="var(--accent-indigo)"
                strokeWidth={2}
                strokeDasharray="5 3"
                dot={(props) => {
                  // Draw a distinct dot only at forecast (non-boundary) points
                  const { cx, cy, payload } = props
                  if (!payload.forecast_km2 || payload.date === boundaryDate) return null
                  return (
                    <circle
                      key={payload.date}
                      cx={cx} cy={cy}
                      r={3}
                      fill="var(--accent-indigo)"
                      stroke="var(--bg-elevated)"
                      strokeWidth={1.5}
                    />
                  )
                }}
                activeDot={{ r: 4, fill: 'var(--accent-indigo)' }}
                connectNulls={false}
                isAnimationActive={true}
                animationDuration={800}
                animationBegin={200}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Legend */}
        <div className="chart-legend">
          <div className="chart-legend-item">
            <div className="chart-legend-line solid" />
            <span>Observed</span>
          </div>
          <div className="chart-legend-item">
            <div className="chart-legend-line dashed" />
            <span>Forecast</span>
          </div>
        </div>
      </div>

      {/* Expanded modal — rendered via portal-like layering */}
      {modalOpen && (
        <ExtentTrendModal
          merged={merged}
          forecastPoints={forecastPoints}
          boundaryDate={boundaryDate}
          yDomain={yDomain}
          onClose={handleModalClose}
        />
      )}
    </>
  )
}
