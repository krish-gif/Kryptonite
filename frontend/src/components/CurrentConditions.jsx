/**
 * CurrentConditions — displays today's sea-ice extent in km²
 * and the day-over-day % change with an up/down indicator.
 *
 * Props:
 *   extentHistory – extent_history array from forecast.json
 */

import { formatExtent, computeDeltaPct } from '../utils/mapHelpers'

const DIRECTION_ICONS = {
  up:      '▲',
  down:    '▼',
  neutral: '—',
}

export default function CurrentConditions({ extentHistory }) {
  if (!extentHistory || extentHistory.length === 0) return null

  const latest = extentHistory[extentHistory.length - 1]
  const delta  = computeDeltaPct(extentHistory)

  return (
    <div className="panel-card">
      <div className="panel-card-label">Current Conditions</div>

      <div className="conditions-extent" aria-label="Sea-ice extent">
        {formatExtent(latest.extent_km2)}
        <span className="conditions-unit">km²</span>
      </div>

      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
        as of {latest.date}
      </div>

      {delta && (
        <div
          className={`conditions-delta ${delta.direction}`}
          aria-label={`Day-over-day change: ${delta.direction === 'up' ? '+' : delta.direction === 'down' ? '-' : ''}${delta.pct.toFixed(2)}%`}
        >
          <span aria-hidden="true">{DIRECTION_ICONS[delta.direction]}</span>
          {delta.pct.toFixed(2)}%
          <span style={{ fontWeight: 400, marginLeft: '2px' }}>vs yesterday</span>
        </div>
      )}
    </div>
  )
}
