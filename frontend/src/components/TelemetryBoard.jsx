/**
 * TelemetryBoard — Route calculation readout panel.
 *
 * Displays: distance, fuel, risk index, ETA, status, and a mock-data notice.
 * Animates in when `telemetry` becomes non-null.
 */

// ── Risk bar colour ─────────────────────────────────────────────
function riskColour(v) {
  if (v >= 0.6) return 'bg-hazard-500'
  if (v >= 0.35) return 'bg-caution-500'
  return 'bg-radar-500'
}

// ── Telemetry metric row ────────────────────────────────────────
function MetricRow({ label, value, unit, highlight = false }) {
  return (
    <div className="flex items-end justify-between py-2
                    border-b border-ocean-600/40 last:border-0">
      <span className="telem-label">{label}</span>
      <span className={`font-mono font-semibold tabular-nums ${highlight ? 'text-sonar-400 text-glow-sonar' : 'text-slate-200'}`}>
        {value}
        {unit && <span className="text-xs text-slate-500 font-normal ml-1">{unit}</span>}
      </span>
    </div>
  )
}

// ── Status badge ────────────────────────────────────────────────
function StatusBadge({ status, level }) {
  const cls =
    level === 'clear'   ? 'status-clear' :
    level === 'caution' ? 'status-caution' :
    'status-reroute'

  const dot =
    level === 'clear'   ? 'bg-radar-500 animate-pulse-radar' :
    level === 'caution' ? 'bg-caution-500' :
    'bg-hazard-500 animate-blink'

  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded font-mono text-xs font-semibold
                     tracking-widest uppercase ${cls}`}>
      <span className={`w-2 h-2 rounded-full ${dot}`} />
      {status}
    </div>
  )
}

export default function TelemetryBoard({ telemetry, error }) {
  if (!telemetry) {
    return (
      <div className="panel-card p-4">
        <div className="section-header">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
          Telemetry Readout
        </div>
        <div className="flex flex-col items-center justify-center py-6 gap-2">
          <div className="w-8 h-8 rounded-full border border-sonar-500/20 flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-sonar-500/40" />
          </div>
          <p className="font-mono text-xs text-slate-600 text-center tracking-wider">
            AWAITING ROUTE<br/>CALCULATION
          </p>
        </div>
      </div>
    )
  }

  const riskPct = Math.round(telemetry.risk_index * 100)

  return (
    <div className="panel-card p-4 animate-fade-in">
      <div className="section-header">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
        </svg>
        Telemetry Readout
      </div>

      {/* Status badge */}
      <div className="mb-3">
        <StatusBadge status={telemetry.status} level={telemetry.status_level} />
      </div>

      {/* Metrics */}
      <div className="mb-3">
        <MetricRow
          label="Total Distance"
          value={telemetry.distance_nm.toLocaleString()}
          unit="NM"
          highlight
        />
        <MetricRow
          label="Est. Fuel Consumption"
          value={telemetry.fuel_tons.toFixed(1)}
          unit="t"
        />
        <MetricRow
          label="ETA"
          value={`${Math.floor(telemetry.eta_hours / 24)}d ${telemetry.eta_hours % 24}h`}
        />
        <MetricRow
          label="Waypoints"
          value={telemetry.waypoints}
        />
      </div>

      {/* Risk index bar */}
      <div>
        <div className="flex justify-between mb-1">
          <span className="telem-label">Risk Index</span>
          <span className={`font-mono text-xs font-semibold ${
            riskPct >= 60 ? 'text-hazard-400' :
            riskPct >= 35 ? 'text-caution-500' :
            'text-radar-400'
          }`}>
            {riskPct}%
          </span>
        </div>
        <div className="risk-bar-track">
          <div
            className={`risk-bar-fill ${riskColour(telemetry.risk_index)}`}
            style={{ width: `${riskPct}%` }}
          />
        </div>
      </div>

      {/* Mock data notice */}
      {error?.isMock && (
        <div className="mt-3 px-2 py-1.5 rounded bg-caution-500/10 border border-caution-500/20">
          <p className="font-mono text-[10px] text-caution-500 leading-relaxed">
            ⚠ SIMULATED — backend offline
          </p>
        </div>
      )}
    </div>
  )
}
