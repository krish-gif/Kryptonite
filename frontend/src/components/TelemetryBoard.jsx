import { useState } from 'react'
import {
  ChevronDown, ChevronUp, Navigation, AlertTriangle, ShieldAlert,
  Wind, Waves, ArrowUp, Compass, Target, Crosshair, Zap, Ship,
  TrendingUp, TrendingDown, Activity, Radio
} from 'lucide-react'

const STATUS_STYLES = {
  clear:   { border: 'border-radar-500/30',   text: 'text-radar-400',   bg: 'bg-radar-500/5'   },
  caution: { border: 'border-caution-500/30', text: 'text-caution-400', bg: 'bg-caution-500/5' },
  reroute: { border: 'border-hazard-500/30',  text: 'text-hazard-400',  bg: 'bg-hazard-500/5'  },
  noroute: { border: 'border-slate-600/30',   text: 'text-slate-500',   bg: 'bg-slate-800/5'   },
}

function Metric({ label, value, unit, accent = false, danger = false, good = false, className = '' }) {
  const textColor = danger ? 'text-rose-400' : good ? 'text-radar-400' : accent ? 'text-sonar-400' : 'text-slate-300'
  return (
    <div className={`flex flex-col gap-0.5 ${className}`}>
      <span className="font-mono text-[9px] tracking-widest text-slate-600 uppercase">{label}</span>
      <span className={`font-mono text-sm font-semibold tabular-nums ${textColor}`}>
        {value ?? '—'}
        {unit && <span className="text-[10px] text-slate-600 ml-1">{unit}</span>}
      </span>
    </div>
  )
}

function MiniBar({ value, max = 1, color = '#00d4ff', label }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  return (
    <div className="flex flex-col gap-0.5">
      {label && <span className="font-mono text-[9px] text-slate-600 tracking-widest uppercase">{label}</span>}
      <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}

function WaypointRow({ wp, isSelected, onSelect }) {
  const sic = wp.ice_concentration !== undefined ? wp.ice_concentration : (wp.sic ?? 0)
  const Icon = sic > 0.40
    ? () => <AlertTriangle size={11} className="text-hazard-400" />
    : sic > 0.15
      ? () => <ShieldAlert size={11} className="text-caution-400" />
      : () => <Navigation size={11} className="text-radar-400" />

  const barColor = sic > 0.40 ? '#f87171' : sic > 0.15 ? '#fbbf24' : '#00d4ff'
  const vEff = wp.effective_ship_speed ?? wp.speed_knots ?? 0

  return (
    <div
      onClick={() => onSelect(wp)}
      className={`flex items-start gap-2 py-2 px-2 border-b border-slate-800/40 last:border-0 rounded cursor-pointer transition-all ${
        isSelected ? 'bg-sonar-500/12 border-sonar-500/30 shadow-sm' : 'hover:bg-white/[0.03]'
      }`}
    >
      <div className="w-4 flex-none flex justify-center mt-0.5"><Icon /></div>
      <div className="flex-1 min-w-0">
        <div className="flex justify-between items-baseline mb-0.5">
          <span className="font-mono text-[10px] text-slate-200 font-bold">T+{wp.eta_hours ?? 0}h</span>
          <span className="font-mono text-[10px] text-violet-300 font-semibold">{vEff.toFixed(1)} kts</span>
        </div>
        <div className="flex justify-between items-baseline mb-1.5">
          <span className="font-mono text-[9px] text-slate-400 truncate">{wp.hazard_type ?? wp.navigation_mode?.replace(/_/g, ' ')}</span>
          <span className="font-mono text-[9px] text-slate-600">
            {wp.coordinates ? `${wp.coordinates[1].toFixed(2)}°, ${wp.coordinates[0].toFixed(2)}°` : ''}
          </span>
        </div>
        {/* Physics mini-row */}
        <div className="flex items-center gap-3 mb-1.5">
          {wp.wind_speed_knots > 0 && (
            <span className="font-mono text-[9px] text-sky-400/90 flex items-center gap-0.5">
              <Wind size={8} />
              {(wp.wind_speed_knots ?? 0).toFixed(1)}kt
              <ArrowUp size={8} className="text-sky-300" style={{ transform: `rotate(${(wp.wind_direction ?? 0) + 180}deg)` }} />
            </span>
          )}
          {(wp.ocean_current_knots ?? 0) > 0.05 && (
            <span className="font-mono text-[9px] text-teal-400/80 flex items-center gap-0.5">
              <Waves size={8} />
              {wp.ocean_current_knots.toFixed(2)}kt
            </span>
          )}
        </div>
        {/* SIC bar */}
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full rounded-full" style={{ width: `${Math.max(2, sic * 100)}%`, background: barColor }} />
          </div>
          <span className={`font-mono text-[9px] w-10 text-right tabular-nums`} style={{ color: barColor }}>
            {(sic * 100).toFixed(1)}%
          </span>
        </div>
      </div>
    </div>
  )
}

export default function TelemetryBoard({
  telemetry,
  error,
  activeRoute = 'optimal',
  onSelectRoute,
  selectedWaypoint = null,
  onSelectWaypoint,
}) {
  const [expanded, setExpanded] = useState(false)

  if (error && !telemetry) {
    return (
      <div className="panel-card p-3 border border-hazard-500/30 bg-hazard-500/5">
        <div className="section-header mb-2 text-hazard-400">⚠ ROUTE ENGINE ERROR</div>
        <p className="font-mono text-[10px] text-slate-400 leading-relaxed">{error.message}</p>
        <p className="font-mono text-[10px] text-slate-500 mt-2">Check the FastAPI backend at port 8000.</p>
      </div>
    )
  }

  if (!telemetry) {
    return (
      <div className="panel-card p-4 text-center flex flex-col items-center gap-3">
        <Radio size={20} className="text-sonar-500/40 animate-pulse" />
        <div>
          <div className="font-mono text-[10px] text-slate-600 tracking-widest uppercase">Awaiting Route</div>
          <div className="font-mono text-[9px] text-slate-700 mt-1">Configure parameters and calculate</div>
        </div>
      </div>
    )
  }

  const st = STATUS_STYLES[telemetry.status_level] ?? STATUS_STYLES.caution
  const wps = telemetry.waypoints || []
  const rs  = telemetry.route_summary ?? {}
  const es  = telemetry.environmental_summary ?? null

  const totalIceKm = (rs.total_marginal_ice_km ?? 0) + (rs.total_pack_ice_km ?? 0)
  const totalKm    = telemetry.distance_km ?? 0
  const icePct     = totalKm > 0 ? ((totalIceKm / totalKm) * 100).toFixed(0) : 0

  const selectedWpObj = selectedWaypoint
    ? wps.find(w => w.coordinates && Math.abs(w.coordinates[0] - selectedWaypoint[0]) < 0.001 && Math.abs(w.coordinates[1] - selectedWaypoint[1]) < 0.001)
    : null

  return (
    <div className="flex flex-col gap-2">

      {/* Route Corridor Tabs */}
      {onSelectRoute && (
        <div className="grid grid-cols-3 gap-1 p-1 bg-ocean-950/80 rounded-lg border border-sonar-500/15">
          {[
            { id: 'cautious',   label: 'CAUTIOUS', color: 'text-emerald-400', border: 'border-emerald-500/50', activeBg: 'bg-emerald-500/10' },
            { id: 'optimal',    label: 'OPTIMAL',  color: 'text-amber-400',   border: 'border-amber-500/50',   activeBg: 'bg-amber-500/10'   },
            { id: 'aggressive', label: 'DIRECT',   color: 'text-rose-400',    border: 'border-rose-500/50',    activeBg: 'bg-rose-500/10'    },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => onSelectRoute(tab.id)}
              className={`py-1.5 rounded font-mono text-[10px] font-bold tracking-wider transition-all ${
                activeRoute === tab.id
                  ? `${tab.activeBg} ${tab.color} border ${tab.border} shadow-sm`
                  : 'text-slate-600 hover:text-slate-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      )}

      {/* Status Banner */}
      <div className={`panel-card p-3 border ${st.border} ${st.bg}`}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className={`w-1.5 h-1.5 rounded-full flex-none ${telemetry.status_level === 'clear' ? 'bg-radar-500 animate-pulse-radar' : 'bg-caution-500 animate-pulse'}`} />
            <span className={`font-mono text-[10px] font-bold tracking-wider ${st.text}`}>{telemetry.status}</span>
          </div>
          <div className="flex gap-1">
            <span className={`font-mono text-[9px] px-1.5 py-0.5 rounded border ${st.border} ${st.text}`}>
              {telemetry.risk_status}
            </span>
          </div>
        </div>
        {/* Route composition bar */}
        <div className="mt-1">
          <div className="flex text-[9px] font-mono justify-between text-slate-600 mb-1">
            <span>ICE EXPOSURE</span>
            <span className={icePct > 30 ? 'text-caution-400' : 'text-slate-500'}>{icePct}%</span>
          </div>
          <div className="h-1.5 bg-slate-800/80 rounded-full overflow-hidden flex">
            {totalKm > 0 && <>
              <div className="h-full bg-cyan-500/70" style={{ width: `${((rs.total_open_water_km ?? 0) / totalKm * 100).toFixed(0)}%` }} />
              <div className="h-full bg-amber-400/70" style={{ width: `${((rs.total_marginal_ice_km ?? 0) / totalKm * 100).toFixed(0)}%` }} />
              <div className="h-full bg-rose-400/70" style={{ width: `${((rs.total_pack_ice_km ?? 0) / totalKm * 100).toFixed(0)}%` }} />
            </>}
          </div>
          <div className="flex gap-3 mt-1 text-[8px] font-mono text-slate-600">
            <span className="text-cyan-500/70">■ Open</span>
            <span className="text-amber-400/70">■ Marginal</span>
            <span className="text-rose-400/70">■ Pack</span>
          </div>
        </div>
      </div>

      {/* Core Metrics */}
      <div className="panel-card p-3">
        <div className="section-header mb-2">
          <Ship size={11} />
          Route Metrics
        </div>

        {/* Snapped coordinates */}
        {telemetry.snapped_start && telemetry.snapped_end && (
          <div className="flex flex-col gap-0.5 mb-3 p-2 bg-slate-900/50 rounded border border-slate-800/60">
            <div className="flex justify-between items-center">
              <span className="font-mono text-[9px] text-radar-500/70 tracking-widest">▶ ENTRY</span>
              <span className="font-mono text-[9px] text-slate-300 tabular-nums">
                {telemetry.snapped_start[0].toFixed(4)}°, {telemetry.snapped_start[1].toFixed(4)}°
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="font-mono text-[9px] text-caution-500/70 tracking-widest">⚓ ARRIVAL</span>
              <span className="font-mono text-[9px] text-slate-300 tabular-nums">
                {telemetry.snapped_end[0].toFixed(4)}°, {telemetry.snapped_end[1].toFixed(4)}°
              </span>
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3 mb-3">
          <Metric label="Total Distance" value={totalKm?.toLocaleString()} unit="km" accent />
          <Metric label="Distance NM" value={telemetry.distance_nm?.toLocaleString()} unit="NM" />
          <Metric label="Est. ETA" value={telemetry.eta_hours} unit="hrs" />
          <Metric label="Waypoints" value={telemetry.waypoints_count} />
        </div>

        {rs && Object.keys(rs).length > 0 && (
          <>
            <div className="h-px bg-slate-800/60 my-2" />
            <div className="grid grid-cols-2 gap-3 mb-2">
              <Metric label="Open Water" value={(rs.total_open_water_km ?? 0).toFixed(0)} unit="km" good />
              <Metric label="Marginal Ice" value={(rs.total_marginal_ice_km ?? 0).toFixed(0)} unit="km" />
              <Metric label="Pack Ice" value={(rs.total_pack_ice_km ?? 0).toFixed(0)} unit="km" danger={rs.total_pack_ice_km > 0} />
              <Metric label="Max SIC" value={((rs.max_ice_concentration_encountered ?? 0) * 100).toFixed(1)} unit="%" danger={(rs.max_ice_concentration_encountered ?? 0) > 0.4} />
            </div>
            {rs.max_ice_concentration_encountered > 0 && (
              <MiniBar
                value={rs.max_ice_concentration_encountered}
                max={1}
                color={rs.max_ice_concentration_encountered > 0.4 ? '#f87171' : rs.max_ice_concentration_encountered > 0.15 ? '#fbbf24' : '#00d4ff'}
                label="Max Ice Concentration"
              />
            )}
            {(rs.critical_hurdles_count ?? 0) > 0 && (
              <div className="mt-2 flex items-center gap-2 text-[9px] font-mono text-rose-400/80">
                <AlertTriangle size={10} />
                {rs.critical_hurdles_count} critical ice segment{rs.critical_hurdles_count !== 1 ? 's' : ''}
              </div>
            )}
          </>
        )}
      </div>

      {/* Environmental Summary */}
      {es && (
        <div className="panel-card p-3 border border-sky-500/15 bg-sky-950/8">
          <div className="section-header mb-2">
            <Activity size={11} className="text-sky-400" />
            <span className="text-sky-400/90">Atmospheric &amp; Ocean</span>
          </div>
          <div className="flex items-center gap-1.5 mb-2">
            <span className="font-mono text-[8px] px-1.5 py-0.5 rounded border border-sky-500/25 text-sky-400/70">
              {es.data_source?.replace(/_/g, ' ').toUpperCase() ?? 'DATA SOURCE'}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Metric label="Max Crosswind" value={es.max_crosswind_knots?.toFixed(1)} unit="kts" />
            <Metric label="Avg Drift" value={es.avg_drift_velocity?.toFixed(2)} unit="kts" />
            <Metric label="Max Drift" value={es.max_drift_velocity?.toFixed(2)} unit="kts" />
          </div>
        </div>
      )}

      {/* Selected Waypoint Physics Detail HUD */}
      {selectedWpObj && (
        <div className="panel-card p-3 border border-amber-500/25 bg-amber-950/8">
          <div className="flex justify-between items-center mb-2.5">
            <div className="section-header m-0 text-amber-400">
              <Target size={11} />
              T+{selectedWpObj.eta_hours ?? 0}h Physics Detail
            </div>
            <button
              onClick={() => onSelectWaypoint && onSelectWaypoint(null)}
              className="font-mono text-[9px] text-slate-500 hover:text-slate-300 transition-colors"
            >
              CLEAR
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 mb-2.5 p-2 bg-slate-900/60 rounded border border-slate-800/60">
            <div>
              <span className="font-mono text-[9px] text-slate-500 block">COORDINATES</span>
              <span className="font-mono text-[10px] text-slate-200">
                {selectedWpObj.coordinates ? `${Math.abs(selectedWpObj.coordinates[1]).toFixed(3)}°S` : '—'}
              </span>
              <span className="font-mono text-[10px] text-slate-200 block">
                {selectedWpObj.coordinates ? `${selectedWpObj.coordinates[0].toFixed(3)}°E` : ''}
              </span>
            </div>
            <div>
              <span className="font-mono text-[9px] text-slate-500 block">HAZARD ZONE</span>
              <span className={`font-mono text-[10px] font-bold ${
                (selectedWpObj.sic ?? 0) > 0.4 ? 'text-rose-400' : (selectedWpObj.sic ?? 0) > 0.15 ? 'text-amber-400' : 'text-cyan-400'
              }`}>
                {selectedWpObj.hazard_type ?? selectedWpObj.navigation_mode ?? '—'}
              </span>
            </div>
          </div>

          {/* V_eff Equation Breakdown */}
          <div className="p-2 bg-slate-900/80 rounded border border-slate-800/60 font-mono text-[10px]">
            <div className="flex justify-between items-center border-b border-slate-800/50 pb-1.5 mb-1.5">
              <span className="text-slate-400 font-bold flex items-center gap-1.5">
                <Zap size={11} className="text-violet-400" />
                EFFECTIVE SPEED
              </span>
              <span className="text-violet-300 font-bold text-sm">
                {(selectedWpObj.effective_ship_speed ?? selectedWpObj.speed_knots ?? 0).toFixed(1)} KTS
              </span>
            </div>
            <div className="flex justify-between text-slate-400 mb-1">
              <span className="flex items-center gap-1"><Ship size={9} /> Base Speed</span>
              <span className="text-slate-300">14.0 kts</span>
            </div>
            {(selectedWpObj.sic ?? 0) > 0 && (
              <div className="flex justify-between text-rose-400 mb-1">
                <span className="flex items-center gap-1"><TrendingDown size={9} /> Ice Resistance ({((selectedWpObj.sic ?? 0) * 100).toFixed(0)}%)</span>
                <span>-{(14.0 * Math.pow(selectedWpObj.sic ?? 0, 1.5) * ((telemetry.safety_weight ?? 1) + 0.5)).toFixed(1)} kts</span>
              </div>
            )}
            {(selectedWpObj.wind_speed_knots ?? 0) > 0 && (
              <div className="flex justify-between text-sky-400 mb-1">
                <span className="flex items-center gap-1"><Wind size={9} /> Wind {(selectedWpObj.wind_speed_knots ?? 0).toFixed(1)}kts @ {(selectedWpObj.wind_direction ?? 0).toFixed(0)}°</span>
                <span className="text-sky-300 flex items-center gap-0.5"><TrendingUp size={8} /></span>
              </div>
            )}
            {(selectedWpObj.drift_velocity ?? 0) > 0.05 && (
              <div className="flex justify-between text-purple-400">
                <span className="flex items-center gap-1"><Compass size={9} /> Ekman Drift</span>
                <span>{(selectedWpObj.drift_velocity ?? 0).toFixed(2)} kts @ {(selectedWpObj.drift_angle ?? 0).toFixed(0)}°</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Waypoint Inspector List */}
      {wps.length > 0 && (
        <div className="panel-card flex flex-col">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center justify-between p-3 bg-ocean-900/50 hover:bg-ocean-800/50 transition-colors rounded-t-lg"
          >
            <div className="flex items-center gap-1.5">
              <Crosshair size={12} className="text-sonar-500" />
              <span className="section-header m-0">Waypoint Inspector ({wps.length})</span>
            </div>
            {expanded ? <ChevronUp size={14} className="text-sonar-500" /> : <ChevronDown size={14} className="text-sonar-500" />}
          </button>

          {expanded && (
            <div className="border-t border-slate-800/50">
              <p className="font-mono text-[9px] text-slate-600 px-3 py-1.5">
                Click a waypoint to inspect physics & highlight on map
              </p>
              <div className="max-h-72 overflow-y-auto custom-scrollbar px-2 pb-2">
                {wps.map((wp, idx) => (
                  <WaypointRow
                    key={idx}
                    wp={wp}
                    isSelected={selectedWaypoint && wp.coordinates && Math.abs(wp.coordinates[0] - selectedWaypoint[0]) < 0.001 && Math.abs(wp.coordinates[1] - selectedWaypoint[1]) < 0.001}
                    onSelect={(w) => onSelectWaypoint && onSelectWaypoint(w.coordinates)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
