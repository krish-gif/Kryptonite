/**
 * App.jsx — Kryptonite Antarctic Navigation System
 *
 * Root layout: full-bleed map canvas + collapsible right command sidebar.
 * All state lives here; children receive only what they need.
 */

import { useState } from 'react'
import MapView        from './components/MapView'
import ControlPanel   from './components/ControlPanel'
import TelemetryBoard from './components/TelemetryBoard'
import { useNavigationEngine } from './hooks/useNavigationEngine'

// ── Header bar ─────────────────────────────────────────────────
function Header({ utcTime }) {
  return (
    <header className="h-12 flex-none flex items-center justify-between
                       px-4 border-b border-sonar-500/10 bg-ocean-900/95 backdrop-blur-sm z-30">
      {/* Logo + title */}
      <div className="flex items-center gap-3">
        <div className="w-7 h-7 rounded flex items-center justify-center
                        bg-sonar-500/10 border border-sonar-500/30">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
               stroke="#00d4ff" strokeWidth="2">
            <polygon points="3 11 22 2 13 21 11 13 3 11"/>
          </svg>
        </div>
        <div>
          <h1 className="font-mono text-sm font-bold tracking-widest text-slate-100 leading-none">
            KRYPTONITE
          </h1>
          <p className="font-mono text-[10px] tracking-widest text-sonar-500/70 leading-none mt-0.5">
            Antarctic Navigation System · SIH 2025
          </p>
        </div>
      </div>

      {/* Centre — system status */}
      <div className="hidden md:flex items-center gap-6">
        {[
          { label: 'A* ENGINE',     val: 'ONLINE', ok: true  },
          { label: 'ConvLSTM',      val: 'READY',  ok: true  },
          { label: 'ICE GRID',      val: 'SYNCED', ok: true  },
          { label: 'BACKEND API',   val: 'STANDBY',ok: false },
        ].map(({ label, val, ok }) => (
          <div key={label} className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${ok ? 'bg-radar-500 animate-pulse-radar' : 'bg-caution-500'}`} />
            <span className="font-mono text-[10px] text-slate-600 tracking-widest">{label}</span>
            <span className={`font-mono text-[10px] font-semibold tracking-wider ${ok ? 'text-radar-400' : 'text-caution-400'}`}>
              {val}
            </span>
          </div>
        ))}
      </div>

      {/* Right — clock */}
      <div className="font-mono text-xs text-slate-500 tabular-nums tracking-wider">
        <span className="text-sonar-500/60 mr-2">UTC</span>
        {utcTime}
      </div>
    </header>
  )
}

// ── Clock hook ─────────────────────────────────────────────────
function useUTCClock() {
  const [time, setTime] = useState(() =>
    new Date().toISOString().slice(11, 19)
  )
  useState(() => {
    const id = setInterval(() => setTime(new Date().toISOString().slice(11, 19)), 1000)
    return () => clearInterval(id)
  })
  return time
}

// ── App ────────────────────────────────────────────────────────
export default function App() {
  const { calculate, result, loading, error } = useNavigationEngine()
  const utcTime = useUTCClock()

  return (
    <div className="flex flex-col w-full h-full bg-ocean-950">
      <Header utcTime={utcTime} />

      {/* Body: map + sidebar */}
      <div className="flex flex-1 min-h-0">

        {/* ── Map — takes remaining space ──────────────── */}
        <main className="flex-1 min-w-0 relative">
          <MapView result={result} loading={loading} />
        </main>

        {/* ── Right command sidebar ─────────────────────── */}
        <aside
          className="w-72 flex-none flex flex-col gap-3 p-3 overflow-y-auto
                     border-l border-sonar-500/10 bg-ocean-900/95 backdrop-blur-sm"
          style={{ scrollbarWidth: 'thin' }}
        >
          {/* Sidebar header */}
          <div className="flex items-center gap-2 pt-1">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none"
                 stroke="#00d4ff" strokeWidth="2">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/>
            </svg>
            <span className="font-mono text-[10px] tracking-widest text-sonar-500/80 uppercase">
              Command Interface
            </span>
          </div>

          {/* Route calculator */}
          <ControlPanel onCalculate={calculate} loading={loading} />

          {/* Divider */}
          <div className="flex items-center gap-2">
            <div className="flex-1 h-px bg-sonar-500/10" />
            <span className="font-mono text-[10px] text-slate-700 tracking-widest">OUTPUT</span>
            <div className="flex-1 h-px bg-sonar-500/10" />
          </div>

          {/* Telemetry readout */}
          <TelemetryBoard telemetry={result?.telemetry ?? null} error={error} />

          {/* Legend */}
          <div className="panel-card p-3 mt-auto">
            <div className="section-header mb-2">Map Legend</div>
            <div className="space-y-1.5">
              {[
                { color: '#00d4ff', label: 'Optimal Route (A*)' },
                { color: '#ff6b00', label: 'Iceberg Position'   },
                { color: 'rgba(255,107,0,0.35)', label: 'High-Risk Corridor', dashed: true },
                { color: 'rgba(255,215,0,0.35)',  label: 'Moderate Corridor',  dashed: true },
              ].map(({ color, label, dashed }) => (
                <div key={label} className="flex items-center gap-2">
                  <div className="flex-none w-8 flex items-center">
                    <div
                      className="h-0.5 w-8 rounded"
                      style={{
                        background: color,
                        borderTop: dashed ? `1px dashed ${color}` : undefined,
                      }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-slate-500">{label}</span>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  )
}
