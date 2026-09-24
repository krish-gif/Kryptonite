import { useState, useEffect } from 'react'
import MapView        from './components/MapView'
import ControlPanel   from './components/ControlPanel'
import TelemetryBoard from './components/TelemetryBoard'
import { useNavigationEngine } from './hooks/useNavigationEngine'
import { getHealth } from './api'

function useUTCClock() {
  const [time, setTime] = useState(() => new Date().toISOString().slice(11, 19))
  useEffect(() => {
    const id = setInterval(() => setTime(new Date().toISOString().slice(11, 19)), 1000)
    return () => clearInterval(id)
  }, [])
  return time
}

function StatusDot({ ok, pulse = true }) {
  return (
    <span className={`inline-block w-1.5 h-1.5 rounded-full flex-none ${
      ok ? `bg-radar-500 ${pulse ? 'animate-pulse-radar' : ''}` : 'bg-caution-500 animate-pulse'
    }`} />
  )
}

function Header({ utcTime, backendOnline, fcLoading, healthData }) {
  const indicators = [
    { label: 'A* ENGINE',   val: 'ONLINE',                           ok: true  },
    { label: 'ConvLSTM',    val: healthData?.model_loaded ? 'READY' : 'NO MODEL', ok: healthData?.model_loaded ?? false },
    { label: 'ICE GRID',    val: fcLoading ? 'SYNCING…' : 'SYNCED', ok: !fcLoading },
    { label: 'ENVIRON',     val: healthData?.env_source ? healthData.env_source.toUpperCase().replace(/_/g,'·') : (backendOnline ? 'LIVE' : 'OFFLINE'), ok: backendOnline },
    { label: 'BACKEND',     val: backendOnline ? 'LIVE' : 'STANDBY',ok: backendOnline },
  ]

  return (
    <header className="h-12 flex-none flex items-center justify-between px-4 border-b border-sonar-500/10 bg-ocean-900/95 backdrop-blur-sm z-30">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-br from-sonar-500/20 to-sonar-700/10 border border-sonar-500/30 shadow-inner">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" strokeWidth="2">
            <polygon points="3 11 22 2 13 21 11 13 3 11"/>
          </svg>
        </div>
        <div>
          <h1 className="font-mono text-sm font-bold tracking-widest text-slate-100 leading-none">KRYPTONITE</h1>
          <p className="font-mono text-[9px] tracking-widest text-sonar-500/60 leading-none mt-0.5">Antarctic Navigation Engine v2.3</p>
        </div>
      </div>

      <div className="hidden md:flex items-center gap-5">
        {indicators.map(({ label, val, ok }) => (
          <div key={label} className="flex items-center gap-1.5">
            <StatusDot ok={ok} />
            <span className="font-mono text-[9px] text-slate-600 tracking-widest">{label}</span>
            <span className={`font-mono text-[9px] font-semibold tracking-wide ${ok ? 'text-radar-400' : 'text-caution-400'}`}>
              {val}
            </span>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3">
        {healthData?.grid && (
          <div className="hidden lg:flex items-center gap-1 font-mono text-[9px] text-slate-600">
            <span className="text-sonar-500/50 mr-1">GRID</span>
            {healthData.grid.height}×{healthData.grid.width}
            <span className="mx-1 text-slate-700">|</span>
            {healthData.grid.lat_range?.[0]}°→{healthData.grid.lat_range?.[1]}°S
          </div>
        )}
        <div className="font-mono text-xs text-slate-500 tabular-nums tracking-wider">
          <span className="text-sonar-500/60 mr-1.5">UTC</span>{utcTime}
        </div>
      </div>
    </header>
  )
}

export default function App() {
  const {
    calculate, result, forecast, iceGrid, envVectors,
    fetchIceGrid, loading, fcLoading, error, setActiveRoute
  } = useNavigationEngine()

  const utcTime = useUTCClock()
  const [leadDay, setLeadDay]           = useState(24)
  const [backendOnline, setBackendOnline] = useState(false)
  const [healthData, setHealthData]     = useState(null)
  const [customPoints, setCustomPoints] = useState([])
  const [selectedWaypoint, setSelectedWaypoint] = useState(null)

  const handleMapClick = (lngLat) => {
    setCustomPoints(prev => {
      if (prev.length >= 2) return [[lngLat.lng, lngLat.lat]]
      return [...prev, [lngLat.lng, lngLat.lat]]
    })
  }

  // Health polling with data extraction
  useEffect(() => {
    const check = async () => {
      try {
        const data = await getHealth()
        setBackendOnline(true)
        setHealthData(data)
      } catch {
        setBackendOnline(false)
        setHealthData(null)
      }
    }
    check()
    const id = setInterval(check, 15000)
    return () => clearInterval(id)
  }, [])

  // Fetch ice grid when horizon changes
  useEffect(() => {
    fetchIceGrid(leadDay)
  }, [leadDay, fetchIceGrid])

  return (
    <div className="flex flex-col w-full h-full bg-ocean-950">
      <Header utcTime={utcTime} backendOnline={backendOnline} fcLoading={fcLoading} healthData={healthData} />

      <div className="flex flex-1 min-h-0">
        {/* Map */}
        <main className="flex-1 min-w-0 relative">
          <MapView
            result={result}
            iceGrid={iceGrid}
            envVectors={envVectors}
            selectedWaypoint={selectedWaypoint}
            loading={loading}
            onMapClick={handleMapClick}
            customPoints={customPoints}
          />
        </main>

        {/* Right Panel */}
        <aside className="w-80 flex-none flex flex-col gap-2.5 p-2.5 overflow-y-auto border-l border-sonar-500/10 bg-ocean-900/95 backdrop-blur-sm custom-scrollbar">

          {/* Forecast horizon selector */}
          <div className="panel-card p-3">
            <div className="section-header mb-2">Dynamic Ice Grid Overlay</div>
            <div className="grid grid-cols-3 gap-1.5">
              {[24, 48, 72].map(h => (
                <button
                  key={h}
                  onClick={() => setLeadDay(h)}
                  className={`py-1.5 rounded font-mono text-xs font-semibold transition-all duration-150 ${
                    leadDay === h
                      ? 'bg-sonar-500/20 border border-sonar-500/60 text-sonar-400'
                      : 'bg-ocean-900 border border-ocean-600 text-slate-500 hover:border-sonar-500/30 hover:text-slate-300'
                  }`}
                >
                  +{h}H
                </button>
              ))}
            </div>
            {forecast?.forecast_date && (
              <div className="mt-2 font-mono text-[9px] text-slate-600">
                Forecast: <span className="text-sonar-400">{forecast.forecast_date?.slice(0, 19)?.replace('T', ' ')} UTC</span>
              </div>
            )}
            {forecast?.extent_km2 && (
              <div className="mt-2 grid grid-cols-3 gap-1 text-center">
                {Object.entries(forecast.extent_km2).map(([key, val]) => (
                  <div key={key} className="bg-ocean-950/60 rounded p-1">
                    <div className="font-mono text-[8px] text-slate-600 uppercase">{key}</div>
                    <div className="font-mono text-[10px] text-sonar-400">{(val/1e6).toFixed(2)}M km²</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <ControlPanel
            onCalculate={calculate}
            loading={loading}
            customPoints={customPoints}
            onClearPoints={() => { setCustomPoints([]); setSelectedWaypoint(null) }}
          />

          <div className="flex items-center gap-2">
            <div className="flex-1 h-px bg-sonar-500/10" />
            <span className="font-mono text-[9px] text-slate-700 tracking-widest">OUTPUT</span>
            <div className="flex-1 h-px bg-sonar-500/10" />
          </div>

          <TelemetryBoard
            telemetry={result?.routes?.[result.activeRoute]?.telemetry ?? null}
            error={error}
            activeRoute={result?.activeRoute ?? 'optimal'}
            onSelectRoute={result ? setActiveRoute : null}
            selectedWaypoint={selectedWaypoint}
            onSelectWaypoint={setSelectedWaypoint}
          />
        </aside>
      </div>
    </div>
  )
}
