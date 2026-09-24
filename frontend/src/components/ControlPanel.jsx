import { useState, useEffect } from 'react'
import { MapPin, Crosshair, Clock, Shield, ChevronDown } from 'lucide-react'

// Expanded list of departure ports
const START_POINTS = [
  { id: 'cpt',  label: 'Cape Town, ZA',       lat: -33.9249, lon:  18.4241 },
  { id: 'hbt',  label: 'Hobart, AU',           lat: -42.8821, lon: 147.3272 },
  { id: 'fkl',  label: 'Stanley, Falklands',   lat: -51.6938, lon: -57.8594 },
  { id: 'bue',  label: 'Ushuaia, Argentina',   lat: -54.8000, lon: -68.3000 },
  { id: 'chr',  label: 'Christchurch, NZ',     lat: -43.5321, lon: 172.6362 },
  { id: 'mau',  label: 'Port-aux-Français, FR',lat: -49.3526, lon:  70.2194 },
]

// Expanded list of Antarctic destinations
const STATIONS = [
  { id: 'maitri',       label: 'Maitri (IND)',           lat: -70.7600, lon:  11.4400 },
  { id: 'bharati',      label: 'Bharati (IND)',           lat: -69.4000, lon:  76.1900 },
  { id: 'sanae',        label: 'SANAE IV (ZA)',           lat: -71.6730, lon:   2.8430 },
  { id: 'novolaz',      label: 'Novolazarevskaya (RU)',   lat: -70.7682, lon:  11.8323 },
  { id: 'syowa',        label: 'Syowa (JA)',              lat: -69.0069, lon:  39.5897 },
  { id: 'mawson',       label: 'Mawson (AU)',             lat: -67.6031, lon:  62.8728 },
  { id: 'davis',        label: 'Davis (AU)',               lat: -68.5770, lon:  77.9680 },
  { id: 'princess',     label: 'Princess Elizabeth (BE)', lat: -71.9500, lon:  23.3470 },
]

function Select({ label, value, onChange, options, disabled }) {
  return (
    <div className="relative">
      <label className="block font-mono text-[9px] text-slate-500 tracking-widest uppercase mb-1">{label}</label>
      <div className="relative">
        <select
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          disabled={disabled}
          className="w-full bg-ocean-950 border border-sonar-500/25 rounded px-2 py-1.5
                     font-mono text-xs text-slate-200 outline-none focus:border-sonar-500/60
                     focus:ring-1 focus:ring-sonar-500/20 transition-all appearance-none pr-7
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {options.map((opt, i) => (
            <option key={opt.id} value={i}>{opt.label}</option>
          ))}
        </select>
        <ChevronDown size={11} className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none" />
      </div>
    </div>
  )
}

export default function ControlPanel({ onCalculate, loading, customPoints = [], onClearPoints }) {
  const [startIdx, setStartIdx] = useState(0)
  const [endIdx,   setEndIdx]   = useState(0)
  const [horizon,  setHorizon]  = useState(48)
  const [safety,   setSafety]   = useState(50)

  const useCustomStart = customPoints.length > 0
  const useCustomEnd   = customPoints.length > 1

  const handleCalc = () => {
    const start = useCustomStart
      ? { lat: customPoints[0][1], lon: customPoints[0][0] }
      : START_POINTS[startIdx]
    const end = useCustomEnd
      ? { lat: customPoints[1][1], lon: customPoints[1][0] }
      : STATIONS[endIdx]
    onCalculate({ start, end, forecastWindow: horizon, safetyWeight: safety })
  }

  const safetyLabel = safety < 20 ? 'AGGRESSIVE' : safety < 50 ? 'MODERATE' : safety < 80 ? 'CAUTIOUS' : 'MAX SAFE'
  const safetyColor = safety < 20 ? 'text-rose-400' : safety < 50 ? 'text-amber-400' : safety < 80 ? 'text-cyan-400' : 'text-radar-400'

  return (
    <div className="panel-card p-3 flex flex-col gap-3">
      <div className="section-header">
        <Crosshair size={11} />
        Route Configuration
      </div>

      {/* Departure Point */}
      <div>
        <div className="flex justify-between items-center mb-1">
          <label className="font-mono text-[9px] text-slate-500 tracking-widest uppercase">Departure Port</label>
          {useCustomStart && (
            <button onClick={onClearPoints} className="font-mono text-[8px] text-hazard-400 hover:text-hazard-300 flex items-center gap-0.5">
              ✕ CLEAR CUSTOM
            </button>
          )}
        </div>
        {useCustomStart ? (
          <div className="w-full bg-ocean-950 border border-radar-500/50 rounded px-2 py-1.5 font-mono text-xs text-radar-400 flex items-center gap-1.5">
            <MapPin size={10} className="flex-none" />
            <span>Map: {customPoints[0][1].toFixed(3)}°, {customPoints[0][0].toFixed(3)}°</span>
          </div>
        ) : (
          <Select
            label=""
            value={startIdx}
            onChange={setStartIdx}
            options={START_POINTS}
          />
        )}
      </div>

      {/* Destination */}
      <div>
        <label className="block font-mono text-[9px] text-slate-500 tracking-widest uppercase mb-1">Antarctic Destination</label>
        {useCustomEnd ? (
          <div className="w-full bg-ocean-950 border border-sonar-500/50 rounded px-2 py-1.5 font-mono text-xs text-sonar-400 flex items-center gap-1.5">
            <MapPin size={10} className="flex-none" />
            <span>Map: {customPoints[1][1].toFixed(3)}°, {customPoints[1][0].toFixed(3)}°</span>
          </div>
        ) : (
          <Select
            label=""
            value={endIdx}
            onChange={setEndIdx}
            options={STATIONS}
          />
        )}
      </div>

      {/* Map Click Hint */}
      <div className="text-center font-mono text-[9px] text-slate-700 py-1 border border-dashed border-slate-800 rounded">
        or click map to set custom start/destination
      </div>

      {/* Safety Weight Slider */}
      <div>
        <div className="flex justify-between items-center mb-1">
          <label className="font-mono text-[9px] text-slate-500 tracking-widest uppercase flex items-center gap-1">
            <Shield size={9} />
            Safety Weight
          </label>
          <span className={`font-mono text-[9px] font-bold ${safetyColor}`}>{safety}% · {safetyLabel}</span>
        </div>
        <input
          type="range"
          min="0" max="100"
          value={safety}
          onChange={e => setSafety(Number(e.target.value))}
          className="w-full accent-sonar-500 h-1.5 rounded-full"
        />
        <div className="flex justify-between mt-0.5">
          <span className="font-mono text-[8px] text-rose-500/60">Aggressive</span>
          <span className="font-mono text-[8px] text-radar-500/60">Max Safe</span>
        </div>
      </div>

      {/* ConvLSTM Forecast Horizon */}
      <div>
        <label className="block font-mono text-[9px] text-slate-500 tracking-widest uppercase mb-1 flex items-center gap-1">
          <Clock size={9} />
          ConvLSTM Horizon
        </label>
        <div className="grid grid-cols-3 gap-1.5">
          {[24, 48, 72].map(h => (
            <button
              key={h}
              type="button"
              onClick={() => setHorizon(h)}
              className={`py-1.5 rounded font-mono text-xs font-semibold transition-all duration-150 ${
                horizon === h
                  ? 'bg-sonar-500/20 border border-sonar-500/60 text-sonar-400 shadow-sm'
                  : 'bg-ocean-950 border border-ocean-600 text-slate-500 hover:border-sonar-500/30 hover:text-slate-300'
              }`}
            >
              +{h}H
            </button>
          ))}
        </div>
      </div>

      {/* Calculate CTA */}
      <button
        type="button"
        onClick={handleCalc}
        disabled={loading}
        className="mt-1 w-full py-2.5 rounded font-bold font-mono text-xs tracking-widest
                   transition-all duration-300 relative overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed
                   bg-gradient-to-r from-ocean-900 to-slate-900
                   border border-radar-500/40 text-radar-400
                   hover:border-radar-500/70 hover:shadow-lg hover:shadow-radar-500/10
                   active:scale-[0.98]"
      >
        <span className="relative z-10">
          {loading ? '⟳ COMPUTING 3 CORRIDORS…' : '⚡ CALCULATE OPTIMAL ROUTE'}
        </span>
      </button>

      {!loading && (
        <p className="font-mono text-[8px] text-slate-700 text-center leading-relaxed -mt-1">
          Computes Cautious · Optimal · Aggressive corridors simultaneously
        </p>
      )}
    </div>
  )
}
