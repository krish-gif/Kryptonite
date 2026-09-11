/**
 * LeadDaySelector — segmented control overlaid on the map.
 *
 * Props:
 *   selectedDay  – currently selected lead day (0 = today, 1/2/3 = forecast)
 *   predictions  – forecast predictions array from forecast.json
 *   onChange     – callback(leadDay: number)
 */

export default function LeadDaySelector({ selectedDay, predictions, onChange }) {
  const options = [
    { value: 0, label: 'Today' },
    ...(predictions || []).map(p => ({
      value: p.lead_day,
      label: `+${p.lead_day}d`,
    })),
  ]

  return (
    <div
      className="lead-day-selector"
      role="group"
      aria-label="Forecast lead day selector"
    >
      {options.map(opt => (
        <button
          key={opt.value}
          id={`lead-day-btn-${opt.value}`}
          className={`lead-day-btn${selectedDay === opt.value ? ' active' : ''}`}
          onClick={() => onChange(opt.value)}
          aria-pressed={selectedDay === opt.value}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}
