/**
 * StaleBanner — shown when forecast.json is stale (> 36 h old)
 * or when the fetch failed entirely.
 */

export default function StaleBanner({ error, isStale }) {
  if (!isStale && !error) return null

  const isError = !!error && !isStale
  const variant  = isError ? 'error' : 'warn'
  const icon     = isError ? '⚠' : '⏱'

  const message = isError
    ? 'Forecast unavailable — could not fetch forecast.json. Showing last known data.'
    : 'Forecast data may be outdated — last update was more than 36 hours ago. The pipeline job may have failed or run late.'

  return (
    <div className={`stale-banner ${variant}`} role="alert" aria-live="polite">
      <span className="stale-banner-icon" aria-hidden="true">{icon}</span>
      <span>{message}</span>
    </div>
  )
}
