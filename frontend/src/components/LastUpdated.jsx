/**
 * LastUpdated — shows the generated_at timestamp from forecast.json
 * with a status indicator dot (green = fresh, yellow = stale, red = error).
 *
 * Props:
 *   generatedAt – ISO 8601 timestamp string from forecast.json
 *   isStale     – boolean
 *   error       – Error | null
 */

function formatRelativeTime(isoStr) {
  try {
    const d = new Date(isoStr)
    const diff = Math.floor((Date.now() - d.getTime()) / 60_000) // minutes

    if (diff < 1)    return 'just now'
    if (diff < 60)   return `${diff}m ago`
    if (diff < 1440) return `${Math.floor(diff / 60)}h ago`
    return `${Math.floor(diff / 1440)}d ago`
  } catch {
    return 'unknown'
  }
}

function formatAbsoluteTime(isoStr) {
  try {
    return new Date(isoStr).toLocaleString('en-US', {
      month:  'short',
      day:    'numeric',
      hour:   '2-digit',
      minute: '2-digit',
      timeZoneName: 'short',
    })
  } catch {
    return isoStr
  }
}

export default function LastUpdated({ generatedAt, isStale, error }) {
  if (!generatedAt) return null

  const dotClass = error
    ? 'last-updated-dot error'
    : isStale
    ? 'last-updated-dot stale'
    : 'last-updated-dot'

  return (
    <div className="panel-card">
      <div className="panel-card-label">Last Updated</div>
      <div className="last-updated">
        <div className={dotClass} aria-hidden="true" />
        <div>
          <div style={{ color: 'var(--text-primary)', fontSize: '13px', fontWeight: 500 }}>
            {formatRelativeTime(generatedAt)}
          </div>
          <div style={{ color: 'var(--text-muted)', fontSize: '11px', marginTop: '1px' }}>
            {formatAbsoluteTime(generatedAt)}
          </div>
        </div>
      </div>
    </div>
  )
}
