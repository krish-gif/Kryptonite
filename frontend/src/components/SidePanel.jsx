/**
 * SidePanel — right column wrapper containing all metric/chart cards.
 *
 * Props:
 *   data    – parsed forecast manifest
 *   isStale – staleness flag
 *   error   – Error | null
 */

import CurrentConditions from './CurrentConditions'
import ExtentTrendChart  from './ExtentTrendChart'
import LastUpdated       from './LastUpdated'

export default function SidePanel({ data, isStale, error }) {
  return (
    <aside className="side-panel" aria-label="Forecast metrics panel">
      <CurrentConditions extentHistory={data?.extent_history} />
      <ExtentTrendChart
        extentHistory={data?.extent_history}
        predictions={data?.predictions}
      />
      <LastUpdated
        generatedAt={data?.generated_at}
        isStale={isStale}
        error={error}
      />
    </aside>
  )
}
