/**
 * ConcentrationLegend — fixed color-scale legend overlaid on the map.
 * Shows the ice concentration gradient from 0% (transparent/dark)
 * to 100% (near-white).
 */

export default function ConcentrationLegend() {
  return (
    <div
      className="concentration-legend"
      role="img"
      aria-label="Sea-ice concentration color scale from 0% to 100%"
    >
      <div className="legend-title">Ice Concentration</div>
      <div className="legend-gradient" />
      <div className="legend-labels">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  )
}
