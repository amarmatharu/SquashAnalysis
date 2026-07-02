/**
 * Court Control panel — tactical movement analysis toward "how to beat this opponent".
 *
 * Shows, per player and head-to-head:
 *   - T-control %, avg distance from T
 *   - Distance covered (work rate), court coverage %
 *   - 3×3 tactical zone occupancy grid
 *   - Plain-English tactical insights
 *
 * Props:
 *   data — the court-control result object from /api/analysis/court-control/{id}
 */

const P_COLORS = { 1: "#DFFF00", 2: "#00F0FF" };
const ZONE_ROWS = ["front", "mid", "back"];
const ZONE_COLS = ["left", "center", "right"];

export default function CourtControlPanel({ data, names = { 1: "Player 1", 2: "Player 2" } }) {
  if (!data || !data.players) return null;
  const nm = (pid) => names[pid] || `Player ${pid}`;
  const loc = (s) => (s || "")
    .replace(/Player 1/g, nm(1)).replace(/Player 2/g, nm(2));

  const p1 = data.players["1"];
  const p2 = data.players["2"];

  if (!data.calibrated) {
    return (
      <div className="rounded-lg border border-yellow-700/40 bg-yellow-900/10 p-4 text-sm text-yellow-200">
        Court not calibrated for this match. Calibrate the court (Rallies tab) to unlock
        T-control, court coverage, and tactical zone analysis. Player detection still ran
        ({p1?.frames_detected || 0} / {p2?.frames_detected || 0} frames).
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Tactical insights — the headline */}
      {data.insights?.length > 0 && (
        <div className="rounded-lg border border-primary/30 bg-primary/5 p-4">
          <div className="text-xs uppercase tracking-wide text-primary mb-2 font-semibold">
            Tactical Read
          </div>
          <ul className="space-y-1.5">
            {data.insights.map((ins, i) => (
              <li key={i} className="text-sm text-foreground flex gap-2">
                <span className="text-primary">▸</span>
                <span>{loc(ins)}</span>
              </li>
            ))}
          </ul>
          <div className="text-[10px] text-muted-foreground mt-2">
            Based on {data.active_play_s}s of active play across {data.rally_count} rallies.
          </div>
        </div>
      )}

      {/* Head-to-head metric bars */}
      <div className="rounded-lg border border-border bg-background p-4 space-y-3">
        <div className="text-sm font-semibold mb-1">Head-to-head</div>
        <CompareBar label="T-control" p1={p1.t_control_pct} p2={p2.t_control_pct} unit="%" higherBetter />
        <CompareBar label="Avg distance from T" p1={p1.avg_dist_from_t_m} p2={p2.avg_dist_from_t_m} unit="m" higherBetter={false} />
        <CompareBar label="Distance covered" p1={p1.total_distance_m} p2={p2.total_distance_m} unit="m" neutral />
        <CompareBar label="Court coverage" p1={p1.court_coverage_pct} p2={p2.court_coverage_pct} unit="%" neutral />
        <CompareBar label="Time in back corners" p1={p1.back_corner_pct} p2={p2.back_corner_pct} unit="%" higherBetter={false} />
      </div>

      {/* Tactical zone grids */}
      <div className="grid grid-cols-2 gap-4">
        {[1, 2].map(pid => {
          const p = data.players[String(pid)];
          return (
            <div key={pid} className="rounded-lg border border-border bg-background p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-3 h-3 rounded-full" style={{ background: P_COLORS[pid] }} />
                <span className="text-sm font-semibold">{nm(pid)}</span>
                <span className="text-[10px] text-muted-foreground ml-auto">
                  dominant: {p.dominant_zone}
                </span>
              </div>
              <ZoneGrid zonePct={p.zone_pct} color={P_COLORS[pid]} />
              <div className="grid grid-cols-3 gap-1 mt-2 text-[10px] text-center">
                <DepthChip label="Front" pct={p.depth_pct.front} />
                <DepthChip label="Mid" pct={p.depth_pct.mid} />
                <DepthChip label="Back" pct={p.depth_pct.back} />
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-[10px] text-muted-foreground text-center">
        Zone grid: front wall at top, back wall (camera side) at bottom. Brighter = more time spent there.
      </p>
    </div>
  );
}

function CompareBar({ label, p1, p2, unit, higherBetter, neutral }) {
  const total = (p1 || 0) + (p2 || 0) || 1;
  const w1 = ((p1 || 0) / total) * 100;
  const w2 = ((p2 || 0) / total) * 100;

  // Highlight the "winner" unless neutral
  let lead = null;
  if (!neutral) {
    if (higherBetter) lead = p1 > p2 ? 1 : p2 > p1 ? 2 : null;
    else lead = p1 < p2 ? 1 : p2 < p1 ? 2 : null;
  }

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className={`font-mono ${lead === 1 ? "text-primary font-bold" : "text-muted-foreground"}`}>
          {p1}{unit}
        </span>
        <span className="text-muted-foreground">{label}</span>
        <span className={`font-mono ${lead === 2 ? "text-[#00F0FF] font-bold" : "text-muted-foreground"}`}>
          {p2}{unit}
        </span>
      </div>
      <div className="flex h-2 rounded-full overflow-hidden bg-muted">
        <div style={{ width: `${w1}%`, background: "#DFFF00" }} className="transition-all" />
        <div style={{ width: `${w2}%`, background: "#00F0FF" }} className="transition-all" />
      </div>
    </div>
  );
}

function ZoneGrid({ zonePct, color }) {
  const max = Math.max(1, ...Object.values(zonePct));
  return (
    <div className="grid grid-cols-3 gap-0.5 aspect-[2/3]">
      {ZONE_ROWS.map(row =>
        ZONE_COLS.map(col => {
          const key = `${row}-${col}`;
          const pct = zonePct[key] || 0;
          const alpha = pct / max;
          return (
            <div key={key}
              className="flex items-center justify-center rounded-sm text-[9px] font-mono border border-border/30"
              style={{
                background: `${color}${Math.round(alpha * 200 + 10).toString(16).padStart(2, "0")}`,
                color: alpha > 0.5 ? "#000" : "#888",
              }}
              title={`${key}: ${pct}%`}>
              {pct > 4 ? `${Math.round(pct)}` : ""}
            </div>
          );
        })
      )}
    </div>
  );
}

function DepthChip({ label, pct }) {
  return (
    <div className="rounded bg-muted/40 py-1">
      <div className="text-muted-foreground">{label}</div>
      <div className="font-mono font-bold text-foreground">{pct}%</div>
    </div>
  );
}
