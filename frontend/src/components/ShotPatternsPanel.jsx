/**
 * Shot Patterns & Error Zones panel.
 *
 * Shows per player:
 *   - shot-origin heatmap (where they hit their shots from)
 *   - error-zone heatmap (where they were positioned when they lost points)
 *   - rally-length-by-winner + tactical insights
 *
 * Props:
 *   data — result from /api/analysis/shot-patterns/{id}
 */

const P_COLORS = { 1: "#DFFF00", 2: "#00F0FF" };
const ZONE_ROWS = ["front", "mid", "back"];
const ZONE_COLS = ["left", "center", "right"];

export default function ShotPatternsPanel({ data, names = { 1: "Player 1", 2: "Player 2" } }) {
  const nm = (pid) => names[pid] || `Player ${pid}`;
  const loc = (s) => (s || "").replace(/Player 1/g, nm(1)).replace(/Player 2/g, nm(2));
  if (!data) return null;

  if (!data.calibrated) {
    return (
      <div className="rounded-lg border border-yellow-700/40 bg-yellow-900/10 p-4 text-sm text-yellow-200">
        Calibrate the court (Rallies tab) to unlock shot-origin and error-zone analysis.
      </div>
    );
  }

  const totalTagged = (data.points_lost?.["1"] || 0) + (data.points_lost?.["2"] || 0);

  return (
    <div className="space-y-4">
      {/* Insights headline */}
      {data.insights?.length > 0 && (
        <div className="rounded-lg border border-primary/30 bg-primary/5 p-4">
          <div className="text-xs uppercase tracking-wide text-primary mb-2 font-semibold">
            Shot &amp; Error Read
          </div>
          <ul className="space-y-1.5">
            {data.insights.map((ins, i) => (
              <li key={i} className="text-sm text-foreground flex gap-2">
                <span className="text-primary">▸</span><span>{loc(ins)}</span>
              </li>
            ))}
          </ul>
          <div className="text-[10px] text-muted-foreground mt-2">
            {data.total_shots?.["1"] + data.total_shots?.["2"]} shots detected over{" "}
            {data.active_play_s}s of play.
            {totalTagged === 0 && " Tag rally outcomes to unlock error zones."}
          </div>
        </div>
      )}

      {/* Shot-origin heatmaps */}
      <div>
        <div className="text-sm font-semibold mb-2">Shot origins — where each player hits from</div>
        <div className="grid grid-cols-2 gap-4">
          {[1, 2].map(pid => (
            <ZoneCard key={pid} pid={pid} name={nm(pid)}
              zonePct={data.shot_origin_pct?.[String(pid)] || {}}
              subtitle={`${data.total_shots?.[String(pid)] || 0} shots`} />
          ))}
        </div>
      </div>

      {/* Error zones — only meaningful with tagged outcomes */}
      {totalTagged > 0 && (
        <div>
          <div className="text-sm font-semibold mb-2">
            Error zones — where each player was when they lost the point
          </div>
          <div className="grid grid-cols-2 gap-4">
            {[1, 2].map(pid => {
              const ez = data.error_zones?.[String(pid)] || {};
              const lost = data.points_lost?.[String(pid)] || 0;
              return (
                <ZoneCard key={pid} pid={pid} name={nm(pid)} zonePct={ez} raw
                  subtitle={`${lost} points lost`} danger />
              );
            })}
          </div>
        </div>
      )}

      {/* Rally length by winner */}
      <div className="rounded-lg border border-border bg-background p-3">
        <div className="text-sm font-semibold mb-2">Winning rally length</div>
        <div className="grid grid-cols-2 gap-4">
          {[1, 2].map(pid => {
            const avg = data.avg_rally_len_by_winner?.[String(pid)];
            return (
              <div key={pid} className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ background: P_COLORS[pid] }} />
                <span className="text-xs text-muted-foreground">{nm(pid)} wins avg</span>
                <span className="font-mono font-bold ml-auto">
                  {avg != null ? `${avg} shots` : "—"}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function ZoneCard({ pid, name, zonePct, subtitle, danger, raw }) {
  const color = danger ? "#FF4444" : P_COLORS[pid];
  const vals = Object.values(zonePct);
  const max = Math.max(1, ...vals);
  return (
    <div className="rounded-lg border border-border bg-background p-3">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-3 h-3 rounded-full" style={{ background: P_COLORS[pid] }} />
        <span className="text-sm font-semibold">{name || `Player ${pid}`}</span>
        <span className="text-[10px] text-muted-foreground ml-auto">{subtitle}</span>
      </div>
      <div className="grid grid-cols-3 gap-0.5 aspect-[2/3]">
        {ZONE_ROWS.map(row =>
          ZONE_COLS.map(col => {
            const key = `${row}-${col}`;
            const v = zonePct[key] || 0;
            const alpha = v / max;
            const hex = Math.round(alpha * 200 + 10).toString(16).padStart(2, "0");
            return (
              <div key={key}
                className="flex items-center justify-center rounded-sm text-[9px] font-mono border border-border/30"
                style={{ background: `${color}${hex}`, color: alpha > 0.5 ? "#000" : "#888" }}
                title={`${key}: ${raw ? v : v + "%"}`}>
                {v > (raw ? 0 : 4) ? (raw ? v : Math.round(v)) : ""}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
