/**
 * 3D Rally Timeline — the full-stack output (Layers 1-6) per rally.
 * Shows the reconstructed shots, 3D ball events (racket / front-wall / floor),
 * and the rally outcome, each dimmed by its confidence. A prominent quality
 * banner makes clear when the reconstruction is reliable vs low-confidence
 * (bounded by ball detection + calibration quality).
 */
const P_COLORS = { 1: "#DFFF00", 2: "#00F0FF" };

const EVENT_STYLE = {
  racket: { label: "hit", color: "#DFFF00" },
  front_wall: { label: "front wall", color: "#FF4444" },
  floor: { label: "bounce", color: "#888" },
  left_wall: { label: "side wall", color: "#00F0FF" },
  right_wall: { label: "side wall", color: "#00F0FF" },
  back_wall: { label: "back wall", color: "#9966FF" },
};

const RESULT_LABEL = {
  good: "good", down_tin: "↓ tin", out: "✗ out", not_up: "not up", winner: "winner",
};

export default function Timeline3DPanel({ data, names = { 1: "Player 1", 2: "Player 2" } }) {
  if (!data || !data.rallies) return null;
  const nm = (pid) => names[pid] || (pid ? `Player ${pid}` : "—");
  const reliable = data.mean_consistency_px != null && data.mean_consistency_px < 15;

  return (
    <div className="space-y-4">
      {/* Quality banner — confidence first */}
      <div className={`rounded-lg px-4 py-3 border ${reliable
        ? "bg-green-900/15 border-green-700/40" : "bg-amber-900/15 border-amber-700/40"}`}>
        <div className="flex items-center gap-2">
          <span className={`text-sm font-semibold ${reliable ? "text-green-400" : "text-amber-400"}`}>
            {reliable ? "Reconstruction reliable" : "Low-confidence reconstruction"}
          </span>
          {data.mean_consistency_px != null && (
            <span className="text-xs text-muted-foreground ml-auto font-mono">
              fit error {data.mean_consistency_px}px
            </span>
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          {data.quality_note ||
            (reliable ? "Shots and tin/out events below are trustworthy."
              : "The 3D fit error is high — improve the court calibration and add ball labels (the flywheel) to sharpen this. Shots below are best-effort and dimmed by confidence.")}
        </p>
      </div>

      {/* Per-rally timelines */}
      {data.rallies.map((r) => (
        <div key={r.rally_id} className="rounded-lg border border-border bg-background p-3">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-primary font-mono font-bold text-sm">Rally {r.rally_id}</span>
            <span className="text-xs text-muted-foreground">
              {r.n_contacts} contacts · {(r.shots || []).length} shots · {r.n_ball_samples} ball samples
            </span>
            {r.outcome?.winner && (
              <span className="ml-auto text-xs px-2 py-0.5 rounded-full"
                style={{ background: `${P_COLORS[r.outcome.winner]}22`, color: P_COLORS[r.outcome.winner] }}>
                {nm(r.outcome.winner)} won ({r.outcome.reason})
                {r.outcome.confidence < 0.5 && <span className="opacity-60"> · low conf</span>}
              </span>
            )}
          </div>

          {r.error ? (
            <div className="text-xs text-muted-foreground">{r.error}</div>
          ) : (
            <>
              {/* Ball event ribbon */}
              {(r.ball_events || []).length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {r.ball_events.map((e, i) => {
                    const st = EVENT_STYLE[e.kind] || { label: e.kind, color: "#666" };
                    return (
                      <span key={i}
                        className="text-[10px] px-1.5 py-0.5 rounded font-mono"
                        style={{ background: `${st.color}22`, color: st.color, opacity: 0.4 + 0.6 * (e.confidence || 0) }}
                        title={`t=${e.t}s conf=${e.confidence}`}>
                        {st.label}{e.striker ? ` ${nm(e.striker)[0]}` : ""}
                      </span>
                    );
                  })}
                </div>
              )}

              {/* Shots table */}
              {(r.shots || []).length > 0 ? (
                <div className="space-y-1">
                  {r.shots.map((s) => (
                    <div key={s.shot_id}
                      className="flex items-center gap-2 text-xs font-mono rounded px-2 py-1 bg-muted/20"
                      style={{ opacity: 0.45 + 0.55 * (s.confidence || 0) }}>
                      <span className="text-muted-foreground w-7">#{s.shot_id}</span>
                      <span className="w-10 text-muted-foreground">{s.t_contact}s</span>
                      <span className="w-3 h-3 rounded-full flex-shrink-0"
                        style={{ background: P_COLORS[s.striker] || "#555" }} />
                      <span className="w-24 truncate">{nm(s.striker)}</span>
                      <span className="flex-1 capitalize">{s.type}{s.hand ? ` · ${s.hand}` : ""}</span>
                      <span className={`w-14 text-right ${s.result === "down_tin" || s.result === "out" ? "text-red-400" : "text-muted-foreground"}`}>
                        {RESULT_LABEL[s.result] || s.result}
                      </span>
                      <span className="w-8 text-right text-muted-foreground">{Math.round((s.confidence || 0) * 100)}%</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-muted-foreground">No shots reconstructed.</div>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  );
}
