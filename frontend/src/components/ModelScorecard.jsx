/**
 * Ball Model Scorecard — the target for "train it till it's the best model".
 * Shows the metrics that actually define quality (localization + CONTINUITY),
 * scored against targets, with the trend across training rounds so you can see
 * the flywheel working.
 */
import { useState, useEffect } from "react";
import axios from "axios";
import { Gauge, Loader2, TrendingUp, TrendingDown } from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// metric: [value, target, lowerIsBetter, unit]
// NOTE: arc length is physically capped (~10-20f) because the ball reverses at
// every wall/floor/racket contact; do NOT chase a high value here. The real
// levers labels can move are localization (median) and precision (mean/FP).
const TARGETS = {
  median_error_px: { target: 5, lowerIsBetter: true, unit: "px", label: "Localization (median)", help: "how precisely it finds the ball — already good" },
  mean_error_px: { target: 12, lowerIsBetter: true, unit: "px", label: "Precision (mean)", help: "low = few false positives — the lever labels move" },
  mean_arc_frames: { target: 14, lowerIsBetter: false, unit: "f", label: "Continuity (arc length)", help: "physically capped ~10-20f (ball reverses at each contact) — near ceiling" },
  avg_fragments_per_window: { target: 6, lowerIsBetter: true, unit: "", label: "Fragmentation", help: "fewer broken arcs per 10s" },
};

function score(value, t) {
  if (value == null) return "unknown";
  const ok = t.lowerIsBetter ? value <= t.target : value >= t.target;
  const close = t.lowerIsBetter ? value <= t.target * 1.6 : value >= t.target * 0.6;
  return ok ? "good" : close ? "close" : "bad";
}
const COLOR = { good: "#30D158", close: "#FFA500", bad: "#FF4444", unknown: "#666" };

export default function ModelScorecard() {
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [running, setRunning] = useState(false);

  const loadHistory = () =>
    axios.get(`${API}/training/eval-history?kind=ball&limit=12`).then(r => {
      const evals = r.data.evals || [];
      setHistory(evals);
      if (evals.length) setLatest(evals[0]);
    }).catch(() => {});
  useEffect(loadHistory, []);

  const runEval = async () => {
    setRunning(true);
    try {
      const r = await axios.post(`${API}/training/eval-ball?sample_size=30`);
      setLatest(r.data);
      loadHistory();
    } catch (e) { /* noop */ } finally { setRunning(false); }
  };

  const metricRow = (key) => {
    const t = TARGETS[key];
    const val = key.startsWith("mean_arc") || key.startsWith("avg_frag")
      ? latest?.continuity?.[key] : latest?.[key];
    const st = score(val, t);
    // previous value for trend
    const prevEval = history[1];
    const prev = key.startsWith("mean_arc") || key.startsWith("avg_frag")
      ? prevEval?.continuity?.[key] : prevEval?.[key];
    const improved = val != null && prev != null &&
      (t.lowerIsBetter ? val < prev : val > prev);
    const worsened = val != null && prev != null &&
      (t.lowerIsBetter ? val > prev : val < prev);
    return (
      <div key={key} className="flex items-center gap-3 p-3 rounded-lg bg-background border border-border">
        <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: COLOR[st] }} />
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium">{t.label}</div>
          <div className="text-[11px] text-muted-foreground">{t.help} · target {t.lowerIsBetter ? "≤" : "≥"} {t.target}{t.unit}</div>
        </div>
        {improved && <TrendingUp className="w-4 h-4 text-[#30D158]" />}
        {worsened && <TrendingDown className="w-4 h-4 text-[#FF4444]" />}
        <div className="text-xl font-bold font-mono" style={{ color: COLOR[st] }}>
          {val != null ? `${val}${t.unit}` : "—"}
        </div>
      </div>
    );
  };

  return (
    <div className="stat-card rounded-lg">
      <div className="flex items-center gap-2 mb-1">
        <Gauge className="w-5 h-5 text-primary" />
        <h3 className="font-heading text-xl font-bold">Ball Model Scorecard</h3>
        <button onClick={runEval} disabled={running}
          className="ml-auto bg-primary text-primary-foreground hover:bg-primary/90 text-sm rounded px-3 py-1.5 flex items-center gap-1.5">
          {running ? <><Loader2 className="w-4 h-4 animate-spin" /> Evaluating…</> : "Run Evaluation"}
        </button>
      </div>
      <p className="text-xs text-muted-foreground mb-4">
        Honest scorecard. Localization is good and the model is near its useful ceiling for this
        footage. <span className="text-foreground font-medium">Precision</span> (fewer false positives,
        via hard negatives) is the only lever labels still move much. Arc length is physically capped
        (~10-20f) — don't chase it. The eval mean is noisy (small sample); trust the median + the trend.
      </p>

      {!latest ? (
        <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg">
          Click <span className="text-primary">Run Evaluation</span> to score the current model.
        </div>
      ) : (
        <>
          <div className="space-y-2 mb-3">
            {["median_error_px", "mean_error_px", "mean_arc_frames", "avg_fragments_per_window"].map(metricRow)}
          </div>
          <div className="text-[11px] text-muted-foreground flex items-center justify-between">
            <span>detection rate: {latest.detection_rate ?? latest?.continuity?.detection_rate ?? "—"}</span>
            <span>{history.length} evals on record · labels: {latest.total_labels_available ?? "—"}</span>
          </div>
        </>
      )}
    </div>
  );
}
