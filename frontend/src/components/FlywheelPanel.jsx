/**
 * Data Flywheel panel — the moat made visible.
 * Shows every human-provided label across the library (what trains the models),
 * what each source trains, and the retrain triggers. Growth here = the system
 * getting smarter over time.
 */
import { useState, useEffect } from "react";
import axios from "axios";
import { Database, Target, ListChecks, Crosshair, RefreshCw } from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export default function FlywheelPanel() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = () => {
    setLoading(true);
    axios.get(`${API}/training/flywheel`).then(r => setData(r.data)).catch(() => {}).finally(() => setLoading(false));
  };
  useEffect(load, []);

  if (loading && !data) return <div className="text-sm text-muted-foreground py-6 text-center">Loading flywheel…</div>;
  if (!data) return null;

  const sources = [
    { key: "ball_position_labels", label: "Ball position labels", icon: <Crosshair className="w-4 h-4" />, color: "#DFFF00" },
    { key: "rally_outcome_tags", label: "Rally outcome tags", icon: <ListChecks className="w-4 h-4" />, color: "#00F0FF" },
    { key: "shot_type_corrections", label: "Shot-type corrections", icon: <Target className="w-4 h-4" />, color: "#FF7A00" },
  ];

  return (
    <div className="stat-card rounded-lg">
      <div className="flex items-center gap-2 mb-1">
        <Database className="w-5 h-5 text-primary" />
        <h3 className="font-heading text-xl font-bold">Data Flywheel</h3>
        <button onClick={load} className="ml-auto text-muted-foreground hover:text-foreground">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>
      <p className="text-xs text-muted-foreground mb-4">
        Every label you create trains the models. The more you tag, the better perception gets — and
        the whole 3D analysis sharpens with it.
      </p>

      <div className="text-center py-4 mb-4 bg-background rounded-lg border border-border">
        <div className="text-4xl font-bold font-mono text-primary">{data.total_human_labels}</div>
        <div className="text-xs text-muted-foreground mt-1">total human labels (the moat)</div>
      </div>

      <div className="space-y-2 mb-4">
        {sources.map(s => (
          <div key={s.key} className="flex items-center gap-3 p-2.5 rounded-lg bg-background border border-border">
            <span style={{ color: s.color }}>{s.icon}</span>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium">{s.label}</div>
              <div className="text-[11px] text-muted-foreground truncate">{data.trains?.[s.key]}</div>
            </div>
            <div className="text-xl font-bold font-mono" style={{ color: s.color }}>
              {data.by_source?.[s.key] ?? 0}
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-2 text-center mb-3">
        <Ctx label="Matches tagged" value={data.context?.matches_with_outcomes} />
        <Ctx label="Players named" value={data.context?.matches_player_identified} />
        <Ctx label="Courts calibrated" value={data.context?.matches_calibrated} />
      </div>

      <div className="text-[11px] text-muted-foreground border-t border-border pt-2">
        <span className="text-foreground font-medium">Drift guard:</span> {data.drift_guard}
      </div>
    </div>
  );
}

function Ctx({ label, value }) {
  return (
    <div className="rounded bg-muted/30 py-2">
      <div className="text-lg font-bold font-mono">{value ?? 0}</div>
      <div className="text-[10px] text-muted-foreground">{label}</div>
    </div>
  );
}
