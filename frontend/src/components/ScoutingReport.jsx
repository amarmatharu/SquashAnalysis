/**
 * Scouting Report — the LLM reasoning layer's coached output.
 *
 * Renders the Claude-written narrative (markdown) when available, and always
 * shows the deterministic structured report (strengths / weaknesses / game plan)
 * as a reliable, data-grounded fallback.
 *
 * Props:
 *   data — result from /api/analysis/scouting/{id}
 */
import { Brain, Shield, Crosshair, Swords } from "lucide-react";

const P_COLORS = { 1: "#DFFF00", 2: "#00F0FF" };

// Minimal markdown renderer (headings, bold, bullets, ordered lists, paragraphs).
function renderMarkdown(md) {
  if (!md) return null;
  const lines = md.split("\n");
  const blocks = [];
  let list = null; // {type, items}

  const flush = () => {
    if (list) { blocks.push(list); list = null; }
  };

  const inline = (text) =>
    text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
      part.startsWith("**") && part.endsWith("**")
        ? <strong key={i} className="text-foreground font-semibold">{part.slice(2, -2)}</strong>
        : <span key={i}>{part}</span>
    );

  lines.forEach((raw, idx) => {
    const line = raw.trimEnd();
    if (!line.trim()) { flush(); return; }
    if (line.startsWith("## ")) {
      flush();
      blocks.push({ type: "h2", text: line.slice(3), key: idx });
    } else if (line.startsWith("# ")) {
      flush();
      blocks.push({ type: "h2", text: line.slice(2), key: idx });
    } else if (/^\s*[-*]\s+/.test(line)) {
      if (!list || list.type !== "ul") { flush(); list = { type: "ul", items: [], key: idx }; }
      list.items.push(line.replace(/^\s*[-*]\s+/, ""));
    } else if (/^\s*\d+\.\s+/.test(line)) {
      if (!list || list.type !== "ol") { flush(); list = { type: "ol", items: [], key: idx }; }
      list.items.push(line.replace(/^\s*\d+\.\s+/, ""));
    } else {
      flush();
      blocks.push({ type: "p", text: line, key: idx });
    }
  });
  flush();

  return blocks.map((b) => {
    if (b.type === "h2")
      return <h3 key={b.key} className="font-heading text-lg font-bold text-primary mt-5 mb-2">{inline(b.text)}</h3>;
    if (b.type === "p")
      return <p key={b.key} className="text-sm text-foreground/90 mb-2 leading-relaxed">{inline(b.text)}</p>;
    if (b.type === "ul")
      return <ul key={b.key} className="space-y-1 mb-3 ml-1">{b.items.map((it, i) =>
        <li key={i} className="text-sm text-foreground/90 flex gap-2"><span className="text-primary mt-0.5">•</span><span>{inline(it)}</span></li>)}</ul>;
    if (b.type === "ol")
      return <ol key={b.key} className="space-y-1 mb-3 ml-1">{b.items.map((it, i) =>
        <li key={i} className="text-sm text-foreground/90 flex gap-2"><span className="text-primary font-mono">{i + 1}.</span><span>{inline(it)}</span></li>)}</ol>;
    return null;
  });
}

export default function ScoutingReport({ data }) {
  if (!data || (!data.deterministic && !data.narrative)) return null;
  const det = data.deterministic;
  const names = det?.players ? { 1: det.players["1"]?.name, 2: det.players["2"]?.name } : {};

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Brain className="w-5 h-5 text-primary" />
        <h3 className="font-heading text-xl font-bold">Scouting Report</h3>
        <span className={`ml-auto text-[10px] px-2 py-0.5 rounded-full ${
          data.llm_used ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"}`}>
          {data.llm_used ? "Claude Opus 4.8" : "Rule-based"}
        </span>
      </div>

      {/* LLM narrative */}
      {data.narrative && (
        <div className="rounded-lg border border-primary/30 bg-primary/5 p-5">
          {renderMarkdown(data.narrative)}
        </div>
      )}

      {/* Deterministic structured report */}
      {det && (
        <div>
          <div className="text-xs uppercase tracking-wide text-muted-foreground mb-3">
            Data-grounded breakdown
          </div>
          <div className="text-sm text-foreground/80 mb-4 p-3 rounded-lg bg-background border border-border">
            {det.summary}
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            {["1", "2"].map((pid) => {
              const p = det.players?.[pid];
              if (!p) return null;
              return (
                <div key={pid} className="rounded-lg border border-border bg-background p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <div className="w-3 h-3 rounded-full" style={{ background: P_COLORS[pid] }} />
                    <span className="font-bold">{p.name}</span>
                  </div>
                  <Section icon={<Shield className="w-3.5 h-3.5 text-green-400" />} label="Strengths" items={p.strengths} />
                  <Section icon={<Crosshair className="w-3.5 h-3.5 text-red-400" />} label="Weaknesses" items={p.weaknesses} />
                  <Section icon={<Swords className="w-3.5 h-3.5 text-primary" />} label={`Game plan vs ${p.name}`} items={p.gameplan_against} />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function Section({ icon, label, items }) {
  return (
    <div className="mb-3">
      <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground mb-1">
        {icon} {label}
      </div>
      <ul className="space-y-1">
        {items.map((it, i) => (
          <li key={i} className="text-xs text-foreground/85 flex gap-1.5">
            <span className="text-muted-foreground">–</span><span>{it}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
