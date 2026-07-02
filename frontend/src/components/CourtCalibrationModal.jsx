import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";
import { toast } from "sonner";
import { Button } from "./ui/button";
import { Loader2, RotateCcw, Check, ChevronLeft, ChevronRight } from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// The 6 points the user must click, in order.
const STEPS = [
  {
    key: "front_left",
    label: "Front-left floor corner",
    desc: "Where the front wall meets the left side wall at floor level",
    color: "#DFFF00",
    group: "floor",
  },
  {
    key: "front_right",
    label: "Front-right floor corner",
    desc: "Where the front wall meets the right side wall at floor level",
    color: "#DFFF00",
    group: "floor",
  },
  {
    key: "back_right",
    label: "Back-right floor corner",
    desc: "Where the back wall meets the right side wall at floor level (near camera)",
    color: "#DFFF00",
    group: "floor",
  },
  {
    key: "back_left",
    label: "Back-left floor corner",
    desc: "Where the back wall meets the left side wall at floor level (near camera)",
    color: "#DFFF00",
    group: "floor",
  },
  {
    key: "tin_left",
    label: "Tin line — left end",
    desc: "The left end of the tin strip at the bottom of the front wall (red/metal line)",
    color: "#FF4444",
    group: "tin",
  },
  {
    key: "tin_right",
    label: "Tin line — right end",
    desc: "The right end of the tin strip at the bottom of the front wall",
    color: "#FF4444",
    group: "tin",
  },
];

const FLOOR_ORDER = ["front_left", "front_right", "back_right", "back_left"];

export default function CourtCalibrationModal({ matchId, onClose, onSaved }) {
  const [frameT, setFrameT] = useState(5);
  const [frameUrl, setFrameUrl] = useState(null);
  const [frameLoading, setFrameLoading] = useState(false);
  const [imgSize, setImgSize] = useState({ w: 1, h: 1 }); // natural size of displayed img
  const [points, setPoints] = useState({}); // key -> [nx, ny] normalized
  const [step, setStep] = useState(0);      // which STEP we're waiting for
  const [saving, setSaving] = useState(false);
  const [existing, setExisting] = useState(null);
  const [overlayUrl, setOverlayUrl] = useState(null);   // 3D verification image
  const [quality, setQuality] = useState(null);          // {q, err}
  const imgRef = useRef(null);

  const fetchFrame = useCallback(async (t) => {
    setFrameLoading(true);
    setFrameUrl(`${API}/matches/${matchId}/frame?t=${t}&_=${Date.now()}`);
    setFrameLoading(false);
  }, [matchId]);

  useEffect(() => {
    // Load existing calibration if any
    axios.get(`${API}/matches/${matchId}/calibrate`).then(r => {
      if (r.data?.calibrated) {
        setExisting(r.data.calibration);
        setPoints(r.data.calibration);
        setStep(STEPS.length); // all done
      }
    }).catch(() => {});
    fetchFrame(5);
  }, [matchId, fetchFrame]);

  const handleImgClick = (e) => {
    if (step >= STEPS.length) return;
    const rect = imgRef.current.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const key = STEPS[step].key;
    setPoints(prev => ({ ...prev, [key]: [nx, ny] }));
    setStep(s => s + 1);
  };

  const undo = () => {
    if (step === 0) return;
    const key = STEPS[step - 1].key;
    setPoints(prev => { const n = { ...prev }; delete n[key]; return n; });
    setStep(s => s - 1);
  };

  const reset = () => { setPoints({}); setStep(0); };

  const save = async () => {
    if (step < STEPS.length) { toast.error("Click all 6 points first"); return; }
    setSaving(true);
    try {
      const r = await axios.post(`${API}/matches/${matchId}/set-court`, points);
      setExisting(points);
      // Show the 3D overlay so the user can visually verify before finishing.
      setQuality({ q: r.data?.calibration_quality, err: r.data?.reproj_err_px });
      setOverlayUrl(`${API}/matches/${matchId}/court-overlay?t=${frameT}&_=${Date.now()}`);
      const good = r.data?.calibration_quality === "excellent" || r.data?.calibration_quality === "good";
      if (good) toast.success(`Calibration ${r.data.calibration_quality} (reproj ${r.data.reproj_err_px}px)`);
      else toast.warning(`Calibration looks ${r.data?.calibration_quality || "off"} — check the overlay`);
    } catch (e) {
      toast.error("Could not save calibration");
    } finally {
      setSaving(false);
    }
  };

  // SVG overlay helpers
  const pct = ([nx, ny]) => ({ x: `${(nx * 100).toFixed(2)}%`, y: `${(ny * 100).toFixed(2)}%` });

  const floorPts = FLOOR_ORDER.map(k => points[k]).filter(Boolean);
  const tinL = points.tin_left;
  const tinR = points.tin_right;

  const done = step >= STEPS.length;
  const current = STEPS[step];

  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="bg-card border border-border rounded-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div>
            <h2 className="text-lg font-bold font-heading">Court Calibration</h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              Mark 4 floor corners + the tin line so the system can detect rallies, tin hits, and service box serves.
            </p>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-xl leading-none px-2">✕</button>
        </div>

        <div className="p-4 space-y-4">
          {/* Frame time scrubber */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground w-20">Frame at {frameT}s</span>
            <input type="range" min={2} max={60} value={frameT}
              onChange={e => setFrameT(Number(e.target.value))}
              onMouseUp={() => fetchFrame(frameT)}
              onTouchEnd={() => fetchFrame(frameT)}
              className="flex-1 accent-primary" />
            <button onClick={() => fetchFrame(frameT)}
              className="text-xs px-2 py-1 bg-muted rounded hover:bg-muted/80">Refresh</button>
          </div>

          {/* Instruction banner */}
          <div className={`rounded-lg px-4 py-2.5 text-sm border ${done ? "bg-green-900/20 border-green-700/40 text-green-400" : "bg-primary/10 border-primary/30 text-foreground"}`}>
            {done ? (
              <span className="flex items-center gap-2"><Check className="w-4 h-4" /> All 6 points marked — review the overlay then Save.</span>
            ) : (
              <>
                <span className="font-semibold text-primary">Step {step + 1}/6 — </span>
                <span style={{ color: current.color }}>{current.label}</span>
                <span className="text-muted-foreground ml-2 text-xs">{current.desc}</span>
              </>
            )}
          </div>

          {/* Image + click area */}
          <div className="relative select-none" style={{ cursor: done ? "default" : "crosshair" }}>
            {frameLoading && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/60 rounded-lg z-10">
                <Loader2 className="w-6 h-6 animate-spin text-primary" />
              </div>
            )}
            {frameUrl && (
              <>
                <img
                  ref={imgRef}
                  src={frameUrl}
                  alt="Court frame"
                  className="w-full rounded-lg block"
                  onClick={handleImgClick}
                  onLoad={e => setImgSize({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
                  draggable={false}
                />
                {/* SVG overlay */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                  {/* Floor quad */}
                  {floorPts.length === 4 && (
                    <polygon
                      points={FLOOR_ORDER.map(k => {
                        const p = points[k];
                        return `${(p[0]*100).toFixed(2)},${(p[1]*100).toFixed(2)}`;
                      }).join(" ")}
                      fill="rgba(223,255,0,0.08)" stroke="#DFFF00" strokeWidth="0.4" strokeDasharray="1,1"
                    />
                  )}
                  {/* Tin line */}
                  {tinL && tinR && (
                    <line
                      x1={`${(tinL[0]*100).toFixed(2)}%`} y1={`${(tinL[1]*100).toFixed(2)}%`}
                      x2={`${(tinR[0]*100).toFixed(2)}%`} y2={`${(tinR[1]*100).toFixed(2)}%`}
                      stroke="#FF4444" strokeWidth="0.6"
                    />
                  )}
                  {/* Dots for each marked point */}
                  {STEPS.map(({ key, color, label }, i) => {
                    const p = points[key];
                    if (!p) return null;
                    return (
                      <g key={key}>
                        <circle cx={`${(p[0]*100).toFixed(2)}%`} cy={`${(p[1]*100).toFixed(2)}%`}
                          r="1.2" fill={color} stroke="black" strokeWidth="0.3" />
                        <text x={`${(p[0]*100).toFixed(2)}%`} y={`${(p[1]*100 - 1.8).toFixed(2)}%`}
                          fill={color} fontSize="2.2" textAnchor="middle"
                          style={{ fontFamily: "monospace", pointerEvents: "none" }}>
                          {i + 1}
                        </text>
                      </g>
                    );
                  })}
                  {/* Cursor label for next point */}
                  {!done && (
                    <text x="50%" y="96%" fill={current.color} fontSize="2.5" textAnchor="middle"
                      style={{ fontFamily: "monospace" }}>
                      ↑ click {current.label}
                    </text>
                  )}
                </svg>
              </>
            )}
            {!frameUrl && !frameLoading && (
              <div className="h-64 flex items-center justify-center text-muted-foreground text-sm border border-dashed border-border rounded-lg">
                Loading frame…
              </div>
            )}
          </div>

          {/* Point list */}
          <div className="grid grid-cols-3 gap-2">
            {STEPS.map(({ key, label, color }, i) => {
              const marked = !!points[key];
              const isCurrent = i === step;
              return (
                <div key={key} className={`flex items-center gap-2 rounded px-2 py-1.5 text-xs border ${
                  marked ? "border-border bg-background" : isCurrent ? "border-primary/40 bg-primary/5" : "border-border/30 bg-background/30"
                }`}>
                  <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0 ${
                    marked ? "bg-green-700 text-white" : isCurrent ? "bg-primary text-black" : "bg-muted text-muted-foreground"
                  }`}>{i + 1}</span>
                  <span className={marked ? "text-foreground" : "text-muted-foreground"}>{label}</span>
                </div>
              );
            })}
          </div>

          {/* 3D verification overlay — shows after save */}
          {overlayUrl && (
            <div className="mb-4 rounded-lg border border-border p-3 bg-background">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-semibold">3D check</span>
                {quality && (
                  <span className={`text-xs px-2 py-0.5 rounded-full ${
                    quality.q === "excellent" || quality.q === "good"
                      ? "bg-green-900/30 text-green-400"
                      : quality.q === "rough" ? "bg-amber-900/30 text-amber-400"
                      : "bg-red-900/30 text-red-400"}`}>
                    {quality.q}{quality.err != null ? ` · reproj ${quality.err}px` : ""}
                  </span>
                )}
                <span className="text-[11px] text-muted-foreground ml-auto">
                  The drawn lines should sit on the real tin, out-lines and floor.
                </span>
              </div>
              <img src={overlayUrl} alt="3D court overlay" className="w-full rounded" />
              {quality && quality.q !== "excellent" && quality.q !== "good" && (
                <p className="text-xs text-amber-400 mt-2">
                  Lines don't line up? Re-click the corners more precisely — front corners where the
                  front wall meets the floor, back corners at the very bottom near the camera.
                </p>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-3 pt-2 border-t border-border">
            <button onClick={undo} disabled={step === 0}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground disabled:opacity-40 px-2 py-1.5 rounded hover:bg-muted">
              <ChevronLeft className="w-3 h-3" /> Undo last
            </button>
            <button onClick={reset}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground px-2 py-1.5 rounded hover:bg-muted">
              <RotateCcw className="w-3 h-3" /> Reset all
            </button>
            <div className="ml-auto flex gap-3">
              <Button variant="outline" onClick={onClose}>Cancel</Button>
              {overlayUrl ? (
                <>
                  <Button variant="outline" onClick={() => { setOverlayUrl(null); setQuality(null); }}>
                    Re-adjust
                  </Button>
                  <Button onClick={() => { onSaved && onSaved(points); }}
                    className="bg-primary text-primary-foreground hover:bg-primary/90">
                    <Check className="w-4 h-4 mr-2" /> Done
                  </Button>
                </>
              ) : (
                <Button onClick={save} disabled={!done || saving}
                  className="bg-primary text-primary-foreground hover:bg-primary/90">
                  {saving ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Saving…</> : <><Check className="w-4 h-4 mr-2" /> Save Calibration</>}
                </Button>
              )}
            </div>
          </div>

          {/* Court diagram reference */}
          <details className="text-xs text-muted-foreground">
            <summary className="cursor-pointer hover:text-foreground">Court layout reference</summary>
            <div className="mt-2 p-3 bg-background rounded border border-border">
              <svg viewBox="0 0 64 97.5" className="w-48 mx-auto" style={{ border: "1px solid #333" }}>
                {/* Court outline */}
                <rect x="0" y="0" width="64" height="97.5" fill="#1a1a1a" stroke="#555" strokeWidth="0.5"/>
                {/* Short line */}
                <line x1="0" y1="54.9" x2="64" y2="54.9" stroke="#888" strokeWidth="0.5"/>
                {/* Half court line */}
                <line x1="32" y1="54.9" x2="32" y2="97.5" stroke="#888" strokeWidth="0.5"/>
                {/* Service boxes */}
                <rect x="0" y="54.9" width="16" height="16" fill="none" stroke="#555" strokeWidth="0.4"/>
                <rect x="48" y="54.9" width="16" height="16" fill="none" stroke="#555" strokeWidth="0.4"/>
                {/* T */}
                <circle cx="32" cy="54.9" r="1.5" fill="#DFFF00"/>
                {/* Tin line (front wall) */}
                <line x1="0" y1="4.8" x2="64" y2="4.8" stroke="#FF4444" strokeWidth="1"/>
                {/* Labels */}
                <text x="32" y="3" fill="#FF4444" fontSize="3" textAnchor="middle">TIN</text>
                <text x="32" y="56.5" fill="#888" fontSize="3" textAnchor="middle">SHORT LINE / T</text>
                <text x="8" y="64" fill="#888" fontSize="3" textAnchor="middle">L box</text>
                <text x="56" y="64" fill="#888" fontSize="3" textAnchor="middle">R box</text>
                <text x="32" y="80" fill="#555" fontSize="3" textAnchor="middle">BACK (camera)</text>
                {/* Corner labels */}
                <text x="2" y="8" fill="#DFFF00" fontSize="3">FL</text>
                <text x="54" y="8" fill="#DFFF00" fontSize="3">FR</text>
                <text x="2" y="95" fill="#DFFF00" fontSize="3">BL</text>
                <text x="54" y="95" fill="#DFFF00" fontSize="3">BR</text>
              </svg>
              <p className="mt-2 text-center text-[10px] text-muted-foreground">
                FL/FR = front corners (near front wall) · BL/BR = back corners (near camera)
              </p>
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}
