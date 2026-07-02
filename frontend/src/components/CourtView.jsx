/**
 * Top-down court diagram showing live player positions.
 * Draws a standard squash singles court with the T, service boxes,
 * short line, and two player dots colour-coded by ID.
 *
 * Props:
 *   positions  — array of { t, players: [{id, feet_x, feet_y, ...}] }
 *   stats      — { "1": {...}, "2": {...} }
 *   frameW     — native video width used for normalization (default 1280)
 *   frameH     — native video height (default 720)
 *   calibration — optional { front_left, front_right, back_left, back_right } (normalized 0-1)
 */
import { useState, useEffect, useRef } from "react";

const COURT_W = 6.4;   // metres
const COURT_L = 9.75;
const SHORT_Y = 5.49;
const SERVICE_BOX = 1.6;
const T = { x: 3.2, y: SHORT_Y };

const P_COLORS = { 1: "#DFFF00", 2: "#00F0FF" };

// Map pixel feet position to normalized court coords (0..1) using a simple
// perspective correction based on the 4 calibration corners.
// Without calibration, falls back to a rough linear approximation.
function pixelToNorm(px, py, calib, frameW = 1280, frameH = 720) {
  if (!calib) {
    // fallback: assume camera is roughly centred and behind the court
    return { x: px / frameW, y: py / frameH };
  }
  // bilinear interpolation using the 4 court corners
  const { front_left: fl, front_right: fr, back_left: bl, back_right: br } = calib;
  // Normalize px/py to 0-1
  const nx = px / frameW;
  const ny = py / frameH;
  // Estimate court coords via inverse bilinear (simplified)
  // u = left-right fraction, v = front-back fraction
  const topX = fl[0] + (fr[0] - fl[0]) * 0.5;   // midpoint top
  const botX = bl[0] + (br[0] - bl[0]) * 0.5;   // midpoint bottom
  const topY = (fl[1] + fr[1]) / 2;
  const botY = (bl[1] + br[1]) / 2;

  const v = topY < botY
    ? Math.max(0, Math.min(1, (ny - topY) / (botY - topY)))
    : Math.max(0, Math.min(1, (ny - botY) / (topY - botY)));

  const leftEdge = fl[0] + (bl[0] - fl[0]) * v;
  const rightEdge = fr[0] + (br[0] - fr[0]) * v;
  const u = rightEdge > leftEdge
    ? Math.max(0, Math.min(1, (nx - leftEdge) / (rightEdge - leftEdge)))
    : nx;

  return { x: u, y: v };
}

// Convert court metres to SVG percentage coords
// Court is drawn with front wall at top (y=0), back wall at bottom (y=COURT_L)
function courtToSvg(cx, cy) {
  return {
    x: `${(cx / COURT_W * 100).toFixed(2)}%`,
    y: `${(cy / COURT_L * 100).toFixed(2)}%`,
  };
}

function normToSvg(nx, ny) {
  // In phone footage, front wall is at top of image, back wall at bottom
  // nx=0 is left wall, nx=1 is right wall
  // ny=0 is front wall, ny=1 is back wall (near camera)
  return courtToSvg(nx * COURT_W, ny * COURT_L);
}

export default function CourtView({ positions = [], stats = {}, frameW = 1280, frameH = 720, calibration = null, names = { 1: "Player 1", 2: "Player 2" } }) {
  const nm = (pid) => names[pid] || `Player ${pid}`;
  const [tick, setTick] = useState(0);
  const tickRef = useRef(0);
  const rafRef = useRef(null);

  // Animate through positions at ~10fps playback
  useEffect(() => {
    if (!positions.length) return;
    const step = () => {
      tickRef.current = (tickRef.current + 1) % positions.length;
      setTick(tickRef.current);
      rafRef.current = setTimeout(step, 100);
    };
    rafRef.current = setTimeout(step, 100);
    return () => clearTimeout(rafRef.current);
  }, [positions]);

  const current = positions[tick] || null;

  const playerDots = current
    ? current.players.map(p => {
        const norm = pixelToNorm(p.feet_x, p.feet_y, calibration, frameW, frameH);
        const svg = normToSvg(norm.x, norm.y);
        return { id: p.id, ...svg, conf: p.conf };
      })
    : [];

  // Heatmap: count time spent in each court zone per player
  // Use coarser 6x10 grid, normalize to 0..1
  const heatmap = { 1: {}, 2: {} };
  const GRID = 8;
  for (const frame of positions) {
    for (const p of frame.players) {
      const norm = pixelToNorm(p.feet_x, p.feet_y, calibration, frameW, frameH);
      const gx = Math.min(GRID - 1, Math.floor(norm.x * GRID));
      const gy = Math.min(GRID - 1, Math.floor(norm.y * GRID));
      const key = `${gx},${gy}`;
      heatmap[p.id] = heatmap[p.id] || {};
      heatmap[p.id][key] = (heatmap[p.id][key] || 0) + 1;
    }
  }

  return (
    <div className="space-y-4">
      {/* Stats cards */}
      <div className="grid grid-cols-2 gap-3">
        {[1, 2].map(pid => {
          const s = stats[String(pid)] || {};
          return (
            <div key={pid} className="rounded-lg border border-border bg-background p-3">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-3 h-3 rounded-full" style={{ background: P_COLORS[pid] }} />
                <span className="text-sm font-semibold">{nm(pid)}</span>
                <span className="text-xs text-muted-foreground ml-auto">
                  {s.frames_detected || 0} frames
                </span>
              </div>
              <div className="space-y-1">
                {s.t_control_pct != null && s.t_control_pct > 0 ? (
                  <>
                    <StatRow label="T-control" value={`${s.t_control_pct}%`} />
                    <StatRow label="Service box" value={`${s.service_box_pct}%`} />
                    <StatRow label="Front court" value={`${s.zone_pct?.front}%`} />
                    <StatRow label="Mid court" value={`${s.zone_pct?.mid}%`} />
                    <StatRow label="Back court" value={`${s.zone_pct?.back}%`} />
                  </>
                ) : (
                  <p className="text-xs text-muted-foreground">Calibrate the court to see court-metre stats.</p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Court diagram */}
      <div className="rounded-lg border border-border bg-background p-3">
        <div className="text-xs text-muted-foreground mb-2 flex items-center justify-between">
          <span>Top-down court view</span>
          {current && <span className="font-mono">t = {current.t?.toFixed(1)}s</span>}
        </div>
        <div className="flex gap-4">
          {/* Animated position court */}
          <div className="flex-1">
            <div className="text-[10px] text-muted-foreground mb-1 text-center">Live positions</div>
            <svg viewBox="0 0 100 156" className="w-full max-w-[160px] mx-auto border border-border rounded"
              style={{ background: "#0a0a0a" }}>
              <CourtLines />
              {playerDots.map(({ id, x, y }) => (
                <g key={id}>
                  <circle cx={x} cy={y} r="4" fill={P_COLORS[id]} opacity="0.9" />
                  <text x={x} y={y} dy="1.5" fontSize="4" textAnchor="middle"
                    fill="black" fontWeight="bold">{id}</text>
                </g>
              ))}
            </svg>
          </div>

          {/* Heatmap per player */}
          {[1, 2].map(pid => {
            const h = heatmap[pid] || {};
            const maxVal = Math.max(1, ...Object.values(h));
            return (
              <div key={pid} className="flex-1">
                <div className="text-[10px] mb-1 text-center" style={{ color: P_COLORS[pid] }}>
                  {nm(pid)} heatmap
                </div>
                <svg viewBox="0 0 100 156" className="w-full max-w-[160px] mx-auto border border-border rounded"
                  style={{ background: "#0a0a0a" }}>
                  {Array.from({ length: GRID }, (_, gy) =>
                    Array.from({ length: GRID }, (_, gx) => {
                      const key = `${gx},${gy}`;
                      const val = h[key] || 0;
                      const alpha = val / maxVal;
                      const color = pid === 1 ? `rgba(223,255,0,${alpha * 0.7})` : `rgba(0,240,255,${alpha * 0.7})`;
                      return (
                        <rect key={key}
                          x={`${gx / GRID * 100}%`} y={`${gy / GRID * 100}%`}
                          width={`${100 / GRID}%`} height={`${100 / GRID}%`}
                          fill={color} />
                      );
                    })
                  )}
                  <CourtLines />
                </svg>
              </div>
            );
          })}
        </div>
        <p className="text-[10px] text-muted-foreground mt-2 text-center">
          Front wall at top · back wall (camera side) at bottom
        </p>
      </div>
    </div>
  );
}

function StatRow({ label, value }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono text-foreground">{value}</span>
    </div>
  );
}

// Reusable SVG court lines (100x156 viewBox, front wall at top)
function CourtLines() {
  const W = 100, L = 156;
  const shortY = (SHORT_Y / COURT_L * L).toFixed(1);
  const midX = (W / 2).toFixed(1);
  const sbox = (SERVICE_BOX / COURT_W * W).toFixed(1);
  return (
    <g stroke="#333" strokeWidth="0.8" fill="none">
      {/* Outer boundary */}
      <rect x="0" y="0" width={W} height={L} stroke="#555" />
      {/* Short line */}
      <line x1="0" y1={shortY} x2={W} y2={shortY} />
      {/* Half-court line */}
      <line x1={midX} y1={shortY} x2={midX} y2={L} />
      {/* Service boxes */}
      <rect x="0" y={shortY} width={sbox} height={sbox} stroke="#444" />
      <rect x={W - sbox} y={shortY} width={sbox} height={sbox} stroke="#444" />
      {/* T mark */}
      <circle cx={midX} cy={shortY} r="2" fill="#555" stroke="none" />
      {/* Tin line (front wall) */}
      <line x1="0" y1="3" x2={W} y2="3" stroke="#FF4444" strokeWidth="1.5" />
      {/* Labels */}
      <text x={W / 2} y="1.8" fontSize="3" fill="#FF4444" textAnchor="middle">TIN</text>
      <text x={W / 2} y={Number(shortY) - 1} fontSize="3" fill="#555" textAnchor="middle">T</text>
    </g>
  );
}
