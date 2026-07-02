import { useState, useEffect, useCallback, useRef } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { toast } from "sonner";
import axios from "axios";
import {
  Target, ArrowLeft, Loader2, Circle, Check, X, HelpCircle, Crosshair, Database,
  MousePointerClick, ChevronLeft, ChevronRight, Save, Eraser, ListVideo, Activity, Clock, Sparkles, Wand2
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Review candidate ball-tracks proposed by the classical detector and confirm
// which are the real ball. Confirmed tracks become the labelled dataset for a
// future TrackNet. Each track shows a filmstrip of crops centred on the
// candidate so the reviewer can judge "is this the ball flying?" at a glance.
const BallAnnotationPage = () => {
  const { matchId } = useParams();
  const navigate = useNavigate();

  // Library navigation: move between videos without leaving this page.
  const [libVideos, setLibVideos] = useState([]);
  useEffect(() => {
    axios.get(`${API}/training/library`)
      .then((r) => setLibVideos((r.data.videos || [])))
      .catch(() => {});
  }, []);
  const libIndex = libVideos.findIndex((v) => v.id === matchId);
  const prevVideo = libIndex > 0 ? libVideos[libIndex - 1] : null;
  const nextVideo = libIndex >= 0 && libIndex < libVideos.length - 1 ? libVideos[libIndex + 1] : null;
  const goToVideo = (v) => { if (v) { navigate(`/annotate-ball/${v.id}`); window.scrollTo(0, 0); } };
  const [tasks, setTasks] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [extracting, setExtracting] = useState(false);
  const [startS, setStartS] = useState(30);
  const [durationS, setDurationS] = useState(5);
  const [labels, setLabels] = useState({}); // `${taskId}:${trackId}` -> label

  // ----- manual marking mode -----
  const [mode, setMode] = useState("review"); // "review" | "manual"
  const [mFrames, setMFrames] = useState([]); // [{frame_index, timestamp, b64}]
  const [mIdx, setMIdx] = useState(0);
  const [mLoading, setMLoading] = useState(false);
  const [mStart, setMStart] = useState(30);
  const [mCount, setMCount] = useState(24);
  const [mStep, setMStep] = useState(2);
  const [nativeDims, setNativeDims] = useState({ w: 1024, h: 576 });
  const [marks, setMarks] = useState({}); // frame_index -> {nx, ny}
  const imgRef = useRef(null);

  // optical-flow propagation: one click -> a tracked strip to review/save
  const [propPoints, setPropPoints] = useState(null); // [{frame_index,nx,ny,crop_b64}]
  const [propLoading, setPropLoading] = useState(false);
  const [propExclude, setPropExclude] = useState({}); // frame_index -> true
  const [propNative, setPropNative] = useState({ w: 1024, h: 576 });

  const doPropagate = async () => {
    if (!curFrame || !curMark) { toast.error("Click the ball first"); return; }
    setPropLoading(true);
    setPropPoints(null);
    setPropExclude({});
    try {
      const res = await axios.post(`${API}/annotation/ball/propagate/${matchId}`, {
        start_frame_index: curFrame.frame_index, nx: curMark.nx, ny: curMark.ny, n_frames: 30,
      });
      setPropPoints(res.data.points || []);
      setPropNative({ w: res.data.native_width, h: res.data.native_height });
      toast.success(`Tracked ${res.data.tracked} frames from one click`);
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Propagation failed");
    } finally {
      setPropLoading(false);
    }
  };

  const savePropagated = async () => {
    const kept = (propPoints || []).filter((p) => !propExclude[p.frame_index]);
    if (!kept.length) { toast.error("No points to save"); return; }
    try {
      const res = await axios.post(`${API}/annotation/ball/manual-label?match_id=${matchId}`, {
        points: kept.map((p) => ({ frame_index: p.frame_index, nx: p.nx, ny: p.ny })),
        native_width: propNative.w, native_height: propNative.h,
      });
      toast.success(`Saved ${res.data.ball_points} ball points from one click`);
      setPropPoints(null);
      const s = await axios.get(`${API}/annotation/ball/stats`);
      setStats(s.data);
    } catch (e) {
      toast.error("Failed to save tracked points");
    }
  };

  const loadFrames = async () => {
    setMLoading(true);
    try {
      const res = await axios.get(
        `${API}/annotation/ball/frames/${matchId}?start_s=${Number(mStart)}&count=${Number(mCount)}&step=${Number(mStep)}`
      );
      setMFrames(res.data.frames || []);
      setNativeDims({ w: res.data.native_width, h: res.data.native_height });
      setMIdx(0);
      setMarks({});
      if (!res.data.frames?.length) toast.error("No frames returned for that window");
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Failed to load frames");
    } finally {
      setMLoading(false);
    }
  };

  const handleMarkClick = (e) => {
    const img = imgRef.current;
    if (!img || !mFrames[mIdx]) return;
    const rect = img.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const fi = mFrames[mIdx].frame_index;
    setMarks((p) => ({ ...p, [fi]: { nx: Math.min(1, Math.max(0, nx)), ny: Math.min(1, Math.max(0, ny)) } }));
    if (mIdx < mFrames.length - 1) setMIdx((i) => i + 1); // auto-advance
  };

  const clearMark = () => {
    const fi = mFrames[mIdx]?.frame_index;
    if (fi == null) return;
    setMarks((p) => { const n = { ...p }; delete n[fi]; return n; });
  };

  const saveMarks = async () => {
    const points = Object.entries(marks).map(([fi, m]) => ({
      frame_index: Number(fi), nx: m.nx, ny: m.ny,
    }));
    if (!points.length) { toast.error("No marks to save"); return; }
    try {
      const res = await axios.post(`${API}/annotation/ball/manual-label?match_id=${matchId}`, {
        points, native_width: nativeDims.w, native_height: nativeDims.h,
      });
      toast.success(`Saved ${res.data.ball_points} manual ball points`);
      setMarks({});
      const s = await axios.get(`${API}/annotation/ball/stats`);
      setStats(s.data);
    } catch (e) {
      toast.error("Failed to save manual marks");
    }
  };

  const curFrame = mFrames[mIdx];
  const curMark = curFrame ? marks[curFrame.frame_index] : null;
  const markedCount = Object.keys(marks).length;

  // ----- structured timeline (M2) -----
  const [tStart, setTStart] = useState(30);
  const [tDuration, setTDuration] = useState(8);
  const [tBuilding, setTBuilding] = useState(false);
  const [timelines, setTimelines] = useState([]);

  const fetchTimelines = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/analysis/timeline/${matchId}`);
      setTimelines(res.data.timelines || []);
    } catch (e) { /* ignore */ }
  }, [matchId]);

  const buildTimeline = async () => {
    setTBuilding(true);
    try {
      await axios.post(`${API}/analysis/timeline/${matchId}`, {
        start_s: Number(tStart), duration_s: Number(tDuration),
      });
      toast.success("Timeline built");
      await fetchTimelines();
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Timeline build failed");
    } finally {
      setTBuilding(false);
    }
  };

  useEffect(() => { if (mode === "timeline") fetchTimelines(); }, [mode, fetchTimelines]);

  // ----- model training (from the UI) -----
  const [trainState, setTrainState] = useState(null); // {status, epoch, epochs, loss, ...}
  const pollRef = useRef(null);

  const pollTraining = useCallback(() => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/annotation/ball/train/status`);
        setTrainState(res.data);
        if (res.data.status === "done" || res.data.status === "failed") {
          clearInterval(pollRef.current);
          if (res.data.status === "done") {
            toast.success("Training done — TrackNet is now active");
            const s = await axios.get(`${API}/annotation/ball/stats`);
            setStats(s.data);
          } else {
            toast.error("Training failed: " + (res.data.error || "unknown"));
          }
        }
      } catch (e) { /* keep polling */ }
    }, 1500);
  }, []);

  const startTraining = async () => {
    try {
      const res = await axios.post(`${API}/annotation/ball/train`, { epochs: 20, min_samples: 30 });
      toast.success(`Training started on ${res.data.samples} points`);
      setTrainState({ status: "running", epoch: 0, epochs: res.data.epochs });
      pollTraining();
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Could not start training");
    }
  };

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // ----- self-training (active learning): mine proposals, approve/reject -----
  const [stProposals, setStProposals] = useState([]);
  const [stStart, setStStart] = useState(30);
  const [stDuration, setStDuration] = useState(15);
  const [stMining, setStMining] = useState(false);

  const fetchProposals = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/selftrain/proposals/${matchId}`);
      setStProposals(r.data.proposals || []);
    } catch (e) { /* ignore */ }
  }, [matchId]);

  useEffect(() => { if (mode === "selftrain") fetchProposals(); }, [mode, fetchProposals]);

  const mineVideo = async () => {
    setStMining(true);
    try {
      const r = await axios.post(`${API}/selftrain/mine/${matchId}`, {
        start_s: Number(stStart), duration_s: Number(stDuration), min_quality: 80,
      });
      toast.success(`Model proposed ${r.data.num_proposals} ball arcs`);
      await fetchProposals();
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Mining failed");
    } finally {
      setStMining(false);
    }
  };

  const reviewProposal = async (proposalId, decision) => {
    setStProposals((p) => p.filter((x) => x.id !== proposalId)); // optimistic remove
    try {
      const r = await axios.post(`${API}/selftrain/review/${matchId}`, { proposal_id: proposalId, decision });
      toast.success(decision === "approve" ? `Approved (+${r.data.ball_points_added} pts)` : "Rejected (hard negative)");
      const s = await axios.get(`${API}/annotation/ball/stats`);
      setStats(s.data);
    } catch (e) {
      toast.error("Review failed");
      fetchProposals();
    }
  };

  const fetchTasks = useCallback(async () => {
    try {
      const [t, s] = await Promise.all([
        axios.get(`${API}/annotation/ball/tasks/${matchId}`),
        axios.get(`${API}/annotation/ball/stats`),
      ]);
      setTasks(t.data.tasks || []);
      setStats(s.data);
      // hydrate any existing labels into local state
      const seed = {};
      (t.data.tasks || []).forEach((task) =>
        (task.labels || []).forEach((l) => { seed[`${task.id}:${l.track_id}`] = l.label; })
      );
      setLabels((prev) => ({ ...seed, ...prev }));
    } catch (e) {
      toast.error("Failed to load annotation tasks");
    } finally {
      setLoading(false);
    }
  }, [matchId]);

  useEffect(() => { fetchTasks(); }, [fetchTasks]);

  // Switching videos: clear per-video working state so nothing carries over.
  useEffect(() => {
    setTasks([]); setMFrames([]); setMIdx(0); setMarks({});
    setPropPoints(null); setPropExclude({}); setTimelines([]); setStProposals([]);
  }, [matchId]);

  const handleExtract = async () => {
    setExtracting(true);
    try {
      await axios.post(`${API}/annotation/ball/extract/${matchId}`, {
        start_s: Number(startS), duration_s: Number(durationS), max_tracks: 12,
      });
      toast.success("Candidates extracted");
      await fetchTasks();
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Extraction failed");
    } finally {
      setExtracting(false);
    }
  };

  const handleLabel = async (taskId, trackId, label) => {
    const key = `${taskId}:${trackId}`;
    setLabels((p) => ({ ...p, [key]: label }));
    try {
      const res = await axios.post(`${API}/annotation/ball/label?match_id=${matchId}`, {
        task_id: taskId, track_id: trackId, label,
      });
      toast.success(`Saved: ${label}${res.data.ball_points ? ` (${res.data.ball_points} pts)` : ""}`);
      const s = await axios.get(`${API}/annotation/ball/stats`);
      setStats(s.data);
    } catch (e) {
      toast.error("Failed to save label");
      setLabels((p) => { const n = { ...p }; delete n[key]; return n; });
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#050505] flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#050505]">
      <nav className="border-b border-border/50 bg-background/80 backdrop-blur-xl sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
              <Target className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-heading text-xl font-bold tracking-tight">SQUASHSENSE</span>
          </Link>
          <div className="flex items-center gap-2">
            <Link to="/training-library">
              <Button variant="ghost" className="text-muted-foreground">
                <ListVideo className="w-4 h-4 mr-2" /> Library
              </Button>
            </Link>
            <Button variant="outline" size="sm" className="border-border"
              onClick={() => goToVideo(prevVideo)} disabled={!prevVideo} title={prevVideo?.title}>
              <ChevronLeft className="w-4 h-4" /> Prev
            </Button>
            {libIndex >= 0 && (
              <span className="text-xs text-muted-foreground font-mono w-14 text-center">
                {libIndex + 1}/{libVideos.length}
              </span>
            )}
            <Button variant="outline" size="sm" className="border-border"
              onClick={() => goToVideo(nextVideo)} disabled={!nextVideo} title={nextVideo?.title}>
              Next <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </nav>

      <div className="max-w-5xl mx-auto px-6 py-8">
        <div className="mb-6">
          <h1 className="font-heading text-3xl font-black tracking-tight mb-2">
            BALL <span className="text-primary">LABELLING</span>
          </h1>
          <p className="text-muted-foreground max-w-2xl">
            The classical detector proposes ball-like tracks; confirm which are the real ball.
            Each <span className="text-primary">confirmed</span> track becomes ground-truth data
            to train the squash ball model. Look at each filmstrip: a real ball moves smoothly and
            fast across frames — reflections and racket motion don't.
          </p>
        </div>

        {/* Mode toggle */}
        <div className="flex gap-2 mb-6">
          <Button variant={mode === "review" ? "default" : "outline"}
            onClick={() => setMode("review")}
            className={mode === "review" ? "bg-primary text-primary-foreground" : "border-border"}>
            <ListVideo className="w-4 h-4 mr-2" /> Review candidates
          </Button>
          <Button variant={mode === "manual" ? "default" : "outline"}
            onClick={() => setMode("manual")}
            className={mode === "manual" ? "bg-primary text-primary-foreground" : "border-border"}>
            <MousePointerClick className="w-4 h-4 mr-2" /> Manual mark
          </Button>
          <Button variant={mode === "selftrain" ? "default" : "outline"}
            onClick={() => setMode("selftrain")}
            className={mode === "selftrain" ? "bg-primary text-primary-foreground" : "border-border"}>
            <Wand2 className="w-4 h-4 mr-2" /> Auto-label
          </Button>
          <Button variant={mode === "timeline" ? "default" : "outline"}
            onClick={() => setMode("timeline")}
            className={mode === "timeline" ? "bg-primary text-primary-foreground" : "border-border"}>
            <Activity className="w-4 h-4 mr-2" /> Timeline
          </Button>
        </div>

        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-3 gap-3 mb-6">
            <div className="bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-2 text-primary mb-1"><Database className="w-4 h-4" />
                <span className="text-2xl font-bold font-mono">{stats.confirmed_ball_points}</span>
              </div>
              <p className="text-xs text-muted-foreground">confirmed ball points</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <span className="text-2xl font-bold font-mono text-[#30D158]">{stats.tracks_labeled_by_type?.ball || 0}</span>
              <p className="text-xs text-muted-foreground">tracks confirmed as ball</p>
            </div>
            <div className="bg-card border border-border rounded-lg p-4">
              <span className="text-2xl font-bold font-mono">{stats.tasks_extracted}</span>
              <p className="text-xs text-muted-foreground">extraction runs</p>
            </div>
          </div>
        )}

        {/* ===================== TRAIN MODEL ===================== */}
        {stats && (
          <div className="bg-card border border-border rounded-lg p-4 mb-6">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <Database className="w-4 h-4 text-primary" />
                  <span className="font-heading font-bold text-sm">Train ball model (TrackNet)</span>
                  {stats.model_active ? (
                    <span className="text-[11px] px-2 py-0.5 rounded-full bg-[#30D158]/20 text-[#30D158]">model active</span>
                  ) : (
                    <span className="text-[11px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground">using classical fallback</span>
                  )}
                </div>
                {/* progress toward the recommended amount */}
                <div className="w-64 h-2 bg-background rounded-full overflow-hidden mb-1">
                  <div className="h-full bg-primary"
                    style={{ width: `${Math.min(100, (stats.confirmed_ball_points / (stats.recommended_min_points || 200)) * 100)}%` }} />
                </div>
                <p className="text-xs text-muted-foreground">
                  {stats.confirmed_ball_points} / {stats.recommended_min_points || 200} points recommended
                  {stats.confirmed_ball_points < (stats.recommended_min_points || 200) && " — you can train early to see it work, but it'll be weak until you label more"}
                </p>
              </div>
              <div className="text-right">
                <Button onClick={startTraining}
                  disabled={trainState?.status === "running"}
                  className="bg-primary text-primary-foreground hover:bg-primary/90">
                  {trainState?.status === "running"
                    ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Training…</>
                    : <><Database className="w-4 h-4 mr-2" /> Train model now</>}
                </Button>
                {trainState?.status === "running" && (
                  <p className="text-xs text-muted-foreground mt-2 font-mono">
                    epoch {trainState.epoch}/{trainState.epochs}
                    {trainState.loss != null && ` · loss ${trainState.loss}`}
                  </p>
                )}
                {trainState?.status === "done" && trainState.report && (
                  <p className="text-xs text-[#30D158] mt-2 font-mono">
                    ✓ trained on {trainState.report.samples} pts · loss {trainState.report.final_loss?.toFixed?.(4)}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ===================== REVIEW MODE ===================== */}
        {mode === "review" && (<>
        {/* Extract controls */}
        <div className="bg-card border border-border rounded-lg p-4 mb-6 flex items-end gap-4 flex-wrap">
          <div className="flex items-center gap-2"><Crosshair className="w-5 h-5 text-primary" />
            <span className="font-heading font-bold text-sm">Extract candidates</span></div>
          <label className="text-xs text-muted-foreground">
            Start (s)
            <input type="number" value={startS} onChange={(e) => setStartS(e.target.value)}
              className="block w-24 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" />
          </label>
          <label className="text-xs text-muted-foreground">
            Duration (s)
            <input type="number" value={durationS} onChange={(e) => setDurationS(e.target.value)}
              className="block w-24 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" />
          </label>
          <Button onClick={handleExtract} disabled={extracting}
            className="bg-primary text-primary-foreground hover:bg-primary/90">
            {extracting ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Extracting…</> : "Extract"}
          </Button>
        </div>

        {/* Tasks */}
        {tasks.length === 0 ? (
          <div className="text-center text-muted-foreground py-16 border border-dashed border-border rounded-lg">
            No candidates yet. Pick a window with a rally and click <span className="text-primary">Extract</span>.
          </div>
        ) : (
          tasks.map((task) => (
            <div key={task.id} className="mb-8">
              <div className="flex items-center gap-2 mb-3 text-sm text-muted-foreground">
                <Circle className="w-3 h-3 fill-primary text-primary" />
                Window {task.start_s}s–{(task.start_s + task.duration_s).toFixed(0)}s ·
                {task.num_tracks} candidate tracks
              </div>
              <div className="space-y-3">
                {task.tracks.map((track) => {
                  const key = `${task.id}:${track.track_id}`;
                  const chosen = labels[key];
                  return (
                    <div key={track.track_id}
                      className={`bg-card border rounded-lg p-3 ${
                        chosen === "ball" ? "border-[#30D158]" :
                        chosen === "not_ball" ? "border-[#FF3B30]/50" :
                        chosen === "unsure" ? "border-[#FF9F0A]/50" : "border-border"}`}>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-mono text-muted-foreground">
                          track #{track.track_id} · quality {track.quality} · {track.num_points} pts
                        </span>
                        <div className="flex gap-2">
                          <Button size="sm" variant="outline" onClick={() => handleLabel(task.id, track.track_id, "ball")}
                            className={`border-[#30D158]/50 ${chosen === "ball" ? "bg-[#30D158]/20" : ""} text-[#30D158] hover:bg-[#30D158]/10`}>
                            <Check className="w-4 h-4 mr-1" /> Ball
                          </Button>
                          <Button size="sm" variant="outline" onClick={() => handleLabel(task.id, track.track_id, "not_ball")}
                            className={`border-[#FF3B30]/50 ${chosen === "not_ball" ? "bg-[#FF3B30]/20" : ""} text-[#FF3B30] hover:bg-[#FF3B30]/10`}>
                            <X className="w-4 h-4 mr-1" /> Not ball
                          </Button>
                          <Button size="sm" variant="outline" onClick={() => handleLabel(task.id, track.track_id, "unsure")}
                            className={`border-[#FF9F0A]/50 ${chosen === "unsure" ? "bg-[#FF9F0A]/20" : ""} text-[#FF9F0A] hover:bg-[#FF9F0A]/10`}>
                            <HelpCircle className="w-4 h-4 mr-1" /> Unsure
                          </Button>
                        </div>
                      </div>
                      {/* Filmstrip of crops — the ball should be at the centre
                          of each crop (marked). Hover any crop to enlarge it. */}
                      <p className="text-[11px] text-muted-foreground mb-1">
                        Is the marked dot the ball moving smoothly across these frames? Hover to zoom.
                      </p>
                      <div className="flex gap-2 overflow-x-auto pb-2 pt-1">
                        {track.points.map((p, i) => (
                          p.crop_b64 ? (
                            <div key={i} title={`frame ${p.frame_index} (${p.x}, ${p.y})`}
                              className="relative flex-shrink-0 group">
                              <img src={`data:image/jpeg;base64,${p.crop_b64}`} alt={`f${p.frame_index}`}
                                className="w-24 h-24 rounded border border-border/50 transition-transform duration-150
                                           group-hover:scale-[2.2] group-hover:z-20 group-hover:relative
                                           group-hover:border-primary group-hover:shadow-xl" />
                              {/* centre marker = where the candidate (ball?) is */}
                              <span className="absolute left-1/2 top-1/2 w-3 h-3 -ml-1.5 -mt-1.5 rounded-full
                                               border-2 border-[#DFFF00] pointer-events-none
                                               group-hover:opacity-0" />
                            </div>
                          ) : null
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))
        )}
        </>)}

        {/* ===================== MANUAL MARK MODE ===================== */}
        {mode === "manual" && (<>
        <div className="bg-card border border-border rounded-lg p-4 mb-6 flex items-end gap-4 flex-wrap">
          <div className="flex items-center gap-2"><MousePointerClick className="w-5 h-5 text-primary" />
            <span className="font-heading font-bold text-sm">Load frames to mark</span></div>
          <label className="text-xs text-muted-foreground">Start (s)
            <input type="number" value={mStart} onChange={(e) => setMStart(e.target.value)}
              className="block w-20 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
          <label className="text-xs text-muted-foreground">Frames
            <input type="number" value={mCount} onChange={(e) => setMCount(e.target.value)}
              className="block w-20 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
          <label className="text-xs text-muted-foreground">Step
            <input type="number" value={mStep} onChange={(e) => setMStep(e.target.value)}
              className="block w-20 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
          <Button onClick={loadFrames} disabled={mLoading}
            className="bg-primary text-primary-foreground hover:bg-primary/90">
            {mLoading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Loading…</> : "Load frames"}
          </Button>
        </div>

        {mFrames.length === 0 ? (
          <div className="text-center text-muted-foreground py-16 border border-dashed border-border rounded-lg">
            Load a window of frames, then <span className="text-primary">click the ball</span> in each. Use this when no auto-detected track is the ball.
          </div>
        ) : (
          <div>
            <div className="text-sm text-muted-foreground mb-2 flex items-center justify-between">
              <span>Frame {mIdx + 1}/{mFrames.length} · index {curFrame?.frame_index} · {curFrame?.timestamp}s</span>
              <span className="text-[#30D158]">{markedCount} marked</span>
            </div>
            <div className="relative inline-block w-full">
              {curFrame && (
                <img ref={imgRef} src={`data:image/jpeg;base64,${curFrame.b64}`} alt="frame"
                  onClick={handleMarkClick}
                  className="w-full rounded-lg cursor-crosshair select-none" draggable={false} />
              )}
              {curMark && (
                <div className="absolute w-4 h-4 -ml-2 -mt-2 rounded-full border-2 border-[#DFFF00] bg-[#DFFF00]/40 pointer-events-none"
                  style={{ left: `${curMark.nx * 100}%`, top: `${curMark.ny * 100}%` }} />
              )}
            </div>
            <p className="text-center text-xs text-muted-foreground mt-2">
              👆 Click the ball (auto-advances). If the ball isn't visible, just skip with →.
            </p>

            {/* Navigation + actions */}
            <div className="flex items-center justify-between mt-3 flex-wrap gap-2">
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="border-border"
                  onClick={() => setMIdx((i) => Math.max(0, i - 1))} disabled={mIdx === 0}>
                  <ChevronLeft className="w-4 h-4" /> Prev
                </Button>
                <Button variant="outline" size="sm" className="border-border"
                  onClick={() => setMIdx((i) => Math.min(mFrames.length - 1, i + 1))} disabled={mIdx === mFrames.length - 1}>
                  Next <ChevronRight className="w-4 h-4" />
                </Button>
                <Button variant="outline" size="sm" className="border-border" onClick={clearMark} disabled={!curMark}>
                  <Eraser className="w-4 h-4 mr-1" /> Clear this
                </Button>
              </div>
              <Button onClick={saveMarks} disabled={markedCount === 0}
                className="bg-[#30D158] text-black hover:bg-[#30D158]/90">
                <Save className="w-4 h-4 mr-2" /> Save {markedCount} ball points
              </Button>
            </div>

            {/* Click-once propagation */}
            <div className="mt-4 bg-background/50 border border-border rounded-lg p-3">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-primary" />
                  <span className="text-sm font-medium">Click once, track many</span>
                  <span className="text-xs text-muted-foreground">— mark the ball above, then auto-track it across ~30 frames</span>
                </div>
                <Button size="sm" onClick={doPropagate} disabled={!curMark || propLoading}
                  className="bg-primary text-primary-foreground hover:bg-primary/90">
                  {propLoading ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Tracking…</>
                    : <><Sparkles className="w-4 h-4 mr-2" /> Propagate from click</>}
                </Button>
              </div>

              {propPoints && (
                <div className="mt-3">
                  <p className="text-[11px] text-muted-foreground mb-1">
                    Tracked {propPoints.length} frames. Click any crop that drifted off the ball to exclude it (dimmed), then save.
                  </p>
                  <div className="flex gap-1 overflow-x-auto pb-2">
                    {propPoints.map((p) => {
                      const excluded = propExclude[p.frame_index];
                      return (
                        <button key={p.frame_index}
                          onClick={() => setPropExclude((e) => ({ ...e, [p.frame_index]: !e[p.frame_index] }))}
                          className={`relative flex-shrink-0 rounded border ${excluded ? "border-[#FF3B30]/60 opacity-30" : p.seed ? "border-primary" : "border-border/50"}`}>
                          {p.crop_b64 && <img src={`data:image/jpeg;base64,${p.crop_b64}`} alt={`f${p.frame_index}`} className="w-14 h-14 rounded" />}
                          <span className="absolute left-1/2 top-1/2 w-2.5 h-2.5 -ml-[5px] -mt-[5px] rounded-full border-2 border-[#DFFF00] pointer-events-none" />
                        </button>
                      );
                    })}
                  </div>
                  <div className="flex gap-2 mt-1">
                    <Button size="sm" onClick={savePropagated}
                      className="bg-[#30D158] text-black hover:bg-[#30D158]/90">
                      <Save className="w-4 h-4 mr-1" /> Save {propPoints.filter((p) => !propExclude[p.frame_index]).length} tracked points
                    </Button>
                    <Button size="sm" variant="outline" className="border-border" onClick={() => setPropPoints(null)}>
                      Discard
                    </Button>
                  </div>
                </div>
              )}
            </div>

            {/* Filmstrip navigation */}
            <div className="flex gap-1 overflow-x-auto mt-4 pb-1">
              {mFrames.map((f, i) => (
                <button key={f.frame_index} onClick={() => setMIdx(i)}
                  className={`relative flex-shrink-0 rounded border ${i === mIdx ? "border-primary" : "border-border/50"}`}>
                  <img src={`data:image/jpeg;base64,${f.b64}`} alt={`f${f.frame_index}`} className="w-16 h-10 object-cover rounded" />
                  {marks[f.frame_index] && (
                    <span className="absolute top-0.5 right-0.5 w-2 h-2 rounded-full bg-[#DFFF00]" />
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
        </>)}

        {/* ===================== AUTO-LABEL (self-training) ===================== */}
        {mode === "selftrain" && (<>
        <div className="bg-card border border-border rounded-lg p-4 mb-4 flex items-end gap-4 flex-wrap">
          <div className="flex items-center gap-2"><Wand2 className="w-5 h-5 text-primary" />
            <span className="font-heading font-bold text-sm">Mine ball arcs with the model</span></div>
          <label className="text-xs text-muted-foreground">Start (s)
            <input type="number" value={stStart} onChange={(e) => setStStart(e.target.value)}
              className="block w-20 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
          <label className="text-xs text-muted-foreground">Duration (s)
            <input type="number" value={stDuration} onChange={(e) => setStDuration(e.target.value)}
              className="block w-20 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
          <Button onClick={mineVideo} disabled={stMining}
            className="bg-primary text-primary-foreground hover:bg-primary/90">
            {stMining ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Mining…</> : <><Wand2 className="w-4 h-4 mr-2" /> Mine this video</>}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mb-6">
          The trained model finds ball arcs and the physics filter keeps only the ones that move like a ball.
          <span className="text-[#30D158]"> Approve</span> the real ones (added to training) and
          <span className="text-[#FF3B30]"> reject</span> false positives (kept as "not ball" — teaches the model what to ignore).
          You judge, you don't click frame-by-frame.
        </p>

        {stProposals.length === 0 ? (
          <div className="text-center text-muted-foreground py-16 border border-dashed border-border rounded-lg">
            No proposals yet. Pick a rally window and click <span className="text-primary">Mine this video</span>.
            (Needs a trained model — train one first if the badge above says "classical fallback".)
          </div>
        ) : (
          <div className="space-y-3">
            {stProposals.map((p) => (
              <div key={p.id} className="bg-card border border-border rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono text-muted-foreground">
                    model proposal · quality {p.quality} · confidence {p.mean_confidence} · {p.num_points} pts
                  </span>
                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" onClick={() => reviewProposal(p.id, "approve")}
                      className="border-[#30D158]/50 text-[#30D158] hover:bg-[#30D158]/10">
                      <Check className="w-4 h-4 mr-1" /> Approve (ball)
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => reviewProposal(p.id, "reject")}
                      className="border-[#FF3B30]/50 text-[#FF3B30] hover:bg-[#FF3B30]/10">
                      <X className="w-4 h-4 mr-1" /> Reject
                    </Button>
                  </div>
                </div>
                <p className="text-[11px] text-muted-foreground mb-2">
                  Is the <span className="text-[#30D158]">green path</span> a ball flying across the court
                  (smooth, on the floor/walls) — or is it on a player's body / a reflection? Yellow ring = start, red = end.
                </p>
                {p.overlay_b64 && (
                  <img src={`data:image/jpeg;base64,${p.overlay_b64}`} alt="proposed arc on frame"
                    className="w-full max-w-2xl rounded-lg border border-border mb-2" />
                )}
                <div className="flex gap-1 overflow-x-auto pb-1">
                  {p.points.map((pt, i) => (
                    pt.crop_b64 ? (
                      <div key={i} className="relative flex-shrink-0 group">
                        <img src={`data:image/jpeg;base64,${pt.crop_b64}`} alt={`f${pt.frame_index}`}
                          className="w-12 h-12 rounded border border-border/50 transition-transform group-hover:scale-[2.5] group-hover:z-20 group-hover:relative" />
                        <span className="absolute left-1/2 top-1/2 w-2 h-2 -ml-1 -mt-1 rounded-full border-2 border-[#DFFF00] pointer-events-none group-hover:opacity-0" />
                      </div>
                    ) : null
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
        </>)}

        {/* ===================== TIMELINE MODE (M2) ===================== */}
        {mode === "timeline" && (<>
        <div className="bg-card border border-border rounded-lg p-4 mb-4 flex items-end gap-4 flex-wrap">
          <div className="flex items-center gap-2"><Activity className="w-5 h-5 text-primary" />
            <span className="font-heading font-bold text-sm">Build rally timeline</span></div>
          <label className="text-xs text-muted-foreground">Start (s)
            <input type="number" value={tStart} onChange={(e) => setTStart(e.target.value)}
              className="block w-20 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
          <label className="text-xs text-muted-foreground">Duration (s)
            <input type="number" value={tDuration} onChange={(e) => setTDuration(e.target.value)}
              className="block w-20 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
          <Button onClick={buildTimeline} disabled={tBuilding}
            className="bg-primary text-primary-foreground hover:bg-primary/90">
            {tBuilding ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Building…</> : "Build timeline"}
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mb-6">
          Detects ball→racket contacts (direction reversals) and groups them into rallies. Needs the
          court marked. Shot <span className="text-primary">type</span> stays blank until the shot
          classifier (M4); accuracy is bounded by ball tracking, so results sharpen once TrackNet is trained.
        </p>

        {timelines.length === 0 ? (
          <div className="text-center text-muted-foreground py-16 border border-dashed border-border rounded-lg">
            No timelines yet. Mark the court (if you haven't), pick a rally window, and click <span className="text-primary">Build timeline</span>.
          </div>
        ) : (
          timelines.map((doc) => {
            const t = doc.timeline || {};
            return (
              <div key={doc.id} className="mb-8 bg-card border border-border rounded-lg p-4">
                <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
                  <span className="text-sm text-muted-foreground flex items-center gap-2">
                    <Clock className="w-4 h-4" /> {doc.start_s}s–{(doc.start_s + doc.duration_s).toFixed(0)}s
                  </span>
                  <div className="flex gap-4 text-xs font-mono">
                    <span className="text-primary">{t.total_shots ?? 0} shots</span>
                    <span className="text-[#00F0FF]">{t.total_rallies ?? 0} rallies</span>
                    <span className="text-muted-foreground">{t.ball_points ?? 0} ball pts</span>
                    <span className="text-muted-foreground">{t.detector}</span>
                  </div>
                </div>
                {(!t.rallies || t.rallies.length === 0) ? (
                  <p className="text-sm text-muted-foreground">
                    No shot contacts found in this window — expected with classical ball tracking
                    (the smooth candidate track has no sharp reversals). Will populate once a trained
                    ball model lands.
                  </p>
                ) : (
                  t.rallies.map((r) => (
                    <div key={r.rally_id} className="mb-3">
                      <div className="text-xs text-muted-foreground mb-1">
                        Rally {r.rally_id} · {r.shot_count} shots · {r.start_t}–{r.end_t}s
                      </div>
                      <div className="space-y-1">
                        {r.shots.map((s) => (
                          <div key={s.shot_id} className="flex items-center gap-3 text-xs font-mono bg-background rounded px-3 py-1.5">
                            <span className="text-muted-foreground w-12">{s.t_contact}s</span>
                            <span className={s.striker === "player1" ? "text-primary" : s.striker === "player2" ? "text-[#00F0FF]" : "text-muted-foreground"}>
                              {s.striker}
                            </span>
                            <span className="text-muted-foreground">turn {s.turn_angle_deg}°</span>
                            <span className="text-muted-foreground">{s.incoming_speed_ms}→{s.outgoing_speed_ms} m/s</span>
                            <span className="text-muted-foreground">conf {s.confidence}</span>
                            <span className="ml-auto text-[#FF9F0A]">{s.shot_type || "type: —"}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))
                )}
              </div>
            );
          })
        )}
        </>)}
      </div>
    </div>
  );
};

export default BallAnnotationPage;
