import { useState, useEffect, useCallback, useRef } from "react";
import { useNavigate, Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { toast } from "sonner";
import axios from "axios";
import {
  Target, ArrowLeft, Loader2, Crosshair, Database, FolderInput, Film, Video, Wand2, Check, X, Youtube
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// The training library: every past-game video that feeds model training. Bulk
// ingest a folder, see per-video labelling progress, jump into labelling, and
// train across the whole library from one place.
const TrainingLibraryPage = () => {
  const navigate = useNavigate();
  const [lib, setLib] = useState(null);
  const [loading, setLoading] = useState(true);
  const [folder, setFolder] = useState("");
  const [ingesting, setIngesting] = useState(false);
  const [ytUrls, setYtUrls] = useState("");
  const [ytState, setYtState] = useState(null);
  const ytPollRef = useRef(null);

  const ingestYoutube = async () => {
    const urls = ytUrls.split("\n").map((u) => u.trim()).filter(Boolean);
    if (!urls.length) { toast.error("Paste at least one YouTube URL"); return; }
    try {
      await axios.post(`${API}/training/ingest-youtube`, { urls });
      toast.success(`Downloading ${urls.length} video(s) in the background…`);
      setYtState({ status: "running", processed: 0, total: urls.length, added: 0 });
      setYtUrls("");
      ytPollRef.current = setInterval(async () => {
        try {
          const s = await axios.get(`${API}/training/ingest-youtube/status`);
          setYtState(s.data);
          if (s.data.status === "done" || s.data.status === "failed") {
            clearInterval(ytPollRef.current);
            toast.success(`Added ${s.data.added} video(s) from YouTube`);
            fetchLibrary();
          }
        } catch (e) { /* keep polling */ }
      }, 2500);
    } catch (e) {
      toast.error(e?.response?.data?.detail || "YouTube ingest failed");
    }
  };
  const [trainState, setTrainState] = useState(null);
  const pollRef = useRef(null);

  // ----- batch self-training: mine all + one-pass review -----
  const [mineState, setMineState] = useState(null);
  const [proposals, setProposals] = useState([]);
  const minePollRef = useRef(null);

  const fetchProposals = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/selftrain/proposals?limit=60`);
      setProposals(r.data.proposals || []);
    } catch (e) { /* ignore */ }
  }, []);

  useEffect(() => { fetchProposals(); }, [fetchProposals]);
  useEffect(() => () => { if (minePollRef.current) clearInterval(minePollRef.current); }, []);

  const mineAll = async () => {
    try {
      const r = await axios.post(`${API}/selftrain/mine-all`, { start_s: 30, duration_s: 12, min_quality: 80, max_videos: 40 });
      toast.success(`Mining ${r.data.videos} videos in the background…`);
      setMineState({ status: "running", processed: 0, total: r.data.videos, proposals: 0 });
      minePollRef.current = setInterval(async () => {
        try {
          const s = await axios.get(`${API}/selftrain/mine-all/status`);
          setMineState(s.data);
          if (s.data.status === "done" || s.data.status === "failed") {
            clearInterval(minePollRef.current);
            if (s.data.status === "done") { toast.success(`Found ${s.data.proposals} proposals to review`); fetchProposals(); }
            else toast.error("Mining failed: " + (s.data.error || "unknown"));
          }
        } catch (e) { /* keep polling */ }
      }, 2000);
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Could not start mining");
    }
  };

  const reviewProposal = async (p, decision) => {
    setProposals((arr) => arr.filter((x) => x.id !== p.id));
    try {
      await axios.post(`${API}/selftrain/review/${p.match_id}`, { proposal_id: p.id, decision });
    } catch (e) { toast.error("Review failed"); fetchProposals(); }
  };

  const fetchLibrary = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/training/library`);
      setLib(res.data);
    } catch (e) {
      toast.error("Failed to load library");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchLibrary(); }, [fetchLibrary]);
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const handleIngest = async () => {
    if (!folder.trim()) { toast.error("Enter a folder path"); return; }
    setIngesting(true);
    try {
      const res = await axios.post(`${API}/training/ingest`, { folder_path: folder.trim() });
      toast.success(`Ingested ${res.data.ingested} videos`);
      setFolder("");
      await fetchLibrary();
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Ingest failed");
    } finally {
      setIngesting(false);
    }
  };

  const startTraining = async () => {
    try {
      const res = await axios.post(`${API}/annotation/ball/train`, { epochs: 20, min_samples: 30 });
      toast.success(`Training on ${res.data.samples} points from your library`);
      setTrainState({ status: "running", epoch: 0, epochs: res.data.epochs });
      pollRef.current = setInterval(async () => {
        try {
          const s = await axios.get(`${API}/annotation/ball/train/status`);
          setTrainState(s.data);
          if (s.data.status === "done" || s.data.status === "failed") {
            clearInterval(pollRef.current);
            if (s.data.status === "done") { toast.success("Training done — model active"); fetchLibrary(); }
            else toast.error("Training failed: " + (s.data.error || "unknown"));
          }
        } catch (e) { /* keep polling */ }
      }, 1500);
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Could not start training");
    }
  };

  if (loading) {
    return <div className="min-h-screen bg-[#050505] flex items-center justify-center">
      <Loader2 className="w-8 h-8 animate-spin text-primary" /></div>;
  }

  const recommended = 200;
  const total = lib?.total_ball_points || 0;

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
          <Link to="/history"><Button variant="ghost" className="text-muted-foreground">
            <ArrowLeft className="w-4 h-4 mr-2" /> History</Button></Link>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="mb-6">
          <h1 className="font-heading text-3xl font-black tracking-tight mb-2">
            TRAINING <span className="text-primary">LIBRARY</span>
          </h1>
          <p className="text-muted-foreground max-w-2xl">
            Every past-game video that feeds the models. Bulk-ingest a folder, label the ball across
            them, and train one model on your whole library. Labels from all videos combine.
          </p>
        </div>

        {/* Ingest + train */}
        <div className="grid md:grid-cols-2 gap-4 mb-8">
          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <FolderInput className="w-5 h-5 text-primary" />
              <span className="font-heading font-bold text-sm">Bulk ingest a folder</span>
            </div>
            <p className="text-xs text-muted-foreground mb-3">
              Server-side path to a folder of videos on this machine. For very large libraries use the CLI:
              <code className="text-primary"> ingest_videos.py /path</code>
            </p>
            <div className="flex gap-2">
              <input value={folder} onChange={(e) => setFolder(e.target.value)}
                placeholder="/Users/you/squash-videos"
                className="flex-1 bg-background border border-border rounded px-3 py-2 text-sm" />
              <Button onClick={handleIngest} disabled={ingesting}
                className="bg-primary text-primary-foreground hover:bg-primary/90">
                {ingesting ? <Loader2 className="w-4 h-4 animate-spin" /> : "Ingest"}
              </Button>
            </div>
          </div>

          <div className="bg-card border border-border rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-5 h-5 text-primary" />
              <span className="font-heading font-bold text-sm">Train on the whole library</span>
              {lib?.model_active
                ? <span className="text-[11px] px-2 py-0.5 rounded-full bg-[#30D158]/20 text-[#30D158]">model active</span>
                : <span className="text-[11px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground">classical fallback</span>}
            </div>
            <div className="w-full h-2 bg-background rounded-full overflow-hidden mb-1">
              <div className="h-full bg-primary" style={{ width: `${Math.min(100, total / recommended * 100)}%` }} />
            </div>
            <p className="text-xs text-muted-foreground mb-3">
              {total} / {recommended} ball points labelled across all videos
            </p>
            <Button onClick={startTraining} disabled={trainState?.status === "running"}
              className="bg-primary text-primary-foreground hover:bg-primary/90">
              {trainState?.status === "running"
                ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> epoch {trainState.epoch}/{trainState.epochs}{trainState.loss != null ? ` · loss ${trainState.loss}` : ""}</>
                : <><Database className="w-4 h-4 mr-2" /> Train model now</>}
            </Button>
          </div>
        </div>

        {/* Add from YouTube */}
        <div className="bg-card border border-border rounded-lg p-4 mb-8">
          <div className="flex items-center gap-2 mb-2">
            <Youtube className="w-5 h-5 text-[#FF3B30]" />
            <span className="font-heading font-bold text-sm">Add from YouTube</span>
          </div>
          <p className="text-xs text-muted-foreground mb-3">
            Paste squash video URLs (one per line). Pro broadcast footage adds court/player diversity that
            strengthens the ball model. Downloaded as ≤720p. <span className="opacity-70">For personal model
            training — respect YouTube's terms and the videos' copyright.</span>
          </p>
          <textarea value={ytUrls} onChange={(e) => setYtUrls(e.target.value)} rows={3}
            placeholder={"https://www.youtube.com/watch?v=...\nhttps://youtu.be/..."}
            className="w-full bg-background border border-border rounded px-3 py-2 text-sm mb-3 font-mono" />
          <div className="flex items-center gap-3">
            <Button onClick={ingestYoutube} disabled={ytState?.status === "running"}
              className="bg-[#FF3B30] text-white hover:bg-[#FF3B30]/90">
              {ytState?.status === "running"
                ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Downloading {ytState.processed}/{ytState.total}…</>
                : <><Youtube className="w-4 h-4 mr-2" /> Add from YouTube</>}
            </Button>
            {ytState?.status === "done" && (
              <span className="text-xs text-[#30D158]">added {ytState.added}{ytState.errors?.length ? ` · ${ytState.errors.length} failed` : ""}</span>
            )}
          </div>
        </div>

        {/* Self-training: mine all + one-pass review */}
        <div className="bg-card border border-border rounded-lg p-4 mb-8">
          <div className="flex items-center justify-between flex-wrap gap-3 mb-2">
            <div className="flex items-center gap-2">
              <Wand2 className="w-5 h-5 text-primary" />
              <span className="font-heading font-bold text-sm">Self-training — mine all videos</span>
            </div>
            <Button onClick={mineAll} disabled={mineState?.status === "running" || !lib?.model_active}
              className="bg-primary text-primary-foreground hover:bg-primary/90">
              {mineState?.status === "running"
                ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Mining {mineState.processed}/{mineState.total}…</>
                : <><Wand2 className="w-4 h-4 mr-2" /> Mine all videos</>}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            {lib?.model_active
              ? "The trained model proposes ball arcs across your library; review them once below. Approve real balls, reject false positives (hard negatives). Then Train."
              : "Train a model first — self-training needs a trained model to propose from."}
            {mineState?.status === "running" && <span className="text-primary"> · {mineState.proposals} proposals so far</span>}
          </p>

          {proposals.length > 0 && (
            <div className="mt-4">
              <div className="text-sm text-muted-foreground mb-2">{proposals.length} proposals to review</div>
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
                {proposals.map((p) => (
                  <div key={p.id} className="bg-background border border-border rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2 flex-wrap gap-2">
                      <span className="text-xs font-mono text-muted-foreground">
                        {p.match_title} · q{p.quality} · conf {p.mean_confidence} · {p.num_points}pts
                      </span>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline" onClick={() => reviewProposal(p, "approve")}
                          className="border-[#30D158]/50 text-[#30D158] hover:bg-[#30D158]/10">
                          <Check className="w-4 h-4 mr-1" /> Ball
                        </Button>
                        <Button size="sm" variant="outline" onClick={() => reviewProposal(p, "reject")}
                          className="border-[#FF3B30]/50 text-[#FF3B30] hover:bg-[#FF3B30]/10">
                          <X className="w-4 h-4 mr-1" /> Not ball
                        </Button>
                      </div>
                    </div>
                    {p.overlay_b64 && (
                      <img src={`data:image/jpeg;base64,${p.overlay_b64}`} alt="arc"
                        className="w-full max-w-xl rounded border border-border" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Library grid */}
        <div className="flex items-center gap-2 mb-3 text-sm text-muted-foreground">
          <Film className="w-4 h-4" /> {lib?.video_count || 0} videos in library
        </div>
        {(!lib?.videos || lib.videos.length === 0) ? (
          <div className="text-center text-muted-foreground py-16 border border-dashed border-border rounded-lg">
            No videos yet. Ingest a folder above, or upload one from the <Link to="/upload" className="text-primary">Upload</Link> page.
          </div>
        ) : (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {lib.videos.map((v) => (
              <div key={v.id} className="bg-card border border-border rounded-lg overflow-hidden">
                <div className="aspect-video bg-muted relative">
                  {v.thumbnail
                    ? <img src={`data:image/jpeg;base64,${v.thumbnail}`} alt={v.title} className="w-full h-full object-cover" />
                    : <div className="w-full h-full flex items-center justify-center"><Video className="w-8 h-8 text-muted-foreground" /></div>}
                  {v.source === "library" && (
                    <span className="absolute top-2 left-2 text-[10px] px-2 py-0.5 rounded-full bg-background/80 text-muted-foreground">library</span>
                  )}
                  <span className={`absolute top-2 right-2 text-[10px] px-2 py-0.5 rounded-full font-mono ${
                    v.ball_points > 0 ? "bg-[#30D158]/20 text-[#30D158]" : "bg-background/80 text-muted-foreground"}`}>
                    {v.ball_points} pts
                  </span>
                </div>
                <div className="p-3">
                  <p className="font-medium text-sm truncate mb-2" title={v.title}>{v.title}</p>
                  <Button size="sm" variant="outline" onClick={() => navigate(`/annotate-ball/${v.id}`)}
                    className="w-full border-primary/50 text-primary hover:bg-primary/10">
                    <Crosshair className="w-4 h-4 mr-2" /> Label ball
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default TrainingLibraryPage;
