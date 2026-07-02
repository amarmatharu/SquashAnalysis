import { useState, useEffect, useRef, Fragment } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Progress } from "../components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { ScrollArea } from "../components/ui/scroll-area";
import axios from "axios";
import { toast } from "sonner";
import { 
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, 
  ResponsiveContainer, Tooltip, Legend, ScatterChart, Scatter
} from "recharts";
import { 
  Target, 
  ArrowLeft,
  Loader2,
  Download,
  RefreshCw,
  Activity,
  Zap,
  Users,
  TrendingUp,
  Clock,
  FileJson,
  FileText,
  Edit2,
  Brain
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";
import ShotCorrectionModal from "../components/ShotCorrectionModal";
import TrainingStats from "../components/TrainingStats";
import CourtCalibrationModal from "../components/CourtCalibrationModal";
import CourtView from "../components/CourtView";
import CourtControlPanel from "../components/CourtControlPanel";
import ShotPatternsPanel from "../components/ShotPatternsPanel";
import ScoutingReport from "../components/ScoutingReport";
import PlayerIdentifyModal from "../components/PlayerIdentifyModal";
import FlywheelPanel from "../components/FlywheelPanel";
import ModelScorecard from "../components/ModelScorecard";
import Timeline3DPanel from "../components/Timeline3DPanel";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const SHOT_COLORS = {
  drive: "#DFFF00",
  drop: "#00F0FF",
  boast: "#FF3B30",
  volley: "#30D158",
  lob: "#FF9F0A",
  kill: "#BF5AF2",
  serve: "#64D2FF"
};

const AnalysisPage = () => {
  const { matchId } = useParams();
  const navigate = useNavigate();
  const [match, setMatch] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [correctionModal, setCorrectionModal] = useState({ isOpen: false, shot: null, index: -1 });
  const [showCalibration, setShowCalibration] = useState(false);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [courtCalibration, setCourtCalibration] = useState(null);
  const [showIdentify, setShowIdentify] = useState(false);
  // Real player names (fall back to Player 1/2 until identified)
  const playerNames = {
    1: (match?.player1_name && match.player1_name !== "Player 1") ? match.player1_name : "Player 1",
    2: (match?.player2_name && match.player2_name !== "Player 2") ? match.player2_name : "Player 2",
  };
  const playersIdentified = !!(match && (
    (match.player1_name && match.player1_name !== "Player 1") ||
    (match.player2_name && match.player2_name !== "Player 2")));

  // Player detection
  const [playerStart, setPlayerStart] = useState(30);
  const [playerDur, setPlayerDur] = useState(60);
  const [playerDetection, setPlayerDetection] = useState(null);
  const [playerRunning, setPlayerRunning] = useState(false);
  const playerPoll = useRef(null);
  useEffect(() => () => { if (playerPoll.current) clearInterval(playerPoll.current); }, []);
  useEffect(() => {
    axios.get(`${API}/analysis/players/${matchId}`).then(r => {
      if (r.data?.status === "done") setPlayerDetection(r.data);
    }).catch(() => {});
  }, [matchId]);

  // Court control (tactical movement over rally windows)
  const [courtControl, setCourtControl] = useState(null);
  const [ccRunning, setCcRunning] = useState(false);
  const ccPoll = useRef(null);
  useEffect(() => () => { if (ccPoll.current) clearInterval(ccPoll.current); }, []);
  useEffect(() => {
    axios.get(`${API}/analysis/court-control/${matchId}`).then(r => {
      if (r.data?.status === "done") setCourtControl(r.data);
    }).catch(() => {});
  }, [matchId]);

  const analyzeCourtControl = async () => {
    setCcRunning(true); setCourtControl(null);
    try {
      await axios.post(`${API}/analysis/court-control/${matchId}`);
      ccPoll.current = setInterval(async () => {
        try {
          const r = await axios.get(`${API}/analysis/court-control/${matchId}`);
          if (r.data.status === "done") {
            clearInterval(ccPoll.current); setCcRunning(false); setCourtControl(r.data);
            toast.success("Court control analyzed");
          } else if (r.data.status === "failed") {
            clearInterval(ccPoll.current); setCcRunning(false);
            toast.error(r.data.error || "Court control failed");
          }
        } catch (e) { /* keep polling */ }
      }, 3000);
    } catch (e) {
      setCcRunning(false);
      toast.error(e?.response?.data?.detail || "Run rally segmentation first");
    }
  };

  // Full match analysis (one-click orchestrator)
  const [fullAnalysis, setFullAnalysis] = useState(null);
  const fullPoll = useRef(null);
  useEffect(() => () => { if (fullPoll.current) clearInterval(fullPoll.current); }, []);

  const refreshAllAnalyses = async () => {
    // Pull every stage's stored result into the UI after a full run
    const get = (p) => axios.get(`${API}/analysis/${p}/${matchId}`).then(r => r.data).catch(() => null);
    const [rs, cc, sp, sc] = await Promise.all([
      get("rallies"), get("court-control"), get("shot-patterns"), get("scouting"),
    ]);
    if (rs?.status === "done") { setRallySeg(rs); setRallyOutcomes(rs.outcomes || {}); }
    if (cc?.status === "done") setCourtControl(cc);
    if (sp?.status === "done") setShotPatterns(sp);
    if (sc?.status === "done") setScouting(sc);
  };

  const runFullAnalysis = async () => {
    try {
      await axios.post(`${API}/analysis/full/${matchId}`);
      setFullAnalysis({ status: "running", stage: "rallies", stage_label: "Segmenting rallies", stages_done: [], total_stages: 4 });
      toast.info("Full match analysis started — this runs all stages.");
      fullPoll.current = setInterval(async () => {
        try {
          const r = await axios.get(`${API}/analysis/full/${matchId}`);
          setFullAnalysis(r.data);
          if (r.data.status === "done") {
            clearInterval(fullPoll.current);
            await refreshAllAnalyses();
            toast.success("Full match analysis complete");
          } else if (r.data.status === "failed") {
            clearInterval(fullPoll.current);
            toast.error(`Analysis failed at ${r.data.stage}: ${r.data.error || ""}`);
          }
        } catch (e) { /* keep polling */ }
      }, 3000);
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Could not start full analysis");
    }
  };

  // Scouting report (LLM reasoning layer)
  const [scouting, setScouting] = useState(null);
  const [scoutRunning, setScoutRunning] = useState(false);
  useEffect(() => {
    axios.get(`${API}/analysis/scouting/${matchId}`).then(r => {
      if (r.data?.status === "done") setScouting(r.data);
    }).catch(() => {});
  }, [matchId]);

  const generateScouting = async () => {
    setScoutRunning(true);
    try {
      const r = await axios.post(`${API}/analysis/scouting/${matchId}`);
      setScouting(r.data);
      toast.success(r.data.llm_used ? "Scouting report generated (Claude)" : "Scouting report generated");
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Run rally segmentation first");
    } finally {
      setScoutRunning(false);
    }
  };

  // Shot patterns & error zones
  const [shotPatterns, setShotPatterns] = useState(null);
  const [spRunning, setSpRunning] = useState(false);
  const spPoll = useRef(null);
  useEffect(() => () => { if (spPoll.current) clearInterval(spPoll.current); }, []);
  useEffect(() => {
    axios.get(`${API}/analysis/shot-patterns/${matchId}`).then(r => {
      if (r.data?.status === "done") setShotPatterns(r.data);
    }).catch(() => {});
  }, [matchId]);

  const analyzeShotPatterns = async () => {
    setSpRunning(true); setShotPatterns(null);
    try {
      await axios.post(`${API}/analysis/shot-patterns/${matchId}`);
      spPoll.current = setInterval(async () => {
        try {
          const r = await axios.get(`${API}/analysis/shot-patterns/${matchId}`);
          if (r.data.status === "done") {
            clearInterval(spPoll.current); setSpRunning(false); setShotPatterns(r.data);
            toast.success("Shot patterns analyzed");
          } else if (r.data.status === "failed") {
            clearInterval(spPoll.current); setSpRunning(false);
            toast.error(r.data.error || "Shot pattern analysis failed");
          }
        } catch (e) { /* keep polling */ }
      }, 3000);
    } catch (e) {
      setSpRunning(false);
      toast.error(e?.response?.data?.detail || "Run rally segmentation first");
    }
  };

  const detectPlayers = async () => {
    setPlayerRunning(true); setPlayerDetection(null);
    try {
      await axios.post(`${API}/analysis/players/${matchId}`, {
        start_s: Number(playerStart), duration_s: Number(playerDur),
      });
      playerPoll.current = setInterval(async () => {
        try {
          const r = await axios.get(`${API}/analysis/players/${matchId}`);
          if (r.data.status === "done") {
            clearInterval(playerPoll.current); setPlayerRunning(false); setPlayerDetection(r.data);
            toast.success("Player detection complete");
          } else if (r.data.status === "failed") {
            clearInterval(playerPoll.current); setPlayerRunning(false);
            toast.error(r.data.error || "Player detection failed");
          }
        } catch (e) { /* keep polling */ }
      }, 3000);
    } catch (e) {
      setPlayerRunning(false);
      toast.error(e?.response?.data?.detail || "Could not start player detection");
    }
  };

  // Ball trace (comet-trail video)
  const [traceStart, setTraceStart] = useState(30);
  const [traceDur, setTraceDur] = useState(8);
  const [traceSmooth, setTraceSmooth] = useState(true);
  const [traceState, setTraceState] = useState(null);
  const [traceUrl, setTraceUrl] = useState(null);
  const tracePoll = useRef(null);
  useEffect(() => () => { if (tracePoll.current) clearInterval(tracePoll.current); }, []);

  // Rally segmentation (perception-based)
  const [rallyStart, setRallyStart] = useState(0);
  const [rallyDur, setRallyDur] = useState(120);
  const [rallySeg, setRallySeg] = useState(null);
  const [rallyRunning, setRallyRunning] = useState(false);
  const [rallyOutcomes, setRallyOutcomes] = useState({});  // rally_id -> outcome string
  const [expandedRally, setExpandedRally] = useState(null);  // rally_id showing clip
  const [clipVersion, setClipVersion] = useState(0);  // bump to cache-bust clips after edits
  const expandedVideoRef = useRef(null);  // the open rally clip's <video>
  const CLIP_BUFFER = 0.4;  // must match backend extract_rally_clip buffer_s
  const rallyPoll = useRef(null);
  useEffect(() => () => { if (rallyPoll.current) clearInterval(rallyPoll.current); }, []);
  useEffect(() => {
    axios.get(`${API}/analysis/rallies/${matchId}`).then((r) => {
      if (r.data?.status === "done") {
        setRallySeg(r.data);
        setRallyOutcomes(r.data.outcomes || {});
      }
    }).catch(() => {});
  }, [matchId]);

  // Squash score: first to 11, win by 2. Let = replay, stroke = point.
  const computeScore = (rallies, outcomes) => {
    let p1 = 0, p2 = 0;
    const history = [];
    for (const r of (rallies || [])) {
      const o = outcomes[r.rally_id];
      if (o === "p1" || o === "stroke_p1") p1++;
      else if (o === "p2" || o === "stroke_p2") p2++;
      history.push({ rally_id: r.rally_id, p1, p2, outcome: o });
    }
    return { p1, p2, history };
  };

  // 3D Rally Timeline (full-stack L1-L6 integration)
  const [timeline3d, setTimeline3d] = useState(null);
  const [tl3dRunning, setTl3dRunning] = useState(false);
  const tl3dPoll = useRef(null);
  useEffect(() => () => { if (tl3dPoll.current) clearInterval(tl3dPoll.current); }, []);
  useEffect(() => {
    axios.get(`${API}/analysis/timeline3d/${matchId}`).then(r => {
      if (r.data?.status === "done") setTimeline3d(r.data);
    }).catch(() => {});
  }, [matchId]);

  const buildTimeline3d = async () => {
    setTl3dRunning(true); setTimeline3d(null);
    try {
      await axios.post(`${API}/analysis/timeline3d/${matchId}`);
      tl3dPoll.current = setInterval(async () => {
        try {
          const r = await axios.get(`${API}/analysis/timeline3d/${matchId}`);
          if (r.data.status === "done") {
            clearInterval(tl3dPoll.current); setTl3dRunning(false); setTimeline3d(r.data);
            toast.success("3D rally timeline built");
          } else if (r.data.status === "failed") {
            clearInterval(tl3dPoll.current); setTl3dRunning(false);
            toast.error(r.data.error || "3D timeline failed");
          }
        } catch (e) { /* keep polling */ }
      }, 3000);
    } catch (e) {
      setTl3dRunning(false);
      toast.error(e?.response?.data?.detail || "Calibrate the court first");
    }
  };

  // Engine-driven scoreboard (Layer 6 rules: games, serve, match) — refreshes on tag changes
  const [scoreboard, setScoreboard] = useState(null);
  useEffect(() => {
    if (!rallySeg || Object.keys(rallyOutcomes).length === 0) { setScoreboard(null); return; }
    const id = setTimeout(() => {
      axios.post(`${API}/analysis/scoreboard/${matchId}`, { first_server: 1 })
        .then(r => setScoreboard(r.data)).catch(() => {});
    }, 300);
    return () => clearTimeout(id);
  }, [rallyOutcomes, rallySeg, matchId]);

  const tagOutcome = async (rallyId, outcome) => {
    setRallyOutcomes((prev) => ({ ...prev, [rallyId]: outcome }));
    try {
      await axios.post(`${API}/analysis/rallies/${matchId}/${rallyId}/outcome`, { outcome });
    } catch (e) {
      toast.error("Could not save outcome");
      setRallyOutcomes((prev) => { const n = { ...prev }; delete n[rallyId]; return n; });
    }
  };

  const segmentRallies = async () => {
    setRallyRunning(true); setRallySeg(null); setRallyOutcomes({});
    try {
      await axios.post(`${API}/analysis/rallies/${matchId}`, {
        start_s: Number(rallyStart), duration_s: Number(rallyDur),
      });
      rallyPoll.current = setInterval(async () => {
        try {
          const r = await axios.get(`${API}/analysis/rallies/${matchId}`);
          if (r.data.status === "done") {
            clearInterval(rallyPoll.current); setRallyRunning(false);
            setRallySeg(r.data); setRallyOutcomes(r.data.outcomes || {});
            toast.success(`Found ${r.data.num_rallies} rallies — clips extracting…`);
          } else if (r.data.status === "failed") {
            clearInterval(rallyPoll.current); setRallyRunning(false);
            toast.error("Segmentation failed");
          }
        } catch (e) { /* keep polling */ }
      }, 2500);
    } catch (e) {
      setRallyRunning(false);
      toast.error(e?.response?.data?.detail || "Could not start segmentation");
    }
  };

  const generateTrace = async () => {
    setTraceUrl(null);
    try {
      await axios.post(`${API}/analysis/trace/${matchId}`, {
        start_s: Number(traceStart), duration_s: Number(traceDur), smooth: traceSmooth,
      });
      setTraceState({ status: "running" });
      tracePoll.current = setInterval(async () => {
        try {
          const r = await axios.get(`${API}/analysis/trace/${matchId}/status`);
          setTraceState(r.data);
          if (r.data.status === "done" || r.data.status === "failed") {
            clearInterval(tracePoll.current);
            if (r.data.status === "done") {
              setTraceUrl(`${API}/analysis/trace/${matchId}/video?t=${Date.now()}`);
              toast.success(`Traced ball in ${r.data.detections}/${r.data.frame_count} frames`);
            } else toast.error("Trace failed: " + (r.data.error || "unknown"));
          }
        } catch (e) { /* keep polling */ }
      }, 2000);
    } catch (e) {
      toast.error(e?.response?.data?.detail || "Could not start trace");
    }
  };

  useEffect(() => {
    fetchMatch();
    axios.get(`${API}/matches/${matchId}/calibrate`).then(r => {
      setIsCalibrated(r.data?.calibrated === true);
      if (r.data?.calibrated) setCourtCalibration(r.data.calibration);
    }).catch(() => {});
  }, [matchId]);

  // Separate effect for polling
  useEffect(() => {
    if (!match || match.status === "completed" || match.status === "failed") {
      return;
    }
    
    const interval = setInterval(() => {
      fetchMatch(true);
    }, 2000);
    
    return () => clearInterval(interval);
  }, [match?.status, matchId]);

  const fetchMatch = async (silent = false) => {
    if (!silent) setLoading(true);
    else setRefreshing(true);
    
    try {
      const response = await axios.get(`${API}/matches/${matchId}`);
      setMatch(response.data);
    } catch (error) {
      console.error("Failed to fetch match:", error);
      if (!silent) toast.error("Failed to load analysis");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleExportJSON = async () => {
    try {
      const response = await axios.get(`${API}/matches/${matchId}/export/json`, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `squashsense_${matchId}.json`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success("JSON exported");
    } catch (error) {
      toast.error("Export failed");
    }
  };

  const handleExportPDF = async () => {
    try {
      const response = await axios.get(`${API}/matches/${matchId}/export/pdf`, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `squashsense_${matchId}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast.success("PDF exported");
    } catch (error) {
      toast.error("Export failed");
    }
  };

  const handleCorrectionSaved = (shotIndex, newShotType, newPlayer) => {
    if (!match) return;
    const updatedShots = [...match.shots];
    updatedShots[shotIndex] = {
      ...updatedShots[shotIndex],
      shot_type: newShotType,
      player: newPlayer,
      user_corrected: true
    };
    
    // Recalculate shot distribution
    const newDistribution = { drive: 0, drop: 0, boast: 0, volley: 0, lob: 0, kill: 0, serve: 0 };
    updatedShots.forEach(shot => {
      if (newDistribution[shot.shot_type] !== undefined) {
        newDistribution[shot.shot_type]++;
      }
    });
    
    setMatch({
      ...match,
      shots: updatedShots,
      shot_distribution: newDistribution
    });
  };

  const openCorrectionModal = (shot, index) => {
    setCorrectionModal({ isOpen: true, shot, index });
  };

  const formatDuration = (seconds) => {
    if (!seconds) return "--:--";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getShotDistributionData = () => {
    if (!match?.shot_distribution) return [];
    return Object.entries(match.shot_distribution)
      .filter(([_, count]) => count > 0)
      .map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
        color: SHOT_COLORS[name] || "#888"
      }));
  };

  const getRallyData = () => {
    if (!match?.rallies) return [];
    return match.rallies.slice(0, 15).map((rally, idx) => ({
      name: `R${rally.rally_number}`,
      shots: rally.shot_count,
      winner: rally.winner === "player1" ? "P1" : "P2"
    }));
  };

  const getPlayerComparisonData = () => {
    if (!match?.player1_stats || !match?.player2_stats) return [];
    const p1 = match.player1_stats;
    const p2 = match.player2_stats;
    return [
      { name: "Shots", player1: p1.shots || 0, player2: p2.shots || 0 },
      { name: "Winners", player1: p1.winners || 0, player2: p2.winners || 0 },
      { name: "Forehand", player1: p1.forehand || 0, player2: p2.forehand || 0 },
      { name: "Backhand", player1: p1.backhand || 0, player2: p2.backhand || 0 }
    ];
  };

  const getMovementData = () => {
    if (!match?.movement_data) return { player1: [], player2: [] };
    const p1 = match.movement_data.filter(m => m.player === "player1").map(m => ({
      x: m.x * 100,
      y: m.y * 100
    }));
    const p2 = match.movement_data.filter(m => m.player === "player2").map(m => ({
      x: m.x * 100,
      y: m.y * 100
    }));
    return { player1: p1, player2: p2 };
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#050505] flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Loading analysis...</p>
        </div>
      </div>
    );
  }

  if (!match) {
    return (
      <div className="min-h-screen bg-[#050505] flex items-center justify-center">
        <div className="text-center">
          <p className="text-muted-foreground mb-4">Match not found</p>
          <Link to="/history">
            <Button>Go to History</Button>
          </Link>
        </div>
      </div>
    );
  }

  const isProcessing = match.status === "processing" || match.status === "pending";
  const shotData = getShotDistributionData();
  const rallyData = getRallyData();
  const playerData = getPlayerComparisonData();
  const movementData = getMovementData();

  return (
    <div className="min-h-screen bg-[#050505]">
      {/* Navigation */}
      <nav className="border-b border-border/50 bg-background/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link to="/history" className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
              <ArrowLeft className="w-4 h-4" />
              <span>Back</span>
            </Link>
            <div className="h-6 w-px bg-border"></div>
            <Link to="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
                <Target className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="font-heading text-xl font-bold tracking-tight">SQUASHSENSE</span>
            </Link>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              onClick={runFullAnalysis}
              disabled={fullAnalysis?.status === "running"}
              className="bg-primary text-primary-foreground hover:bg-primary/90 relative z-10"
              data-testid="full-analysis-btn"
            >
              {fullAnalysis?.status === "running"
                ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    {fullAnalysis.stage_label || "Analyzing"}…
                    <span className="ml-2 text-xs opacity-80">
                      {(fullAnalysis.stages_done?.length || 0)}/{fullAnalysis.total_stages || 4}
                    </span></>
                : <><Brain className="w-4 h-4 mr-2" /> Analyze Full Match</>}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => fetchMatch()}
              disabled={refreshing}
              data-testid="refresh-btn"
              className="relative z-10"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button 
                  variant="outline" 
                  className="border-border"
                  disabled={isProcessing}
                  data-testid="export-dropdown"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="bg-card border-border">
                <DropdownMenuItem onClick={handleExportJSON} className="cursor-pointer">
                  <FileJson className="w-4 h-4 mr-2" />
                  Export JSON
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleExportPDF} className="cursor-pointer">
                  <FileText className="w-4 h-4 mr-2" />
                  Export PDF
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <h1 className="font-heading text-3xl font-black tracking-tight">{match.title}</h1>
            <span className={`px-3 py-1 rounded text-xs font-mono ${
              match.status === "completed" ? "badge-success" :
              match.status === "failed" ? "badge-error" : "badge-warning"
            }`}>
              {match.status.toUpperCase()}
            </span>
          </div>
          <p className="text-muted-foreground">
            {new Date(match.upload_time).toLocaleDateString()} • Duration: {formatDuration(match.duration)}
          </p>
          
          {isProcessing && (
            <div className="mt-4 bg-card border border-border rounded-lg p-4">
              <div className="flex items-center gap-3 mb-2">
                <Loader2 className="w-5 h-5 animate-spin text-primary" />
                <span className="font-medium">Analyzing match...</span>
                <span className="text-muted-foreground">{match.progress}%</span>
              </div>
              <Progress value={match.progress} className="h-2" />
            </div>
          )}
        </div>

        {/* Player Identification Cards */}
        {(match.player1_frame || match.player2_frame) && (
          <div className="grid grid-cols-2 gap-6 mb-8">
            <div className="bg-card border border-primary/50 rounded-lg p-4 flex items-center gap-4">
              {match.player1_frame ? (
                <img 
                  src={`data:image/jpeg;base64,${match.player1_frame}`}
                  alt="Player 1"
                  className="w-20 h-16 object-cover rounded-lg border-2 border-primary"
                />
              ) : (
                <div className="w-20 h-16 bg-primary/20 rounded-lg flex items-center justify-center">
                  <Users className="w-8 h-8 text-primary" />
                </div>
              )}
              <div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-primary"></span>
                  <span className="font-heading text-xl font-bold text-primary">{match.player1_name || "Player 1"}</span>
                </div>
                <p className="text-sm text-muted-foreground">Shown in yellow/green in charts</p>
              </div>
            </div>
            
            <div className="bg-card border border-[#00F0FF]/50 rounded-lg p-4 flex items-center gap-4">
              {match.player2_frame ? (
                <img 
                  src={`data:image/jpeg;base64,${match.player2_frame}`}
                  alt="Player 2"
                  className="w-20 h-16 object-cover rounded-lg border-2 border-[#00F0FF]"
                />
              ) : (
                <div className="w-20 h-16 bg-[#00F0FF]/20 rounded-lg flex items-center justify-center">
                  <Users className="w-8 h-8 text-[#00F0FF]" />
                </div>
              )}
              <div>
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-[#00F0FF]"></span>
                  <span className="font-heading text-xl font-bold text-[#00F0FF]">{match.player2_name || "Player 2"}</span>
                </div>
                <p className="text-sm text-muted-foreground">Shown in cyan/blue in charts</p>
              </div>
            </div>
          </div>
        )}

        {/* Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
          <div className="stat-card rounded-lg" data-testid="stat-total-shots">
            <div className="flex items-center gap-2 text-muted-foreground mb-2">
              <Zap className="w-4 h-4" />
              <span className="text-sm">Total Shots</span>
            </div>
            <div className="font-mono text-4xl font-bold text-primary">{match.total_shots}</div>
          </div>
          
          <div className="stat-card rounded-lg" data-testid="stat-total-rallies">
            <div className="flex items-center gap-2 text-muted-foreground mb-2">
              <Activity className="w-4 h-4" />
              <span className="text-sm">Total Rallies</span>
            </div>
            <div className="font-mono text-4xl font-bold text-[#00F0FF]">{match.total_rallies}</div>
          </div>
          
          <div className="stat-card rounded-lg" data-testid="stat-avg-rally">
            <div className="flex items-center gap-2 text-muted-foreground mb-2">
              <TrendingUp className="w-4 h-4" />
              <span className="text-sm">Avg Rally Length</span>
            </div>
            <div className="font-mono text-4xl font-bold text-[#30D158]">
              {match.total_rallies > 0 ? (match.total_shots / match.total_rallies).toFixed(1) : "0"}
            </div>
          </div>
          
          <div className="stat-card rounded-lg" data-testid="stat-duration">
            <div className="flex items-center gap-2 text-muted-foreground mb-2">
              <Clock className="w-4 h-4" />
              <span className="text-sm">Duration</span>
            </div>
            <div className="font-mono text-4xl font-bold">{formatDuration(match.duration)}</div>
          </div>
        </div>

        {/* Analysis Tabs */}
        <Tabs defaultValue="shots" className="space-y-6">
          <TabsList className="bg-card border border-border p-1 flex-wrap">
            <TabsTrigger value="shots" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Shot Analysis
            </TabsTrigger>
            <TabsTrigger value="rallies" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Rally Breakdown
            </TabsTrigger>
            <TabsTrigger value="players" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Player Stats
            </TabsTrigger>
            <TabsTrigger value="movement" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Movement
            </TabsTrigger>
            <TabsTrigger value="insights" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              Insights
            </TabsTrigger>
            <TabsTrigger value="training" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Brain className="w-4 h-4 mr-1" />
              AI Training
            </TabsTrigger>
          </TabsList>

          {/* Shot Analysis Tab */}
          <TabsContent value="shots" className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              {/* Pie Chart */}
              <div className="stat-card rounded-lg" data-testid="shot-distribution-chart">
                <h3 className="font-heading text-xl font-bold mb-4">Shot Distribution</h3>
                {shotData.length > 0 ? (
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={shotData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={90}
                          paddingAngle={2}
                          dataKey="value"
                        >
                          {shotData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ 
                            background: '#0A0A0A', 
                            border: '1px solid #27272A',
                            borderRadius: '8px'
                          }}
                        />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-64 flex items-center justify-center text-muted-foreground">
                    No shot data available
                  </div>
                )}
              </div>

              {/* Shot List */}
              <div className="stat-card rounded-lg">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-heading text-xl font-bold">Shot Breakdown</h3>
                  <span className="text-xs text-muted-foreground flex items-center gap-1">
                    <Edit2 className="w-3 h-3" /> Click shots below to correct
                  </span>
                </div>
                <div className="space-y-3 mb-6">
                  {Object.entries(match.shot_distribution || {}).map(([shot, count]) => (
                    <div key={shot} className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div 
                          className="w-3 h-3 rounded-full"
                          style={{ background: SHOT_COLORS[shot] || "#888" }}
                        />
                        <span className="capitalize">{shot}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-bold">{count}</span>
                        <span className="text-muted-foreground text-sm">
                          ({match.total_shots > 0 ? ((count / match.total_shots) * 100).toFixed(1) : 0}%)
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                
                {/* Individual Shots - Clickable for correction */}
                <div className="border-t border-border pt-4">
                  <h4 className="text-sm font-medium text-muted-foreground mb-3">All Shots (click to correct)</h4>
                  <ScrollArea className="h-48">
                    <div className="space-y-2">
                      {(match.shots || []).map((shot, idx) => (
                        <div 
                          key={idx}
                          onClick={() => openCorrectionModal(shot, idx)}
                          className={`flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                            shot.user_corrected 
                              ? 'bg-green-500/10 border border-green-500/30' 
                              : 'bg-muted/30 hover:bg-muted/50'
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <span className="text-xs text-muted-foreground w-8">#{idx + 1}</span>
                            <div 
                              className="w-2 h-2 rounded-full"
                              style={{ background: SHOT_COLORS[shot.shot_type] || "#888" }}
                            />
                            <span className="text-sm capitalize">{shot.shot_type}</span>
                          </div>
                          <div className="flex items-center gap-3">
                            <span className={`text-xs ${shot.player === 'player1' ? 'text-primary' : 'text-[#00F0FF]'}`}>
                              {shot.player === 'player1' ? 'P1' : 'P2'}
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {shot.timestamp?.toFixed(1)}s
                            </span>
                            {shot.user_corrected && (
                              <span className="text-xs text-green-400">✓</span>
                            )}
                            <Edit2 className="w-3 h-3 text-muted-foreground" />
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              </div>
            </div>
          </TabsContent>

          {/* Rally Breakdown Tab */}
          <TabsContent value="rallies" className="space-y-6">
            {/* Rally segmentation control */}
            <div className="stat-card rounded-lg">
              <h3 className="font-heading text-xl font-bold mb-1">Rally Segmentation</h3>
              <p className="text-xs text-muted-foreground mb-4">
                Detects when each rally starts and ends from ball trajectory. Watch each clip and tag the outcome to track the score.
              </p>
              {/* Court calibration status + button */}
              <div className="flex items-center gap-3 mb-4 p-3 rounded-lg bg-background border border-border">
                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${isCalibrated ? "bg-green-500" : "bg-yellow-500"}`} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium">{isCalibrated ? "Court calibrated" : "Court not calibrated"}</div>
                  <div className="text-xs text-muted-foreground">
                    {isCalibrated
                      ? "Tin hits and service box detection are active."
                      : "Calibrate the court to enable tin detection, service box sensing, and automatic point outcomes."}
                  </div>
                </div>
                <Button size="sm" variant={isCalibrated ? "outline" : "default"}
                  className={isCalibrated ? "" : "bg-primary text-primary-foreground"}
                  onClick={() => setShowCalibration(true)}>
                  {isCalibrated ? "Recalibrate" : "Calibrate Court"}
                </Button>
              </div>

              {/* Player identification status + button */}
              <div className="flex items-center gap-3 mb-4 p-3 rounded-lg bg-background border border-border">
                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${playersIdentified ? "bg-green-500" : "bg-yellow-500"}`} />
                {match?.player1_frame && (
                  <img src={`data:image/jpeg;base64,${match.player1_frame}`} alt="P1"
                    className="w-8 h-10 object-cover rounded border border-border" />
                )}
                {match?.player2_frame && (
                  <img src={`data:image/jpeg;base64,${match.player2_frame}`} alt="P2"
                    className="w-8 h-10 object-cover rounded border border-border" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium">
                    {playersIdentified
                      ? <>{playerNames[1]} <span className="text-muted-foreground">vs</span> {playerNames[2]}</>
                      : "Players not identified"}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {playersIdentified
                      ? "Names + shirt colours locked. Analysis will use these names."
                      : "Name each player so the report says who's who (needs calibration first)."}
                  </div>
                </div>
                <Button size="sm" variant={playersIdentified ? "outline" : "default"}
                  className={playersIdentified ? "" : "bg-primary text-primary-foreground"}
                  disabled={!isCalibrated}
                  title={isCalibrated ? "" : "Calibrate the court first"}
                  onClick={() => setShowIdentify(true)}>
                  {playersIdentified ? "Edit Players" : "Identify Players"}
                </Button>
              </div>

              <div className="flex items-end gap-4 flex-wrap mb-4">
                <label className="text-xs text-muted-foreground">Start (s)
                  <input type="number" value={rallyStart} onChange={(e) => setRallyStart(e.target.value)}
                    className="block w-24 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
                <label className="text-xs text-muted-foreground">Duration (s)
                  <input type="number" value={rallyDur} onChange={(e) => setRallyDur(e.target.value)}
                    className="block w-24 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
                <Button onClick={segmentRallies} disabled={rallyRunning}
                  className="bg-primary text-primary-foreground hover:bg-primary/90">
                  {rallyRunning ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Segmenting…</>
                    : <><Activity className="w-4 h-4 mr-2" /> Detect Rallies</>}
                </Button>
              </div>

              {rallySeg ? (() => {
                // Display rallies in TIME order with sequential numbers (1,2,3…);
                // internal rally_id (used for clips/API) can be non-sequential after edits.
                const sortedRallies = [...(rallySeg.rallies || [])].sort((a, b) => a.start_t - b.start_t);
                const posOf = {}; sortedRallies.forEach((r, i) => { posOf[r.rally_id] = i + 1; });
                const score = computeScore(sortedRallies, rallyOutcomes);
                const tagged = Object.keys(rallyOutcomes).length;
                const total = sortedRallies.length;
                return (
                  <div>
                    {/* Summary stats */}
                    <div className="grid grid-cols-3 gap-3 mb-4 text-center">
                      <div><div className="text-2xl font-bold font-mono text-primary">{rallySeg.num_rallies}</div><div className="text-xs text-muted-foreground">rallies</div></div>
                      <div><div className="text-2xl font-bold font-mono">{rallySeg.active_play_pct}%</div><div className="text-xs text-muted-foreground">active play</div></div>
                      <div><div className="text-2xl font-bold font-mono">{rallySeg.span_s}s</div><div className="text-xs text-muted-foreground">analyzed</div></div>
                    </div>
                    <div className="flex items-center gap-3 mb-4 text-xs flex-wrap">
                      <span className={`px-2 py-0.5 rounded-full border ${
                        rallySeg.method === "audio" ? "border-green-700/40 text-green-400 bg-green-900/20"
                        : rallySeg.method === "dual_signal_v2" ? "border-border text-muted-foreground" : "border-border text-muted-foreground"}`}>
                        {rallySeg.method === "audio" ? `🔊 Audio (${rallySeg.total_strikes || 0} ball strikes)`
                          : rallySeg.method === "dual_signal_v2" ? "Event-based (ball+players)" : "Motion-only (legacy)"}
                      </span>
                      <span className="text-muted-foreground">Fix any wrong boundaries with Merge/Split, then →</span>
                      <button
                        onClick={async () => {
                          try {
                            const res = await axios.post(`${API}/analysis/rallies/${matchId}/confirm-boundaries`);
                            toast.success(`Saved ${res.data.labeled_rallies} boundaries as ground truth — thank you!`);
                          } catch (e) { toast.error("Could not save"); }
                        }}
                        className="px-3 py-1 rounded-full bg-primary/20 text-primary border border-primary/40 hover:bg-primary/30 font-medium">
                        ✓ Confirm boundaries (save as ground truth)
                      </button>
                    </div>

                    {/* SCORECARD (rules engine: live score, game point, running progression) */}
                    {(() => {
                      const p1 = scoreboard?.final ? scoreboard.final.current_game.p1 : score.p1;
                      const p2 = scoreboard?.final ? scoreboard.final.current_game.p2 : score.p2;
                      const hi = Math.max(p1, p2), lo = Math.min(p1, p2);
                      const leader = p1 > p2 ? 1 : p2 > p1 ? 2 : null;
                      const gamePoint = leader && hi >= 10 && (hi - lo) >= 1 && (hi + 1 - lo) >= 2;
                      const gameWon = hi >= 11 && (hi - lo) >= 2;
                      let status, statusColor = "text-muted-foreground";
                      if (scoreboard?.final?.match_over) { status = `${playerNames[scoreboard.final.match_winner]} wins the match`; statusColor = "text-green-400"; }
                      else if (gameWon) { status = `${playerNames[leader]} wins the game ${hi}–${lo}`; statusColor = "text-green-400"; }
                      else if (gamePoint) { status = `Game point — ${playerNames[leader]}`; statusColor = "text-yellow-400"; }
                      else if (leader) status = `${playerNames[leader]} leads ${hi}–${lo}`;
                      else status = `Tied ${p1}–${p2}`;
                      return (
                        <div className="mb-4 rounded-lg border border-border bg-background p-4">
                          <div className="flex items-center justify-center gap-6">
                            <div className="text-center">
                              {match?.player1_frame && <img src={`data:image/jpeg;base64,${match.player1_frame}`} alt={playerNames[1]} className="w-10 h-14 object-cover rounded border-2 border-primary mx-auto mb-1" />}
                              <div className={`text-4xl font-bold font-mono ${leader === 1 ? "text-primary" : "text-primary/60"}`}>{p1}</div>
                              <div className="text-xs mt-0.5 flex items-center gap-1 justify-center text-primary">
                                {playerNames[1]}{scoreboard?.final?.server === 1 && <span title="serving">●</span>}
                              </div>
                            </div>
                            <div className="text-2xl font-mono text-muted-foreground">–</div>
                            <div className="text-center">
                              {match?.player2_frame && <img src={`data:image/jpeg;base64,${match.player2_frame}`} alt={playerNames[2]} className="w-10 h-14 object-cover rounded border-2 border-[#00F0FF] mx-auto mb-1" />}
                              <div className={`text-4xl font-bold font-mono ${leader === 2 ? "text-[#00F0FF]" : "text-[#00F0FF]/60"}`}>{p2}</div>
                              <div className="text-xs mt-0.5 flex items-center gap-1 justify-center text-[#00F0FF]">
                                {playerNames[2]}{scoreboard?.final?.server === 2 && <span title="serving">●</span>}
                              </div>
                            </div>
                          </div>
                          <div className={`text-center text-sm font-medium mt-3 ${statusColor}`}>{status}</div>
                          <div className="text-center text-[11px] text-muted-foreground mt-0.5">
                            {tagged}/{total} rallies tagged · PAR, first to 11 (win by 2)
                          </div>
                          {/* running point-by-point progression */}
                          {scoreboard?.running?.length > 0 && (
                            <div className="flex flex-wrap gap-1 justify-center mt-3 pt-3 border-t border-border/40">
                              {scoreboard.running.filter(e => e.winner).map((e, i) => (
                                <span key={i}
                                  className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                                  style={{ background: e.winner === 1 ? "#DFFF0022" : "#00F0FF22", color: e.winner === 1 ? "#DFFF00" : "#00F0FF" }}
                                  title={`Rally ${e.rally_id}: ${playerNames[e.winner]}`}>
                                  {e.p1}-{e.p2}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      );
                    })()}

                    {/* Timeline bar */}
                    <div className="relative h-6 bg-background rounded mb-4 overflow-hidden border border-border">
                      {sortedRallies.map((r) => {
                        const s0 = rallySeg.start_s || 0;
                        const left = ((r.start_t - s0) / rallySeg.span_s) * 100;
                        const w = (r.duration_s / rallySeg.span_s) * 100;
                        const o = rallyOutcomes[r.rally_id];
                        const color = o === "p1" || o === "stroke_p1" ? "#DFFF00"
                          : o === "p2" || o === "stroke_p2" ? "#00F0FF"
                          : o === "let" ? "#888"
                          : o === "warmup" ? "#333"
                          : "#6366f1";
                        return <div key={r.rally_id} className="absolute top-0 bottom-0 cursor-pointer"
                          style={{ left: `${left}%`, width: `${Math.max(0.5, w)}%`, background: color, opacity: 0.8 }}
                          onClick={() => setExpandedRally(r.rally_id === expandedRally ? null : r.rally_id)}
                          title={`Rally ${posOf[r.rally_id]}: ${r.start_t}s–${r.end_t}s`} />;
                      })}
                    </div>

                    {/* Rally cards */}
                    <div className="space-y-3">
                      {sortedRallies.map((r, rIdx) => {
                        const o = rallyOutcomes[r.rally_id];
                        const isOpen = expandedRally === r.rally_id;
                        const isLast = rIdx === sortedRallies.length - 1;
                        const OUTCOMES = [
                          { id: "p1", label: `${playerNames[1]} Won`, color: "bg-primary text-primary-foreground" },
                          { id: "p2", label: `${playerNames[2]} Won`, color: "bg-[#00F0FF]/20 text-[#00F0FF] border border-[#00F0FF]/40" },
                          { id: "let", label: "Let", color: "bg-muted text-muted-foreground" },
                          { id: "stroke_p1", label: `Stroke (${playerNames[1]})`, color: "bg-orange-500/20 text-orange-400 border border-orange-500/40" },
                          { id: "stroke_p2", label: `Stroke (${playerNames[2]})`, color: "bg-purple-500/20 text-purple-400 border border-purple-500/40" },
                          { id: "warmup", label: "Warm-up / Skip", color: "bg-zinc-800 text-zinc-500 border border-zinc-700" },
                        ];
                        return (
                          <div key={r.rally_id} className={`rounded-lg border ${o ? "border-border" : r.end_reason === "tin" ? "border-red-900/50" : "border-dashed border-border/60"} bg-background/50`}>
                            {/* Header row */}
                            <div className="flex items-center gap-3 px-3 py-2 cursor-pointer"
                              onClick={() => setExpandedRally(isOpen ? null : r.rally_id)}>
                              <span className="text-primary font-mono font-bold w-16 text-sm">Rally {posOf[r.rally_id]}</span>
                              <span className="text-muted-foreground text-xs font-mono">{r.start_t}s → {r.end_t}s ({r.duration_s}s)</span>
                              <span className="text-xs text-muted-foreground ml-1">~{r.shots} shots</span>
                              <div className="ml-auto flex items-center gap-2">
                                {/* Auto-detected end event badge */}
                                {r.end_reason === "tin" && !o && (
                                  <span className="text-xs px-2 py-0.5 rounded-full bg-red-900/40 text-red-400 border border-red-800/40">
                                    🎯 Tin detected
                                  </span>
                                )}
                                {o && (
                                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                                    o === "p1" || o === "stroke_p1" ? "bg-primary/20 text-primary"
                                    : o === "p2" || o === "stroke_p2" ? "bg-[#00F0FF]/20 text-[#00F0FF]"
                                    : o === "warmup" ? "bg-zinc-800 text-zinc-500"
                                    : "bg-muted text-muted-foreground"}`}>
                                    {OUTCOMES.find(x => x.id === o)?.label}
                                  </span>
                                )}
                                <span className="text-muted-foreground text-xs">{isOpen ? "▲" : "▼"}</span>
                              </div>
                            </div>
                            {/* Expanded: video clip + tagging */}
                            {isOpen && (
                              <div className="px-3 pb-3 border-t border-border/40 pt-3 space-y-3">
                                <video
                                  key={`${r.rally_id}-${clipVersion}`}
                                  ref={expandedVideoRef}
                                  controls
                                  className="w-full rounded max-h-64 bg-black"
                                  src={`${process.env.REACT_APP_BACKEND_URL}/api/analysis/rallies/${matchId}/clip/${r.rally_id}?v=${clipVersion}`}
                                />
                                {/* Player photos so you can match on-court player → name */}
                                {(match?.player1_frame || match?.player2_frame) && (
                                  <div className="flex items-center gap-4 text-xs">
                                    {match?.player1_frame && (
                                      <div className="flex items-center gap-2">
                                        <img src={`data:image/jpeg;base64,${match.player1_frame}`} alt={playerNames[1]}
                                          className="w-9 h-12 object-cover rounded border-2 border-primary" />
                                        <span className="text-primary font-medium">{playerNames[1]}</span>
                                      </div>
                                    )}
                                    {match?.player2_frame && (
                                      <div className="flex items-center gap-2">
                                        <img src={`data:image/jpeg;base64,${match.player2_frame}`} alt={playerNames[2]}
                                          className="w-9 h-12 object-cover rounded border-2 border-[#00F0FF]" />
                                        <span className="text-[#00F0FF] font-medium">{playerNames[2]}</span>
                                      </div>
                                    )}
                                  </div>
                                )}
                                <div>
                                  <div className="text-xs text-muted-foreground mb-2">Who got the point?</div>
                                  <div className="flex flex-wrap gap-2">
                                    {OUTCOMES.map(({ id, label, color }) => (
                                      <button key={id}
                                        onClick={() => tagOutcome(r.rally_id, id)}
                                        className={`px-3 py-1.5 rounded text-xs font-medium transition-all ${o === id ? color + " ring-2 ring-offset-1 ring-offset-background ring-white/30" : "bg-muted/40 text-muted-foreground hover:bg-muted"}`}>
                                        {label}
                                      </button>
                                    ))}
                                  </div>
                                </div>
                                {/* Fix boundaries: Merge (this clip continues into the next) or Split (this clip has two rallies) */}
                                <div className="border-t border-border/30 pt-2">
                                  <div className="text-xs text-muted-foreground mb-2">Boundary not right?</div>
                                  <div className="flex flex-wrap gap-2">
                                    {!isLast && (
                                      <button
                                        onClick={async () => {
                                          try {
                                            await axios.post(`${API}/analysis/rallies/${matchId}/${r.rally_id}/merge_next`);
                                            await new Promise(rs => setTimeout(rs, 1200));
                                            const u = await axios.get(`${API}/analysis/rallies/${matchId}`);
                                            setRallySeg(u.data); setClipVersion(v => v + 1); setExpandedRally(r.rally_id);
                                            toast.success(`Merged with next — clip updated`);
                                          } catch (e) { toast.error("Merge failed"); }
                                        }}
                                        className="px-3 py-1.5 rounded text-xs font-medium bg-yellow-500/10 text-yellow-400 border border-yellow-500/30 hover:bg-yellow-500/20 transition-all">
                                        ↔ Merge with Rally {posOf[r.rally_id] + 1} (one rally)
                                      </button>
                                    )}
                                    <button
                                      onClick={async () => {
                                        // Split at the EXACT moment you've paused/scrubbed the clip to.
                                        const vt = expandedVideoRef.current?.currentTime;
                                        if (vt == null) { toast.error("Play the clip to a point first"); return; }
                                        // clip time → absolute video time (clip starts at start_t - buffer)
                                        const absT = Math.max(0, r.start_t - CLIP_BUFFER) + vt;
                                        try {
                                          const res = await axios.post(
                                            `${API}/analysis/rallies/${matchId}/${r.rally_id}/split?at_t=${absT.toFixed(2)}`);
                                          await new Promise(rs => setTimeout(rs, 1200)); // let clips re-cut
                                          const u = await axios.get(`${API}/analysis/rallies/${matchId}`);
                                          setRallySeg(u.data);
                                          setClipVersion(v => v + 1);           // force fresh clips
                                          setExpandedRally(res.data.new_id);     // jump to the REMAINING segment to keep splitting
                                          toast.success(`Split at ${res.data.split_at}s — continue on the remaining part`);
                                        } catch (e) { toast.error(e?.response?.data?.detail || "Split failed"); }
                                      }}
                                      className="px-3 py-1.5 rounded text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-500/20 transition-all">
                                      ✂ Split here (at the paused moment)
                                    </button>
                                    <button
                                      onClick={async () => {
                                        try {
                                          const res = await axios.post(`${API}/analysis/rallies/${matchId}/${r.rally_id}/split`);
                                          await new Promise(rs => setTimeout(rs, 1200));
                                          const u = await axios.get(`${API}/analysis/rallies/${matchId}`);
                                          setRallySeg(u.data); setClipVersion(v => v + 1); setExpandedRally(res.data.new_id);
                                          toast.success(`Auto-split at ${res.data.split_at}s — continue on the remaining part`);
                                        } catch (e) { toast.error("Split failed"); }
                                      }}
                                      className="px-3 py-1.5 rounded text-xs font-medium bg-muted/40 text-muted-foreground border border-border hover:bg-muted transition-all">
                                      auto-split
                                    </button>
                                  </div>
                                  <p className="text-[11px] text-muted-foreground mt-1">
                                    To split precisely: pause the clip exactly where the next rally's serve begins, then click <span className="text-indigo-400">Split here</span>.
                                  </p>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })() : (
                <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg">
                  Pick a span and click <span className="text-primary">Detect Rallies</span>.
                </div>
              )}
            </div>

            {/* 3D Rally Timeline — full-stack reconstruction (Layers 1-6) */}
            <div className="stat-card rounded-lg">
              <div className="flex items-start justify-between mb-1">
                <div>
                  <h3 className="font-heading text-xl font-bold mb-1">3D Rally Timeline</h3>
                  <p className="text-xs text-muted-foreground max-w-xl">
                    Runs the full perception stack — court 3D, ball 3D, players, pose, events, rules —
                    to reconstruct each rally's shots, tin/out events and outcome. Confidence-aware:
                    it tells you when it can't yet be trusted (needs a clean calibration + a strong ball model).
                  </p>
                </div>
                <Button onClick={buildTimeline3d} disabled={tl3dRunning || !isCalibrated}
                  title={isCalibrated ? "" : "Calibrate the court first"}
                  className="bg-primary text-primary-foreground hover:bg-primary/90 shrink-0">
                  {tl3dRunning
                    ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Reconstructing…</>
                    : <><Brain className="w-4 h-4 mr-2" /> Build 3D Timeline</>}
                </Button>
              </div>
              {tl3dRunning && (
                <div className="text-xs text-muted-foreground py-4 text-center">
                  Running ball + player + pose + 3D reconstruction across rallies — a couple of minutes.
                </div>
              )}
              {timeline3d && !tl3dRunning ? (
                <div className="mt-4"><Timeline3DPanel data={timeline3d} names={playerNames} /></div>
              ) : !tl3dRunning && (
                <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg mt-3">
                  {isCalibrated
                    ? <>Run rally segmentation, then click <span className="text-primary">Build 3D Timeline</span>.</>
                    : <>Calibrate the court first — the 3D timeline needs it.</>}
                </div>
              )}
            </div>

            <div className="stat-card rounded-lg" data-testid="rally-chart">
              <h3 className="font-heading text-xl font-bold mb-4">Rally Length Distribution</h3>
              {rallyData.length > 0 ? (
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={rallyData}>
                      <XAxis dataKey="name" stroke="#52525B" />
                      <YAxis stroke="#52525B" />
                      <Tooltip 
                        contentStyle={{ 
                          background: '#0A0A0A', 
                          border: '1px solid #27272A',
                          borderRadius: '8px'
                        }}
                      />
                      <Bar 
                        dataKey="shots" 
                        fill="#DFFF00"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-72 flex items-center justify-center text-muted-foreground">
                  No rally data available
                </div>
              )}
            </div>

            {/* Rally Timeline */}
            <div className="stat-card rounded-lg">
              <h3 className="font-heading text-xl font-bold mb-4">Rally Timeline</h3>
              <ScrollArea className="h-64">
                <div className="space-y-2">
                  {(match.rallies || []).map((rally, idx) => (
                    <div 
                      key={idx}
                      className="flex items-center gap-4 p-3 bg-muted/30 rounded-lg"
                    >
                      <div className="font-mono text-sm text-muted-foreground w-12">
                        R{rally.rally_number}
                      </div>
                      <div className="flex-1">
                        <div 
                          className="h-6 rounded-sm"
                          style={{
                            width: `${Math.min(rally.shot_count * 10, 100)}%`,
                            background: rally.winner === "player1" ? "#DFFF00" : "#00F0FF"
                          }}
                        />
                      </div>
                      <div className="text-sm">
                        <span className="font-mono font-bold">{rally.shot_count}</span>
                        <span className="text-muted-foreground"> shots</span>
                      </div>
                      <div className={`text-xs font-mono ${
                        rally.winner === "player1" ? "text-primary" : "text-[#00F0FF]"
                      }`}>
                        {rally.winner === "player1" ? "P1" : "P2"} - {rally.winning_shot}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </TabsContent>

          {/* Player Stats Tab */}
          <TabsContent value="players" className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              {/* Player 1 */}
              <div className="stat-card rounded-lg border-primary/30" data-testid="player1-stats">
                <div className="flex items-center gap-4 mb-4">
                  {match.player1_frame ? (
                    <img 
                      src={`data:image/jpeg;base64,${match.player1_frame}`}
                      alt={match.player1_name || "Player 1"}
                      className="w-16 h-12 object-cover rounded-lg border border-primary/50"
                    />
                  ) : (
                    <div className="w-16 h-12 rounded-lg bg-primary/20 flex items-center justify-center">
                      <Users className="w-6 h-6 text-primary" />
                    </div>
                  )}
                  <div>
                    <h3 className="font-heading text-xl font-bold text-primary">{match.player1_name || "Player 1"}</h3>
                    <p className="text-sm text-muted-foreground">Statistics</p>
                  </div>
                </div>
                <div className="space-y-3">
                  {Object.entries(match.player1_stats || {}).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-center">
                      <span className="text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</span>
                      <span className="font-mono font-bold">{typeof value === 'number' ? value.toFixed(value % 1 === 0 ? 0 : 1) : value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Player 2 */}
              <div className="stat-card rounded-lg border-[#00F0FF]/30" data-testid="player2-stats">
                <div className="flex items-center gap-4 mb-4">
                  {match.player2_frame ? (
                    <img 
                      src={`data:image/jpeg;base64,${match.player2_frame}`}
                      alt={match.player2_name || "Player 2"}
                      className="w-16 h-12 object-cover rounded-lg border border-[#00F0FF]/50"
                    />
                  ) : (
                    <div className="w-16 h-12 rounded-lg bg-[#00F0FF]/20 flex items-center justify-center">
                      <Users className="w-6 h-6 text-[#00F0FF]" />
                    </div>
                  )}
                  <div>
                    <h3 className="font-heading text-xl font-bold text-[#00F0FF]">{match.player2_name || "Player 2"}</h3>
                    <p className="text-sm text-muted-foreground">Statistics</p>
                  </div>
                </div>
                <div className="space-y-3">
                  {Object.entries(match.player2_stats || {}).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-center">
                      <span className="text-muted-foreground capitalize">{key.replace(/_/g, ' ')}</span>
                      <span className="font-mono font-bold">{typeof value === 'number' ? value.toFixed(value % 1 === 0 ? 0 : 1) : value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Comparison Chart */}
            <div className="stat-card rounded-lg" data-testid="player-comparison-chart">
              <h3 className="font-heading text-xl font-bold mb-4">Player Comparison</h3>
              {playerData.length > 0 ? (
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={playerData} layout="vertical">
                      <XAxis type="number" stroke="#52525B" />
                      <YAxis type="category" dataKey="name" stroke="#52525B" width={80} />
                      <Tooltip 
                        contentStyle={{ 
                          background: '#0A0A0A', 
                          border: '1px solid #27272A',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend />
                      <Bar dataKey="player1" fill="#DFFF00" name={match.player1_name || "Player 1"} radius={[0, 4, 4, 0]} />
                      <Bar dataKey="player2" fill="#00F0FF" name={match.player2_name || "Player 2"} radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="h-64 flex items-center justify-center text-muted-foreground">
                  No comparison data available
                </div>
              )}
            </div>
          </TabsContent>

          {/* Movement Tab */}
          <TabsContent value="movement" className="space-y-6">

            {/* Court Control — tactical movement analysis (the headline feature) */}
            <div className="stat-card rounded-lg">
              <div className="flex items-start justify-between mb-1">
                <div>
                  <h3 className="font-heading text-xl font-bold mb-1">Court Control</h3>
                  <p className="text-xs text-muted-foreground mb-0 max-w-xl">
                    Tactical movement analysis over the detected rallies (active play only): T-control,
                    court coverage, work rate, and where each player gets pushed — the foundation for "how to beat this opponent".
                  </p>
                </div>
                <Button onClick={analyzeCourtControl} disabled={ccRunning}
                  className="bg-primary text-primary-foreground hover:bg-primary/90 shrink-0">
                  {ccRunning
                    ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing…</>
                    : <><TrendingUp className="w-4 h-4 mr-2" /> Analyze Court Control</>}
                </Button>
              </div>
              {ccRunning && (
                <div className="text-xs text-muted-foreground py-4 text-center">
                  Running player detection across all rallies — this takes a minute or two.
                </div>
              )}
              {courtControl && !ccRunning ? (
                <div className="mt-4"><CourtControlPanel data={courtControl} names={playerNames} /></div>
              ) : !ccRunning && (
                <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg mt-3">
                  Run <span className="text-primary">rally segmentation</span> first (Rallies tab), then click
                  <span className="text-primary"> Analyze Court Control</span>.
                </div>
              )}
            </div>

            {/* Shot Patterns & Error Zones */}
            <div className="stat-card rounded-lg">
              <div className="flex items-start justify-between mb-1">
                <div>
                  <h3 className="font-heading text-xl font-bold mb-1">Shot Patterns &amp; Error Zones</h3>
                  <p className="text-xs text-muted-foreground mb-0 max-w-xl">
                    Where each player hits from, and — once you tag rally outcomes — the court zones
                    where they lose points. Tag outcomes in the Rallies tab for the full picture.
                  </p>
                </div>
                <Button onClick={analyzeShotPatterns} disabled={spRunning}
                  className="bg-primary text-primary-foreground hover:bg-primary/90 shrink-0">
                  {spRunning
                    ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Analyzing…</>
                    : <><Target className="w-4 h-4 mr-2" /> Analyze Shot Patterns</>}
                </Button>
              </div>
              {spRunning && (
                <div className="text-xs text-muted-foreground py-4 text-center">
                  Running ball + player detection across all rallies — a minute or two.
                </div>
              )}
              {shotPatterns && !spRunning ? (
                <div className="mt-4"><ShotPatternsPanel data={shotPatterns} names={playerNames} /></div>
              ) : !spRunning && (
                <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg mt-3">
                  Run <span className="text-primary">rally segmentation</span> first, then click
                  <span className="text-primary"> Analyze Shot Patterns</span>.
                </div>
              )}
            </div>

            {/* Player Detection */}
            <div className="stat-card rounded-lg">
              <h3 className="font-heading text-xl font-bold mb-1">Player Detection</h3>
              <p className="text-xs text-muted-foreground mb-4">
                Tracks both players using YOLO across a video span. Produces a top-down court view, position heatmaps, and T-control stats.
              </p>
              <div className="flex items-end gap-4 flex-wrap mb-4">
                <label className="text-xs text-muted-foreground">Start (s)
                  <input type="number" value={playerStart} onChange={e => setPlayerStart(e.target.value)}
                    className="block w-24 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
                <label className="text-xs text-muted-foreground">Duration (s)
                  <input type="number" value={playerDur} onChange={e => setPlayerDur(e.target.value)}
                    className="block w-24 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
                <Button onClick={detectPlayers} disabled={playerRunning}
                  className="bg-primary text-primary-foreground hover:bg-primary/90">
                  {playerRunning
                    ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Detecting…</>
                    : <><Users className="w-4 h-4 mr-2" /> Detect Players</>}
                </Button>
                {playerDetection && !playerRunning && (
                  <span className="text-xs text-[#30D158] pb-1">
                    {playerDetection.total_frames} frames · {playerDetection.span_s}s
                  </span>
                )}
              </div>
              {playerDetection ? (
                <CourtView
                  positions={playerDetection.positions || []}
                  stats={playerDetection.stats || {}}
                  calibration={courtCalibration}
                  names={playerNames}
                />
              ) : (
                <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg">
                  {isCalibrated
                    ? <>Pick a span and click <span className="text-primary">Detect Players</span>.</>
                    : <>Calibrate the court first (Rallies tab) for court-metre stats, then detect players.</>}
                </div>
              )}
            </div>

            {/* Ball Trace — comet-trail video of the ball trajectory */}
            <div className="stat-card rounded-lg">
              <h3 className="font-heading text-xl font-bold mb-1">Ball Trace</h3>
              <p className="text-xs text-muted-foreground mb-4">
                Runs the trained ball model over a window and draws a glowing trail following the ball.
              </p>
              <div className="flex items-end gap-4 flex-wrap mb-4">
                <label className="text-xs text-muted-foreground">Start (s)
                  <input type="number" value={traceStart} onChange={(e) => setTraceStart(e.target.value)}
                    className="block w-24 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
                <label className="text-xs text-muted-foreground">Duration (s)
                  <input type="number" value={traceDur} onChange={(e) => setTraceDur(e.target.value)}
                    className="block w-24 mt-1 bg-background border border-border rounded px-2 py-1 text-sm" /></label>
                <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer pb-1">
                  <input type="checkbox" checked={traceSmooth} onChange={(e) => setTraceSmooth(e.target.checked)} />
                  Smooth (drop outliers)
                </label>
                <Button onClick={generateTrace} disabled={traceState?.status === "running"}
                  className="bg-primary text-primary-foreground hover:bg-primary/90">
                  {traceState?.status === "running"
                    ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating…</>
                    : <><Activity className="w-4 h-4 mr-2" /> Trace Ball</>}
                </Button>
                {traceState?.status === "done" && (
                  <span className="text-xs text-[#30D158] pb-1">
                    tracked {traceState.detections}/{traceState.frame_count} frames
                  </span>
                )}
              </div>
              {traceUrl ? (
                <video src={traceUrl} controls autoPlay loop className="w-full max-w-3xl rounded-lg border border-border bg-black" />
              ) : (
                <div className="text-sm text-muted-foreground py-8 text-center border border-dashed border-border rounded-lg">
                  Pick a rally window and click <span className="text-primary">Trace Ball</span> to see the ball's trajectory.
                </div>
              )}
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {/* Court Heatmap */}
              <div className="stat-card rounded-lg" data-testid="court-heatmap">
                <h3 className="font-heading text-xl font-bold mb-4">Court Position Heatmap</h3>
                <div className="court-container rounded-lg overflow-hidden">
                  {/* Court lines */}
                  <div className="court-line" style={{ top: '25%', left: '10%', right: '10%', height: '1px' }} />
                  <div className="court-line" style={{ top: '50%', left: '10%', right: '10%', height: '1px' }} />
                  <div className="court-line" style={{ left: '50%', top: '10%', bottom: '10%', width: '1px' }} />
                  
                  {/* Movement dots */}
                  {movementData.player1.slice(0, 50).map((pos, idx) => (
                    <div
                      key={`p1-${idx}`}
                      className="absolute w-3 h-3 rounded-full bg-primary/60"
                      style={{
                        left: `${pos.x}%`,
                        top: `${pos.y}%`,
                        transform: 'translate(-50%, -50%)'
                      }}
                    />
                  ))}
                  {movementData.player2.slice(0, 50).map((pos, idx) => (
                    <div
                      key={`p2-${idx}`}
                      className="absolute w-3 h-3 rounded-full bg-[#00F0FF]/60"
                      style={{
                        left: `${pos.x}%`,
                        top: `${pos.y}%`,
                        transform: 'translate(-50%, -50%)'
                      }}
                    />
                  ))}
                  
                  {/* Legend */}
                  <div className="absolute bottom-2 left-2 flex items-center gap-4 text-xs bg-black/60 px-2 py-1 rounded">
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 rounded-full bg-primary" />
                      <span>{match.player1_name || "P1"}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <div className="w-3 h-3 rounded-full bg-[#00F0FF]" />
                      <span>{match.player2_name || "P2"}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Swing Analysis */}
              <div className="stat-card rounded-lg">
                <h3 className="font-heading text-xl font-bold mb-4">Swing Analysis</h3>
                <div className="space-y-6">
                  {(match.swing_analysis || []).map((swing, idx) => (
                    <div key={idx} className="space-y-3">
                      <div className="flex items-center gap-2">
                        <div 
                          className={`w-8 h-8 rounded-full flex items-center justify-center ${
                            swing.player === "player1" ? "bg-primary/20" : "bg-[#00F0FF]/20"
                          }`}
                        >
                          <span className={`text-sm font-bold ${
                            swing.player === "player1" ? "text-primary" : "text-[#00F0FF]"
                          }`}>
                            {swing.player === "player1" ? "P1" : "P2"}
                          </span>
                        </div>
                        <span className="font-medium capitalize">{swing.player.replace("_", " ")}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Forehand</span>
                          <div className="font-mono font-bold">{swing.forehand_count} ({swing.forehand_ratio}%)</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Backhand</span>
                          <div className="font-mono font-bold">{swing.backhand_count} ({swing.backhand_ratio}%)</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Movement & Tactics (kinematics) — Baclig et al. (2020) methodology */}
            {(() => {
              const pm = match.player_metrics || {};
              const p1 = pm.player1, p2 = pm.player2;
              if (!p1 && !p2) {
                return (
                  <div className="stat-card rounded-lg text-sm text-muted-foreground">
                    Movement kinematics appear here once the match is analyzed <span className="text-primary">with the court marked</span>
                    (distance, T-dominance, % behind/left of the T, speeds).
                  </div>
                );
              }
              const rows = [
                ["Distance run", (m) => m ? `${m.distance_m} m` : "—",
                  (m) => m && m.distance_unfiltered_m ? `${Math.max(0, (m.distance_unfiltered_m - m.distance_m)).toFixed(1)} m jitter filtered out` : ""],
                ["Active speed (excl. standing)", (m) => m ? `${m.avg_speed_active_ms} m/s` : "—",
                  (m) => m && m.pct_time_moving != null ? `moving ${m.pct_time_moving}% of the time` : ""],
                ["T-dominance (time near T)", (m) => m ? `${m.t_dominance_pct}%` : "—",
                  (m) => m ? `avg ${m.mean_dist_to_t_m} m from the T` : ""],
                ["% behind the T (pinned deep)", (m) => m && m.pct_behind_t != null ? `${m.pct_behind_t}%` : "—", () => ""],
                ["% on left / backhand side", (m) => m && m.pct_left_of_t != null ? `${m.pct_left_of_t}%` : "—", () => ""],
                ["Court coverage", (m) => m ? `${m.court_coverage_pct}%` : "—", () => ""],
              ];
              return (
                <div className="stat-card rounded-lg">
                  <h3 className="font-heading text-xl font-bold mb-1">Movement &amp; Tactics</h3>
                  <p className="text-xs text-muted-foreground mb-4">
                    Court-grounded kinematics (foot coordinates smoothed via a 5th-order filter). Lower distance from the T
                    and less time pinned behind/left = stronger court control.
                  </p>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <div className="text-muted-foreground text-xs pb-2">Metric</div>
                    <div className="text-primary font-bold text-xs pb-2">{match.player1_name || "Player 1"}</div>
                    <div className="text-[#00F0FF] font-bold text-xs pb-2">{match.player2_name || "Player 2"}</div>
                    {rows.map(([label, val, sub], i) => (
                      <Fragment key={i}>
                        <div className="text-muted-foreground border-t border-border/40 py-2">{label}</div>
                        <div className="font-mono border-t border-border/40 py-2">
                          {val(p1)}{sub(p1) && <div className="text-[10px] text-muted-foreground font-sans">{sub(p1)}</div>}
                        </div>
                        <div className="font-mono border-t border-border/40 py-2">
                          {val(p2)}{sub(p2) && <div className="text-[10px] text-muted-foreground font-sans">{sub(p2)}</div>}
                        </div>
                      </Fragment>
                    ))}
                  </div>
                </div>
              );
            })()}
          </TabsContent>

          {/* Insights Tab */}
          <TabsContent value="insights" className="space-y-6">

            {/* Scouting Report — LLM reasoning layer over all analyses */}
            <div className="stat-card rounded-lg">
              <div className="flex items-start justify-between mb-3">
                <p className="text-xs text-muted-foreground max-w-xl">
                  A coached scouting report synthesised from court control, shot patterns, error zones
                  and tagged outcomes — each player's strengths, weaknesses, and a game plan to beat them.
                </p>
                <Button onClick={generateScouting} disabled={scoutRunning}
                  className="bg-primary text-primary-foreground hover:bg-primary/90 shrink-0">
                  {scoutRunning
                    ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Generating…</>
                    : <><Brain className="w-4 h-4 mr-2" /> {scouting ? "Regenerate" : "Generate"} Scouting Report</>}
                </Button>
              </div>
              {scouting ? (
                <ScoutingReport data={scouting} />
              ) : !scoutRunning && (
                <div className="text-sm text-muted-foreground py-6 text-center border border-dashed border-border rounded-lg">
                  Run rally segmentation, court control and shot patterns (and tag a few outcomes), then
                  click <span className="text-primary">Generate Scouting Report</span>.
                </div>
              )}
            </div>

            <div className="stat-card rounded-lg" data-testid="key-insights">
              <h3 className="font-heading text-xl font-bold mb-4">Key Insights</h3>
              {(match.key_insights || []).length > 0 ? (
                <div className="space-y-4">
                  {match.key_insights.map((insight, idx) => (
                    <div 
                      key={idx}
                      className="flex items-start gap-3 p-4 bg-muted/30 rounded-lg"
                    >
                      <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                        <TrendingUp className="w-4 h-4 text-primary" />
                      </div>
                      <p className="text-foreground">{insight}</p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  {isProcessing ? "Generating insights..." : "No insights available"}
                </div>
              )}
            </div>
          </TabsContent>

          {/* AI Training Tab */}
          <TabsContent value="training" className="space-y-6">
            <ModelScorecard />
            <FlywheelPanel />
            <TrainingStats />
            
            <div className="bg-card border border-border rounded-lg p-6">
              <h3 className="font-heading text-xl font-bold mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-primary" />
                How to Help Train the AI
              </h3>
              <div className="space-y-4 text-sm text-muted-foreground">
                <p>
                  Every correction you make helps improve the AI's accuracy. Here's how it works:
                </p>
                <ol className="list-decimal list-inside space-y-2 ml-2">
                  <li>Go to the <strong className="text-foreground">Shot Analysis</strong> tab</li>
                  <li>Click on any shot in the list to open the correction modal</li>
                  <li>Select the correct shot type and player</li>
                  <li>Your correction is saved as training data</li>
                </ol>
                <p className="mt-4">
                  Once we collect <strong className="text-primary">100+ corrections</strong>, we can fine-tune 
                  a custom model specifically for squash analysis, making future analyses much more accurate.
                </p>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* Shot Correction Modal */}
      {showCalibration && (
        <CourtCalibrationModal
          matchId={matchId}
          onClose={() => setShowCalibration(false)}
          onSaved={(cal) => { setIsCalibrated(true); setCourtCalibration(cal); setShowCalibration(false); }}
        />
      )}

      {showIdentify && (
        <PlayerIdentifyModal
          matchId={matchId}
          existing={match}
          onClose={() => setShowIdentify(false)}
          onSaved={() => { setShowIdentify(false); fetchMatch(); }}
        />
      )}

      <ShotCorrectionModal
        isOpen={correctionModal.isOpen}
        onClose={() => setCorrectionModal({ isOpen: false, shot: null, index: -1 })}
        shot={correctionModal.shot}
        shotIndex={correctionModal.index}
        matchId={matchId}
        onCorrectionSaved={handleCorrectionSaved}
      />
    </div>
  );
};

export default AnalysisPage;
