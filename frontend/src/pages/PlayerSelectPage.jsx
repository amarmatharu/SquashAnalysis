import { useState, useRef, useEffect, useCallback } from "react";
import { useNavigate, useLocation, Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { toast } from "sonner";
import axios from "axios";
import {
  Target,
  ArrowRight,
  Check,
  RotateCcw,
  Loader2,
  Grid3x3
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

// Court floor corners are marked in this order. Real-world meaning matches the
// backend CourtCalibration: front = front wall (far), back = camera side.
const CORNER_DEFS = [
  { key: "front_left", label: "Front-Left", hint: "where the LEFT wall meets the FRONT wall" },
  { key: "front_right", label: "Front-Right", hint: "where the RIGHT wall meets the FRONT wall" },
  { key: "back_right", label: "Back-Right", hint: "near corner on the RIGHT (camera side)" },
  { key: "back_left", label: "Back-Left", hint: "near corner on the LEFT (camera side)" }
];
const COURT_COLOR = "#30D158";

const PlayerSelectPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const canvasRef = useRef(null);
  const imageRef = useRef(null);

  const [matchData, setMatchData] = useState(location.state || null);
  const [currentStep, setCurrentStep] = useState(1);
  const [player1Crop, setPlayer1Crop] = useState(null);
  const [player2Crop, setPlayer2Crop] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [frameLoaded, setFrameLoaded] = useState(false);

  // Court calibration (optional). mode === "court" => clicks place floor corners.
  const [mode, setMode] = useState("players");
  const [courtCorners, setCourtCorners] = useState([]); // [{x,y,nx,ny}]

  useEffect(() => {
    if (!matchData || !matchData.thumbnail) {
      toast.error("No video frame available");
      navigate("/upload");
    }
  }, [matchData, navigate]);

  const redraw = useCallback((p1, p2, corners) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Player 1 selection
    if (p1) {
      ctx.strokeStyle = "#DFFF00";
      ctx.lineWidth = 4;
      ctx.strokeRect(p1.x, p1.y, p1.size, p1.size);
      ctx.fillStyle = "rgba(223, 255, 0, 0.2)";
      ctx.fillRect(p1.x, p1.y, p1.size, p1.size);
      ctx.fillStyle = "#DFFF00";
      ctx.font = "bold 18px Arial";
      ctx.fillText("P1", p1.x + 8, p1.y + 24);
    }

    // Player 2 selection
    if (p2) {
      ctx.strokeStyle = "#00F0FF";
      ctx.lineWidth = 4;
      ctx.strokeRect(p2.x, p2.y, p2.size, p2.size);
      ctx.fillStyle = "rgba(0, 240, 255, 0.2)";
      ctx.fillRect(p2.x, p2.y, p2.size, p2.size);
      ctx.fillStyle = "#00F0FF";
      ctx.font = "bold 18px Arial";
      ctx.fillText("P2", p2.x + 8, p2.y + 24);
    }

    // Court polygon
    if (corners && corners.length > 0) {
      ctx.strokeStyle = COURT_COLOR;
      ctx.lineWidth = 3;
      ctx.beginPath();
      corners.forEach((c, i) => {
        if (i === 0) ctx.moveTo(c.x, c.y);
        else ctx.lineTo(c.x, c.y);
      });
      if (corners.length === 4) {
        ctx.closePath();
        ctx.fillStyle = "rgba(48, 209, 88, 0.15)";
        ctx.fill();
      }
      ctx.stroke();

      corners.forEach((c, i) => {
        ctx.beginPath();
        ctx.arc(c.x, c.y, 7, 0, Math.PI * 2);
        ctx.fillStyle = COURT_COLOR;
        ctx.fill();
        ctx.fillStyle = "#000";
        ctx.font = "bold 12px Arial";
        ctx.fillText(String(i + 1), c.x - 3, c.y + 4);
      });
    }
  }, []);

  const handleCanvasClick = useCallback(
    (e) => {
      if (!frameLoaded) return;
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      // ----- Court calibration mode: place floor corners -----
      if (mode === "court") {
        if (courtCorners.length >= 4) return;
        const corner = {
          x,
          y,
          nx: x / canvas.width, // normalized 0..1 fractions sent to backend
          ny: y / canvas.height
        };
        const next = [...courtCorners, corner];
        setCourtCorners(next);
        redraw(player1Crop, player2Crop, next);
        if (next.length < 4) {
          toast.success(`Marked ${CORNER_DEFS[next.length - 1].label}. Next: ${CORNER_DEFS[next.length].label}`);
        } else {
          toast.success("Court marked! You can confirm now.");
        }
        return;
      }

      // ----- Player selection mode -----
      if (currentStep > 2) return;
      const cropSize = 120;
      const cropX = Math.max(0, Math.min(x - cropSize / 2, canvas.width - cropSize));
      const cropY = Math.max(0, Math.min(y - cropSize / 2, canvas.height - cropSize));

      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = cropSize;
      tempCanvas.height = cropSize;
      const tempCtx = tempCanvas.getContext("2d");
      tempCtx.drawImage(canvas, cropX, cropY, cropSize, cropSize, 0, 0, cropSize, cropSize);
      const base64Data = tempCanvas.toDataURL("image/jpeg", 0.9).split(",")[1];
      const newCrop = { x: cropX, y: cropY, size: cropSize, base64: base64Data };

      if (currentStep === 1) {
        setPlayer1Crop(newCrop);
        setCurrentStep(2);
        redraw(newCrop, null, courtCorners);
        toast.success("Player 1 selected! Now click on Player 2");
      } else if (currentStep === 2) {
        setPlayer2Crop(newCrop);
        setCurrentStep(3);
        redraw(player1Crop, newCrop, courtCorners);
        toast.success("Both players selected!");
      }
    },
    [mode, currentStep, frameLoaded, player1Crop, player2Crop, courtCorners, redraw]
  );

  const handleImageLoad = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    const maxWidth = Math.min(800, window.innerWidth - 48);
    const scale = maxWidth / img.naturalWidth;
    canvas.width = img.naturalWidth * scale;
    canvas.height = img.naturalHeight * scale;
    canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
    setFrameLoaded(true);
  }, []);

  const handleResetPlayers = () => {
    setPlayer1Crop(null);
    setPlayer2Crop(null);
    setCurrentStep(1);
    setMode("players");
    redraw(null, null, courtCorners);
    toast.info("Reset. Click on Player 1");
  };

  const startCourtMode = () => {
    setMode("court");
    setCourtCorners([]);
    redraw(player1Crop, player2Crop, []);
    toast.info(`Click the ${CORNER_DEFS[0].label} floor corner`);
  };

  const resetCourt = () => {
    setCourtCorners([]);
    redraw(player1Crop, player2Crop, []);
    toast.info(`Click the ${CORNER_DEFS[0].label} floor corner`);
  };

  const handleConfirm = async () => {
    if (!player1Crop || !player2Crop) {
      toast.error("Please select both players");
      return;
    }
    setIsSubmitting(true);

    // Save court calibration first (optional, non-fatal) so the perception
    // pass can produce real movement metrics.
    if (courtCorners.length === 4) {
      const [fl, fr, br, bl] = courtCorners;
      try {
        await axios.post(`${API}/matches/${matchData.matchId}/set-court`, {
          front_left: [fl.nx, fl.ny],
          front_right: [fr.nx, fr.ny],
          back_right: [br.nx, br.ny],
          back_left: [bl.nx, bl.ny]
        });
      } catch (err) {
        console.error("Court calibration failed (continuing):", err);
      }
    }

    try {
      await axios.post(`${API}/matches/${matchData.matchId}/set-players`, {
        player1_frame: player1Crop.base64,
        player2_frame: player2Crop.base64
      });
      toast.success("Starting analysis...");
      navigate(`/analysis/${matchData.matchId}`);
    } catch (error) {
      console.error("Error:", error);
      toast.error("Starting analysis anyway...");
      navigate(`/analysis/${matchData.matchId}`);
    }
  };

  const handleSkip = async () => {
    try {
      await axios.post(`${API}/matches/${matchData.matchId}/start-analysis`);
    } catch (e) {}
    navigate(`/analysis/${matchData.matchId}`);
  };

  if (!matchData) return null;

  const playersDone = player1Crop && player2Crop;
  const courtDone = courtCorners.length === 4;

  return (
    <div className="min-h-screen bg-[#050505]">
      <nav className="border-b border-border/50 bg-background/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
              <Target className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-heading text-xl font-bold tracking-tight">SQUASHSENSE</span>
          </Link>
          <Button variant="ghost" onClick={handleSkip} className="text-muted-foreground">
            Skip <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-8">
        <div className="text-center mb-6">
          <h1 className="font-heading text-3xl sm:text-4xl font-black tracking-tight mb-2">
            {mode === "court" ? (
              <>MARK THE <span className="text-[#30D158]">COURT</span></>
            ) : (
              <>SELECT <span className="text-primary">PLAYERS</span></>
            )}
          </h1>
          <p className="text-muted-foreground">
            {mode === "court"
              ? "Click the four floor corners so we can measure real movement (metres)"
              : "Click on each player's face or body to identify them"}
          </p>
        </div>

        {/* Step Indicator */}
        <div className="flex items-center justify-center gap-3 mb-6">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            currentStep >= 1 ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              player1Crop ? "bg-primary text-black" : "bg-primary/50 text-black"
            }`}>
              {player1Crop ? <Check className="w-4 h-4" /> : "1"}
            </span>
            <span>Player 1</span>
          </div>
          <div className="w-8 h-px bg-border" />
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            currentStep >= 2 ? "bg-[#00F0FF]/20 text-[#00F0FF]" : "bg-muted text-muted-foreground"
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              player2Crop ? "bg-[#00F0FF] text-black" : currentStep >= 2 ? "bg-[#00F0FF]/50 text-black" : "bg-muted-foreground/30"
            }`}>
              {player2Crop ? <Check className="w-4 h-4" /> : "2"}
            </span>
            <span>Player 2</span>
          </div>
          <div className="w-8 h-px bg-border" />
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            courtDone ? "bg-[#30D158]/20 text-[#30D158]" : "bg-muted text-muted-foreground"
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              courtDone ? "bg-[#30D158] text-black" : "bg-muted-foreground/30"
            }`}>
              {courtDone ? <Check className="w-4 h-4" /> : <Grid3x3 className="w-3.5 h-3.5" />}
            </span>
            <span>Court <span className="opacity-60">(optional)</span></span>
          </div>
        </div>

        {/* Video Frame */}
        <div className="bg-card border border-border rounded-lg p-4 mb-6">
          <div className="flex justify-center">
            <img
              ref={imageRef}
              src={`data:image/jpeg;base64,${matchData.thumbnail}`}
              alt="Match frame"
              className="hidden"
              onLoad={handleImageLoad}
            />
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className={`rounded-lg ${
                mode === "court" ? (courtCorners.length < 4 ? "cursor-crosshair" : "") : currentStep <= 2 ? "cursor-crosshair" : ""
              }`}
              style={{ maxWidth: "100%", maxHeight: "60vh" }}
            />
            {!frameLoaded && (
              <div className="flex items-center justify-center bg-muted rounded-lg min-h-[300px] min-w-[400px]">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
              </div>
            )}
          </div>

          <p className="text-center text-sm text-muted-foreground mt-4">
            {mode === "court" ? (
              courtCorners.length < 4 ? (
                <>👆 Click the <span className="text-[#30D158] font-semibold">{CORNER_DEFS[courtCorners.length].label}</span> corner — {CORNER_DEFS[courtCorners.length].hint}</>
              ) : (
                "✅ Court marked — confirm to start analysis"
              )
            ) : (
              <>
                {currentStep === 1 && "👆 Click on PLAYER 1 (yellow)"}
                {currentStep === 2 && "👆 Click on PLAYER 2 (cyan)"}
                {currentStep === 3 && "✅ Players selected — optionally mark the court below"}
              </>
            )}
          </p>
        </div>

        {/* Player Previews */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className={`bg-card border-2 rounded-lg p-4 text-center ${player1Crop ? "border-primary" : "border-border"}`}>
            <div className="flex items-center justify-center gap-2 mb-3">
              <span className="w-4 h-4 rounded-full bg-primary" />
              <span className="font-heading font-bold text-primary">Player 1</span>
            </div>
            {player1Crop ? (
              <img src={`data:image/jpeg;base64,${player1Crop.base64}`} alt="Player 1" className="w-32 h-32 object-cover rounded-lg border-2 border-primary mx-auto" />
            ) : (
              <div className="w-32 h-32 bg-muted rounded-lg flex items-center justify-center text-muted-foreground mx-auto text-sm">Click on frame</div>
            )}
          </div>
          <div className={`bg-card border-2 rounded-lg p-4 text-center ${player2Crop ? "border-[#00F0FF]" : "border-border"}`}>
            <div className="flex items-center justify-center gap-2 mb-3">
              <span className="w-4 h-4 rounded-full bg-[#00F0FF]" />
              <span className="font-heading font-bold text-[#00F0FF]">Player 2</span>
            </div>
            {player2Crop ? (
              <img src={`data:image/jpeg;base64,${player2Crop.base64}`} alt="Player 2" className="w-32 h-32 object-cover rounded-lg border-2 border-[#00F0FF] mx-auto" />
            ) : (
              <div className="w-32 h-32 bg-muted rounded-lg flex items-center justify-center text-muted-foreground mx-auto text-sm">Click on frame</div>
            )}
          </div>
        </div>

        {/* Court calibration controls (appear once players are chosen) */}
        {playersDone && (
          <div className="bg-card border border-border rounded-lg p-4 mb-6">
            <div className="flex items-center justify-between flex-wrap gap-3">
              <div className="flex items-center gap-2">
                <Grid3x3 className="w-5 h-5 text-[#30D158]" />
                <div>
                  <p className="font-heading font-bold text-sm">Court calibration <span className="text-muted-foreground font-normal">(optional)</span></p>
                  <p className="text-xs text-muted-foreground">
                    {courtDone ? "Court marked — real movement metrics enabled" : "Unlocks distance, speed, court coverage & T-dominance"}
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                {mode !== "court" && !courtDone && (
                  <Button variant="outline" onClick={startCourtMode} className="border-[#30D158]/50 text-[#30D158] hover:bg-[#30D158]/10">
                    <Grid3x3 className="w-4 h-4 mr-2" /> Mark court
                  </Button>
                )}
                {(mode === "court" || courtDone) && (
                  <Button variant="outline" onClick={resetCourt} className="border-border">
                    <RotateCcw className="w-4 h-4 mr-2" /> Re-mark court
                  </Button>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center justify-center gap-4">
          <Button variant="outline" onClick={handleResetPlayers} disabled={!player1Crop && !player2Crop} className="border-border">
            <RotateCcw className="w-4 h-4 mr-2" /> Reset players
          </Button>
          <Button
            onClick={handleConfirm}
            disabled={!playersDone || isSubmitting || (mode === "court" && courtCorners.length > 0 && !courtDone)}
            className="bg-primary text-primary-foreground hover:bg-primary/90 px-8"
          >
            {isSubmitting ? (
              <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Starting...</>
            ) : (
              <><Check className="w-4 h-4 mr-2" /> Confirm & Analyze</>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default PlayerSelectPage;
