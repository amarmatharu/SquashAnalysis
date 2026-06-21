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
  Loader2
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

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

  useEffect(() => {
    if (!matchData || !matchData.thumbnail) {
      toast.error("No match data found");
      navigate("/upload");
    }
  }, [matchData, navigate]);

  const redrawCanvas = useCallback((p1, p2) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    
    // Draw P1 selection
    if (p1) {
      ctx.strokeStyle = '#DFFF00';
      ctx.lineWidth = 4;
      ctx.strokeRect(p1.x, p1.y, p1.size, p1.size);
      ctx.fillStyle = 'rgba(223, 255, 0, 0.2)';
      ctx.fillRect(p1.x, p1.y, p1.size, p1.size);
      ctx.fillStyle = '#DFFF00';
      ctx.font = 'bold 18px Arial';
      ctx.fillText('P1', p1.x + 8, p1.y + 24);
    }
    
    // Draw P2 selection
    if (p2) {
      ctx.strokeStyle = '#00F0FF';
      ctx.lineWidth = 4;
      ctx.strokeRect(p2.x, p2.y, p2.size, p2.size);
      ctx.fillStyle = 'rgba(0, 240, 255, 0.2)';
      ctx.fillRect(p2.x, p2.y, p2.size, p2.size);
      ctx.fillStyle = '#00F0FF';
      ctx.font = 'bold 18px Arial';
      ctx.fillText('P2', p2.x + 8, p2.y + 24);
    }
  }, []);

  const handleCanvasClick = useCallback((e) => {
    if (currentStep > 2 || !frameLoaded) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    
    const cropSize = 120;
    const cropX = Math.max(0, Math.min(x - cropSize/2, canvas.width - cropSize));
    const cropY = Math.max(0, Math.min(y - cropSize/2, canvas.height - cropSize));
    
    // Create crop
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = cropSize;
    tempCanvas.height = cropSize;
    const tempCtx = tempCanvas.getContext('2d');
    
    tempCtx.drawImage(
      canvas,
      cropX, cropY, cropSize, cropSize,
      0, 0, cropSize, cropSize
    );
    
    const cropDataUrl = tempCanvas.toDataURL('image/jpeg', 0.9);
    const base64Data = cropDataUrl.split(',')[1];
    
    const newCrop = { x: cropX, y: cropY, size: cropSize, base64: base64Data };
    
    if (currentStep === 1) {
      setPlayer1Crop(newCrop);
      setCurrentStep(2);
      redrawCanvas(newCrop, null);
      toast.success("Player 1 selected! Now click on Player 2");
    } else if (currentStep === 2) {
      setPlayer2Crop(newCrop);
      setCurrentStep(3);
      redrawCanvas(player1Crop, newCrop);
      toast.success("Both players selected!");
    }
  }, [currentStep, frameLoaded, player1Crop, redrawCanvas]);

  const handleImageLoad = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    
    const maxWidth = Math.min(800, window.innerWidth - 48);
    const scale = maxWidth / img.naturalWidth;
    canvas.width = img.naturalWidth * scale;
    canvas.height = img.naturalHeight * scale;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    setFrameLoaded(true);
  }, []);

  const handleReset = () => {
    setPlayer1Crop(null);
    setPlayer2Crop(null);
    setCurrentStep(1);
    redrawCanvas(null, null);
    toast.info("Reset. Click on Player 1");
  };

  const handleConfirm = async () => {
    if (!player1Crop || !player2Crop) {
      toast.error("Please select both players");
      return;
    }

    setIsSubmitting(true);

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
            SELECT <span className="text-primary">PLAYERS</span>
          </h1>
          <p className="text-muted-foreground">
            Click on each player's face or body to identify them
          </p>
        </div>

        {/* Step Indicator */}
        <div className="flex items-center justify-center gap-3 mb-6">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            currentStep >= 1 ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              player1Crop ? 'bg-primary text-black' : 'bg-primary/50 text-black'
            }`}>
              {player1Crop ? <Check className="w-4 h-4" /> : '1'}
            </span>
            <span>Player 1</span>
          </div>
          <div className="w-8 h-px bg-border" />
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
            currentStep >= 2 ? 'bg-[#00F0FF]/20 text-[#00F0FF]' : 'bg-muted text-muted-foreground'
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
              player2Crop ? 'bg-[#00F0FF] text-black' : currentStep >= 2 ? 'bg-[#00F0FF]/50 text-black' : 'bg-muted-foreground/30'
            }`}>
              {player2Crop ? <Check className="w-4 h-4" /> : '2'}
            </span>
            <span>Player 2</span>
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
              className={`rounded-lg ${currentStep <= 2 ? 'cursor-crosshair' : ''}`}
              style={{ maxWidth: '100%', maxHeight: '60vh' }}
            />
            
            {!frameLoaded && (
              <div className="flex items-center justify-center bg-muted rounded-lg min-h-[300px] min-w-[400px]">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
              </div>
            )}
          </div>
          
          <p className="text-center text-sm text-muted-foreground mt-4">
            {currentStep === 1 && "👆 Click on PLAYER 1 (yellow)"}
            {currentStep === 2 && "👆 Click on PLAYER 2 (cyan)"}
            {currentStep === 3 && "✅ Confirm your selections"}
          </p>
        </div>

        {/* Player Previews */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className={`bg-card border-2 rounded-lg p-4 text-center ${player1Crop ? 'border-primary' : 'border-border'}`}>
            <div className="flex items-center justify-center gap-2 mb-3">
              <span className="w-4 h-4 rounded-full bg-primary" />
              <span className="font-heading font-bold text-primary">Player 1</span>
            </div>
            {player1Crop ? (
              <img 
                src={`data:image/jpeg;base64,${player1Crop.base64}`}
                alt="Player 1"
                className="w-32 h-32 object-cover rounded-lg border-2 border-primary mx-auto"
              />
            ) : (
              <div className="w-32 h-32 bg-muted rounded-lg flex items-center justify-center text-muted-foreground mx-auto text-sm">
                Click on frame
              </div>
            )}
          </div>
          
          <div className={`bg-card border-2 rounded-lg p-4 text-center ${player2Crop ? 'border-[#00F0FF]' : 'border-border'}`}>
            <div className="flex items-center justify-center gap-2 mb-3">
              <span className="w-4 h-4 rounded-full bg-[#00F0FF]" />
              <span className="font-heading font-bold text-[#00F0FF]">Player 2</span>
            </div>
            {player2Crop ? (
              <img 
                src={`data:image/jpeg;base64,${player2Crop.base64}`}
                alt="Player 2"
                className="w-32 h-32 object-cover rounded-lg border-2 border-[#00F0FF] mx-auto"
              />
            ) : (
              <div className="w-32 h-32 bg-muted rounded-lg flex items-center justify-center text-muted-foreground mx-auto text-sm">
                Click on frame
              </div>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center justify-center gap-4">
          <Button
            variant="outline"
            onClick={handleReset}
            disabled={!player1Crop && !player2Crop}
            className="border-border"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Reset
          </Button>
          
          <Button
            onClick={handleConfirm}
            disabled={!player1Crop || !player2Crop || isSubmitting}
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
