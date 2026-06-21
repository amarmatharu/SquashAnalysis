import { useState, useRef, useEffect } from "react";
import { useNavigate, useLocation, Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { toast } from "sonner";
import axios from "axios";
import { 
  Target, 
  ArrowLeft,
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
  const [currentStep, setCurrentStep] = useState(1); // 1 = select P1, 2 = select P2, 3 = confirm
  const [player1Crop, setPlayer1Crop] = useState(null);
  const [player2Crop, setPlayer2Crop] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [frameLoaded, setFrameLoaded] = useState(false);

  useEffect(() => {
    if (!matchData) {
      toast.error("No match data found");
      navigate("/upload");
    }
  }, [matchData, navigate]);

  const handleCanvasClick = (e) => {
    if (currentStep > 2) return;
    
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Calculate crop area (100x100 around click point)
    const cropSize = 80;
    const cropX = Math.max(0, Math.min(x - cropSize/2, canvas.width - cropSize));
    const cropY = Math.max(0, Math.min(y - cropSize/2, canvas.height - cropSize));
    
    // Create crop from the displayed image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = cropSize;
    tempCanvas.height = cropSize;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Scale coordinates back to original image size
    const scaleX = imageRef.current.naturalWidth / canvas.width;
    const scaleY = imageRef.current.naturalHeight / canvas.height;
    
    tempCtx.drawImage(
      imageRef.current,
      cropX * scaleX, cropY * scaleY, cropSize * scaleX, cropSize * scaleY,
      0, 0, cropSize, cropSize
    );
    
    const cropDataUrl = tempCanvas.toDataURL('image/jpeg', 0.8);
    const base64Data = cropDataUrl.split(',')[1];
    
    if (currentStep === 1) {
      setPlayer1Crop({ x: cropX, y: cropY, size: cropSize, base64: base64Data });
      setCurrentStep(2);
      toast.success("Player 1 selected! Now click on Player 2");
    } else if (currentStep === 2) {
      setPlayer2Crop({ x: cropX, y: cropY, size: cropSize, base64: base64Data });
      setCurrentStep(3);
      toast.success("Player 2 selected! Review and confirm");
    }
    
    // Redraw canvas with selections
    drawSelections(cropX, cropY, cropSize, currentStep);
  };

  const drawSelections = (newX, newY, size, step) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Redraw image
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
    
    // Draw Player 1 selection
    if (player1Crop || step === 1) {
      const p1 = step === 1 ? { x: newX, y: newY, size } : player1Crop;
      ctx.strokeStyle = '#DFFF00';
      ctx.lineWidth = 3;
      ctx.strokeRect(p1.x, p1.y, p1.size, p1.size);
      ctx.fillStyle = '#DFFF00';
      ctx.font = 'bold 14px Inter';
      ctx.fillText('P1', p1.x + 5, p1.y + 18);
    }
    
    // Draw Player 2 selection
    if (player2Crop || step === 2) {
      const p2 = step === 2 ? { x: newX, y: newY, size } : player2Crop;
      ctx.strokeStyle = '#00F0FF';
      ctx.lineWidth = 3;
      ctx.strokeRect(p2.x, p2.y, p2.size, p2.size);
      ctx.fillStyle = '#00F0FF';
      ctx.font = 'bold 14px Inter';
      ctx.fillText('P2', p2.x + 5, p2.y + 18);
    }
  };

  const handleImageLoad = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    
    // Set canvas size to match displayed image
    const maxWidth = 800;
    const scale = Math.min(maxWidth / img.naturalWidth, 1);
    canvas.width = img.naturalWidth * scale;
    canvas.height = img.naturalHeight * scale;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    setFrameLoaded(true);
  };

  const handleReset = () => {
    setPlayer1Crop(null);
    setPlayer2Crop(null);
    setCurrentStep(1);
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);
    
    toast.info("Selections reset. Click on Player 1");
  };

  const handleConfirm = async () => {
    if (!player1Crop || !player2Crop) {
      toast.error("Please select both players");
      return;
    }

    setIsSubmitting(true);

    try {
      // Update match with player reference frames
      await axios.post(`${API}/matches/${matchData.matchId}/set-players`, {
        player1_frame: player1Crop.base64,
        player2_frame: player2Crop.base64
      });

      toast.success("Players identified! Starting analysis...");
      navigate(`/analysis/${matchData.matchId}`);
    } catch (error) {
      console.error("Error setting players:", error);
      toast.error("Failed to save player selection");
      setIsSubmitting(false);
    }
  };

  const handleSkip = () => {
    navigate(`/analysis/${matchData.matchId}`);
  };

  if (!matchData) {
    return null;
  }

  return (
    <div className="min-h-screen bg-[#050505]">
      {/* Navigation */}
      <nav className="border-b border-border/50 bg-background/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
              <Target className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-heading text-xl font-bold tracking-tight">SQUASHSENSE</span>
          </Link>
          
          <Button variant="ghost" onClick={handleSkip} className="text-muted-foreground">
            Skip this step
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-6 py-12">
        <div className="text-center mb-8">
          <h1 className="font-heading text-4xl font-black tracking-tight mb-4">
            IDENTIFY <span className="text-primary">PLAYERS</span>
          </h1>
          <p className="text-muted-foreground">
            Click on each player in the frame so the AI can track them correctly
          </p>
        </div>

        {/* Step Indicator */}
        <div className="flex items-center justify-center gap-4 mb-8">
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${
            currentStep >= 1 ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'
          }`}>
            <span className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-bold">
              {player1Crop ? <Check className="w-4 h-4" /> : '1'}
            </span>
            <span className="font-medium">Select Player 1</span>
          </div>
          <div className="w-8 h-px bg-border" />
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${
            currentStep >= 2 ? 'bg-[#00F0FF]/20 text-[#00F0FF]' : 'bg-muted text-muted-foreground'
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold ${
              currentStep >= 2 ? 'bg-[#00F0FF] text-black' : 'bg-muted-foreground/30'
            }`}>
              {player2Crop ? <Check className="w-4 h-4" /> : '2'}
            </span>
            <span className="font-medium">Select Player 2</span>
          </div>
          <div className="w-8 h-px bg-border" />
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full ${
            currentStep >= 3 ? 'bg-green-500/20 text-green-400' : 'bg-muted text-muted-foreground'
          }`}>
            <span className={`w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold ${
              currentStep >= 3 ? 'bg-green-500 text-black' : 'bg-muted-foreground/30'
            }`}>
              3
            </span>
            <span className="font-medium">Confirm</span>
          </div>
        </div>

        {/* Video Frame */}
        <div className="bg-card border border-border rounded-lg p-4 mb-6">
          <div className="relative inline-block w-full">
            {/* Hidden image for loading */}
            <img
              ref={imageRef}
              src={`data:image/jpeg;base64,${matchData.thumbnail}`}
              alt="Match frame"
              className="hidden"
              onLoad={handleImageLoad}
            />
            
            {/* Interactive canvas */}
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className={`w-full rounded-lg ${currentStep <= 2 ? 'cursor-crosshair' : 'cursor-default'}`}
              data-testid="player-select-canvas"
            />
            
            {!frameLoaded && (
              <div className="absolute inset-0 flex items-center justify-center bg-muted rounded-lg">
                <Loader2 className="w-8 h-8 animate-spin text-primary" />
              </div>
            )}
          </div>
          
          <p className="text-center text-sm text-muted-foreground mt-4">
            {currentStep === 1 && "Click on Player 1 (will be shown in green/yellow)"}
            {currentStep === 2 && "Now click on Player 2 (will be shown in cyan)"}
            {currentStep === 3 && "Review your selections and confirm"}
          </p>
        </div>

        {/* Player Previews */}
        {(player1Crop || player2Crop) && (
          <div className="grid grid-cols-2 gap-6 mb-8">
            <div className={`bg-card border rounded-lg p-4 ${player1Crop ? 'border-primary' : 'border-border'}`}>
              <h3 className="font-heading font-bold text-primary mb-3 flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-primary" />
                {matchData.player1Name || "Player 1"}
              </h3>
              {player1Crop ? (
                <img 
                  src={`data:image/jpeg;base64,${player1Crop.base64}`}
                  alt="Player 1"
                  className="w-24 h-24 object-cover rounded border border-primary/50"
                />
              ) : (
                <div className="w-24 h-24 bg-muted rounded flex items-center justify-center text-muted-foreground">
                  Click to select
                </div>
              )}
            </div>
            
            <div className={`bg-card border rounded-lg p-4 ${player2Crop ? 'border-[#00F0FF]' : 'border-border'}`}>
              <h3 className="font-heading font-bold text-[#00F0FF] mb-3 flex items-center gap-2">
                <span className="w-3 h-3 rounded-full bg-[#00F0FF]" />
                {matchData.player2Name || "Player 2"}
              </h3>
              {player2Crop ? (
                <img 
                  src={`data:image/jpeg;base64,${player2Crop.base64}`}
                  alt="Player 2"
                  className="w-24 h-24 object-cover rounded border border-[#00F0FF]/50"
                />
              ) : (
                <div className="w-24 h-24 bg-muted rounded flex items-center justify-center text-muted-foreground">
                  Click to select
                </div>
              )}
            </div>
          </div>
        )}

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
            data-testid="confirm-players-btn"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Starting Analysis...
              </>
            ) : (
              <>
                <Check className="w-4 h-4 mr-2" />
                Confirm & Analyze
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
};

export default PlayerSelectPage;
