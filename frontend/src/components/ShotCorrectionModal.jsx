import { useState } from "react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { toast } from "sonner";
import axios from "axios";
import { Check, X, Edit2, Brain } from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const SHOT_TYPES = [
  { value: "drive", label: "Drive", color: "#DFFF00" },
  { value: "drop", label: "Drop", color: "#00F0FF" },
  { value: "boast", label: "Boast", color: "#FF3B30" },
  { value: "volley", label: "Volley", color: "#30D158" },
  { value: "lob", label: "Lob", color: "#FF9F0A" },
  { value: "kill", label: "Kill", color: "#BF5AF2" },
  { value: "serve", label: "Serve", color: "#64D2FF" },
];

const ShotCorrectionModal = ({ 
  isOpen, 
  onClose, 
  shot, 
  shotIndex, 
  matchId, 
  onCorrectionSaved 
}) => {
  const [shotType, setShotType] = useState(shot?.shot_type || "drive");
  const [player, setPlayer] = useState(shot?.player || "player1");
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    if (!shot) return;
    
    setSaving(true);
    try {
      await axios.post(`${API}/matches/${matchId}/correct-shot`, {
        shot_index: shotIndex,
        corrected_shot_type: shotType,
        corrected_player: player,
      });
      
      toast.success("Correction saved! This helps train the AI.");
      onCorrectionSaved(shotIndex, shotType, player);
      onClose();
    } catch (error) {
      console.error("Correction error:", error);
      toast.error("Failed to save correction");
    } finally {
      setSaving(false);
    }
  };

  if (!shot) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-card border-border max-w-md">
        <DialogHeader>
          <DialogTitle className="font-heading flex items-center gap-2">
            <Edit2 className="w-5 h-5 text-primary" />
            Correct Shot #{shotIndex + 1}
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-6 pt-4">
          {/* Current Detection */}
          <div className="bg-muted/30 rounded-lg p-4">
            <p className="text-sm text-muted-foreground mb-2">AI Detected:</p>
            <div className="flex items-center gap-4">
              <span className="px-3 py-1 rounded-full bg-primary/20 text-primary font-mono text-sm">
                {shot.shot_type}
              </span>
              <span className="text-muted-foreground">by</span>
              <span className="px-3 py-1 rounded-full bg-[#00F0FF]/20 text-[#00F0FF] font-mono text-sm">
                {shot.player}
              </span>
            </div>
            {shot.confidence && (
              <p className="text-xs text-muted-foreground mt-2">
                Confidence: {(shot.confidence * 100).toFixed(0)}%
              </p>
            )}
          </div>

          {/* Correction Form */}
          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Correct Shot Type</label>
              <Select value={shotType} onValueChange={setShotType}>
                <SelectTrigger className="bg-background border-border">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-card border-border">
                  {SHOT_TYPES.map((type) => (
                    <SelectItem key={type.value} value={type.value}>
                      <div className="flex items-center gap-2">
                        <span 
                          className="w-3 h-3 rounded-full" 
                          style={{ background: type.color }}
                        />
                        {type.label}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Correct Player</label>
              <Select value={player} onValueChange={setPlayer}>
                <SelectTrigger className="bg-background border-border">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-card border-border">
                  <SelectItem value="player1">
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-primary" />
                      Player 1
                    </div>
                  </SelectItem>
                  <SelectItem value="player2">
                    <div className="flex items-center gap-2">
                      <span className="w-3 h-3 rounded-full bg-[#00F0FF]" />
                      Player 2
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* Info */}
          <div className="flex items-start gap-2 text-xs text-muted-foreground bg-primary/5 rounded-lg p-3">
            <Brain className="w-4 h-4 mt-0.5 text-primary flex-shrink-0" />
            <p>
              Your corrections help train the AI to be more accurate. 
              The more corrections submitted, the smarter the system becomes!
            </p>
          </div>

          {/* Actions */}
          <div className="flex justify-end gap-3">
            <Button variant="outline" onClick={onClose} className="border-border">
              <X className="w-4 h-4 mr-2" />
              Cancel
            </Button>
            <Button 
              onClick={handleSave} 
              disabled={saving}
              className="bg-primary text-primary-foreground"
            >
              <Check className="w-4 h-4 mr-2" />
              {saving ? "Saving..." : "Save Correction"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ShotCorrectionModal;
