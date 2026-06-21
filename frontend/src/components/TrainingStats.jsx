import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import axios from "axios";
import { Brain, Database, TrendingUp, Download, CheckCircle } from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const TrainingStats = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API}/training/stats`);
      setStats(response.data);
    } catch (error) {
      console.error("Failed to fetch training stats:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    try {
      const response = await axios.get(`${API}/training/export`);
      const blob = new Blob([JSON.stringify(response.data, null, 2)], { type: 'application/json' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `squashsense_training_data.json`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error("Export failed:", error);
    }
  };

  if (loading) {
    return (
      <div className="bg-card border border-border rounded-lg p-6 animate-pulse">
        <div className="h-6 bg-muted rounded w-1/3 mb-4"></div>
        <div className="h-4 bg-muted rounded w-2/3"></div>
      </div>
    );
  }

  if (!stats) return null;

  const progressToTraining = Math.min(100, (stats.total_corrections / 100) * 100);

  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
            <Brain className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h3 className="font-heading text-lg font-bold">AI Training Progress</h3>
            <p className="text-sm text-muted-foreground">Help improve the model</p>
          </div>
        </div>
        
        {stats.training_ready && (
          <Button variant="outline" size="sm" onClick={handleExport} className="border-border">
            <Download className="w-4 h-4 mr-2" />
            Export Data
          </Button>
        )}
      </div>

      {/* Progress to training */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-muted-foreground">Corrections until training ready</span>
          <span className="text-sm font-mono">{stats.total_corrections} / 100</span>
        </div>
        <Progress value={progressToTraining} className="h-2" />
        {stats.training_ready ? (
          <p className="text-xs text-green-400 mt-2 flex items-center gap-1">
            <CheckCircle className="w-3 h-3" />
            Ready for model fine-tuning!
          </p>
        ) : (
          <p className="text-xs text-muted-foreground mt-2">
            Need {100 - stats.total_corrections} more corrections
          </p>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <Database className="w-5 h-5 mx-auto mb-2 text-primary" />
          <div className="font-mono text-2xl font-bold">{stats.total_corrections}</div>
          <div className="text-xs text-muted-foreground">Corrections</div>
        </div>
        
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <TrendingUp className="w-5 h-5 mx-auto mb-2 text-[#00F0FF]" />
          <div className="font-mono text-2xl font-bold">{stats.model_accuracy_estimate}%</div>
          <div className="text-xs text-muted-foreground">Est. Accuracy</div>
        </div>
        
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <CheckCircle className="w-5 h-5 mx-auto mb-2 text-green-400" />
          <div className="font-mono text-2xl font-bold">{stats.total_shots_analyzed}</div>
          <div className="text-xs text-muted-foreground">Shots Analyzed</div>
        </div>
      </div>

      {/* Corrections by type */}
      {Object.keys(stats.corrections_by_shot_type || {}).length > 0 && (
        <div className="mt-6">
          <p className="text-sm text-muted-foreground mb-3">Corrections by shot type:</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(stats.corrections_by_shot_type).map(([type, count]) => (
              <span 
                key={type}
                className="px-2 py-1 rounded text-xs font-mono bg-muted/50"
              >
                {type}: {count}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingStats;
