import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Progress } from "../components/ui/progress";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "../components/ui/alert-dialog";
import axios from "axios";
import { toast } from "sonner";
import { 
  Target, 
  Upload, 
  ArrowLeft,
  Play,
  Loader2,
  Trash2,
  Download,
  MoreVertical,
  FileJson,
  FileText
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const HistoryPage = () => {
  const navigate = useNavigate();
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deleteId, setDeleteId] = useState(null);

  useEffect(() => {
    fetchMatches();
  }, []);

  const fetchMatches = async () => {
    try {
      const response = await axios.get(`${API}/matches`);
      setMatches(response.data);
    } catch (error) {
      console.error("Failed to fetch matches:", error);
      toast.error("Failed to load match history");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (matchId) => {
    try {
      await axios.delete(`${API}/matches/${matchId}`);
      setMatches(matches.filter(m => m.id !== matchId));
      toast.success("Match deleted successfully");
    } catch (error) {
      console.error("Delete error:", error);
      toast.error("Failed to delete match");
    }
    setDeleteId(null);
  };

  const handleExportJSON = async (matchId) => {
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
      toast.success("JSON exported successfully");
    } catch (error) {
      toast.error("Export failed");
    }
  };

  const handleExportPDF = async (matchId) => {
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
      toast.success("PDF exported successfully");
    } catch (error) {
      toast.error("Export failed");
    }
  };

  const getStatusBadge = (status) => {
    const styles = {
      completed: "badge-success",
      processing: "badge-warning",
      pending: "badge-warning",
      failed: "badge-error"
    };
    return (
      <span className={`px-2 py-1 rounded text-xs font-mono ${styles[status] || styles.pending}`}>
        {status.toUpperCase()}
      </span>
    );
  };

  const formatDuration = (seconds) => {
    if (!seconds) return "--:--";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

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
          
          <div className="flex items-center gap-4">
            <Link to="/" className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
              <ArrowLeft className="w-4 h-4" />
              <span>Back</span>
            </Link>
            <Link to="/upload">
              <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
                <Upload className="w-4 h-4 mr-2" />
                New Analysis
              </Button>
            </Link>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="mb-8">
          <h1 className="font-heading text-4xl font-black tracking-tight mb-2">
            MATCH <span className="text-primary">HISTORY</span>
          </h1>
          <p className="text-muted-foreground">
            All your analyzed squash matches
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-24">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : matches.length === 0 ? (
          <div className="text-center py-24 bg-card border border-border rounded-lg">
            <Play className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
            <h2 className="font-heading text-2xl font-bold mb-2">No matches yet</h2>
            <p className="text-muted-foreground mb-6">Upload your first squash match to get started</p>
            <Link to="/upload">
              <Button className="bg-primary text-primary-foreground">
                <Upload className="w-4 h-4 mr-2" />
                Upload Match
              </Button>
            </Link>
          </div>
        ) : (
          <div className="grid gap-4">
            {matches.map((match) => (
              <div 
                key={match.id}
                className="stat-card rounded-lg"
                data-testid={`history-match-${match.id}`}
              >
                <div className="flex items-center gap-6">
                  {/* Thumbnail */}
                  <div 
                    onClick={() => navigate(`/analysis/${match.id}`)}
                    className="cursor-pointer flex-shrink-0"
                  >
                    {match.thumbnail ? (
                      <img 
                        src={`data:image/jpeg;base64,${match.thumbnail}`}
                        alt={match.title}
                        className="w-32 h-20 object-cover rounded bg-muted"
                      />
                    ) : (
                      <div className="w-32 h-20 bg-muted rounded flex items-center justify-center">
                        <Play className="w-8 h-8 text-muted-foreground" />
                      </div>
                    )}
                  </div>
                  
                  {/* Info */}
                  <div 
                    className="flex-1 cursor-pointer"
                    onClick={() => navigate(`/analysis/${match.id}`)}
                  >
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-heading text-xl font-bold">{match.title}</h3>
                      {getStatusBadge(match.status)}
                    </div>
                    <div className="flex items-center gap-6 text-sm text-muted-foreground">
                      <span>{new Date(match.upload_time).toLocaleDateString()}</span>
                      <span>{formatDuration(match.duration)}</span>
                      <span>{match.total_shots} shots</span>
                      <span>{match.total_rallies} rallies</span>
                    </div>
                    {match.status === "processing" && (
                      <div className="mt-2">
                        <Progress value={match.progress} className="h-1 w-48" />
                        <span className="text-xs text-muted-foreground">Analyzing... {match.progress}%</span>
                      </div>
                    )}
                  </div>
                  
                  {/* Actions */}
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon" data-testid={`match-actions-${match.id}`}>
                        <MoreVertical className="w-5 h-5" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="bg-card border-border">
                      <DropdownMenuItem 
                        onClick={() => handleExportJSON(match.id)}
                        className="cursor-pointer"
                        disabled={match.status !== "completed"}
                      >
                        <FileJson className="w-4 h-4 mr-2" />
                        Export JSON
                      </DropdownMenuItem>
                      <DropdownMenuItem 
                        onClick={() => handleExportPDF(match.id)}
                        className="cursor-pointer"
                        disabled={match.status !== "completed"}
                      >
                        <FileText className="w-4 h-4 mr-2" />
                        Export PDF
                      </DropdownMenuItem>
                      <DropdownMenuItem 
                        onClick={() => setDeleteId(match.id)}
                        className="cursor-pointer text-destructive focus:text-destructive"
                      >
                        <Trash2 className="w-4 h-4 mr-2" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!deleteId} onOpenChange={() => setDeleteId(null)}>
        <AlertDialogContent className="bg-card border-border">
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Match Analysis</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete this match analysis? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="border-border">Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={() => handleDelete(deleteId)}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default HistoryPage;
