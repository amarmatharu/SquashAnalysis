import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "../components/ui/button";
import { Progress } from "../components/ui/progress";
import axios from "axios";
import { toast } from "sonner";
import { 
  Target, 
  Upload, 
  History,
  ArrowRight,
  Play,
  Loader2
} from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

const DashboardPage = () => {
  const navigate = useNavigate();
  const [recentMatches, setRecentMatches] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchRecentMatches();
  }, []);

  const fetchRecentMatches = async () => {
    try {
      const response = await axios.get(`${API}/matches`);
      setRecentMatches(response.data.slice(0, 3));
    } catch (error) {
      console.error("Failed to fetch matches:", error);
    } finally {
      setLoading(false);
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
            <Link to="/history">
              <Button variant="ghost" className="text-muted-foreground hover:text-foreground">
                <History className="w-4 h-4 mr-2" />
                History
              </Button>
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
        {/* Welcome Section */}
        <div className="mb-12">
          <h1 className="font-heading text-4xl font-black tracking-tight mb-2">
            WELCOME TO <span className="text-primary">SQUASHSENSE</span>
          </h1>
          <p className="text-muted-foreground">
            Your AI-powered squash match analysis dashboard
          </p>
        </div>

        {/* Quick Actions */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          <div 
            onClick={() => navigate('/upload')}
            className="stat-card rounded-lg cursor-pointer group"
            data-testid="quick-upload-card"
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-heading text-2xl font-bold mb-2">New Analysis</h3>
                <p className="text-muted-foreground">Upload a match video for AI analysis</p>
              </div>
              <div className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center group-hover:bg-primary/30 transition-colors">
                <Upload className="w-8 h-8 text-primary" />
              </div>
            </div>
          </div>

          <div 
            onClick={() => navigate('/history')}
            className="stat-card rounded-lg cursor-pointer group"
            data-testid="quick-history-card"
          >
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-heading text-2xl font-bold mb-2">Match History</h3>
                <p className="text-muted-foreground">View all your analyzed matches</p>
              </div>
              <div className="w-16 h-16 rounded-full bg-[#00F0FF]/20 flex items-center justify-center group-hover:bg-[#00F0FF]/30 transition-colors">
                <History className="w-8 h-8 text-[#00F0FF]" />
              </div>
            </div>
          </div>
        </div>

        {/* Recent Matches */}
        <div>
          <div className="flex items-center justify-between mb-6">
            <h2 className="font-heading text-2xl font-bold">Recent Analyses</h2>
            <Link to="/history" className="text-primary hover:underline text-sm flex items-center gap-1">
              View all <ArrowRight className="w-4 h-4" />
            </Link>
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
            </div>
          ) : recentMatches.length === 0 ? (
            <div className="text-center py-12 bg-card border border-border rounded-lg">
              <Play className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
              <p className="text-muted-foreground">No matches analyzed yet</p>
              <Link to="/upload">
                <Button className="mt-4 bg-primary text-primary-foreground">
                  Upload Your First Match
                </Button>
              </Link>
            </div>
          ) : (
            <div className="space-y-4">
              {recentMatches.map((match) => (
                <div 
                  key={match.id}
                  onClick={() => navigate(`/analysis/${match.id}`)}
                  className="stat-card rounded-lg cursor-pointer"
                  data-testid={`match-card-${match.id}`}
                >
                  <div className="flex items-center gap-4">
                    {match.thumbnail ? (
                      <img 
                        src={`data:image/jpeg;base64,${match.thumbnail}`}
                        alt={match.title}
                        className="w-24 h-16 object-cover rounded bg-muted"
                      />
                    ) : (
                      <div className="w-24 h-16 bg-muted rounded flex items-center justify-center">
                        <Play className="w-6 h-6 text-muted-foreground" />
                      </div>
                    )}
                    
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-heading font-bold">{match.title}</h3>
                        {getStatusBadge(match.status)}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {new Date(match.upload_time).toLocaleDateString()} • 
                        {match.total_shots} shots • {match.total_rallies} rallies
                      </p>
                      {match.status === "processing" && (
                        <Progress value={match.progress} className="h-1 mt-2 w-48" />
                      )}
                    </div>
                    
                    <ArrowRight className="w-5 h-5 text-muted-foreground" />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
