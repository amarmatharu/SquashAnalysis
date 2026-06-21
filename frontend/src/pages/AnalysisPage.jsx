import { useState, useEffect } from "react";
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
  FileText
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "../components/ui/dropdown-menu";

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

  useEffect(() => {
    fetchMatch();
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
          <TabsList className="bg-card border border-border p-1">
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
                <h3 className="font-heading text-xl font-bold mb-4">Shot Breakdown</h3>
                <div className="space-y-3">
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
              </div>
            </div>
          </TabsContent>

          {/* Rally Breakdown Tab */}
          <TabsContent value="rallies" className="space-y-6">
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
          </TabsContent>

          {/* Insights Tab */}
          <TabsContent value="insights" className="space-y-6">
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
        </Tabs>
      </div>
    </div>
  );
};

export default AnalysisPage;
