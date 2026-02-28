import { Link } from "react-router-dom";
import { Button } from "../components/ui/button";
import { 
  Activity, 
  BarChart3, 
  Target, 
  Zap, 
  Play,
  ArrowRight,
  Check
} from "lucide-react";

const LandingPage = () => {
  const features = [
    {
      icon: <Target className="w-6 h-6" />,
      title: "Shot Categorization",
      description: "AI identifies drives, drops, boasts, volleys, lobs, and kills with precision"
    },
    {
      icon: <Activity className="w-6 h-6" />,
      title: "Rally Analysis",
      description: "Track rally lengths, patterns, and winning shot combinations"
    },
    {
      icon: <BarChart3 className="w-6 h-6" />,
      title: "Movement Tracking",
      description: "Visualize court coverage and player positioning heatmaps"
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: "Swing Mechanics",
      description: "Analyze forehand/backhand ratios and technique quality"
    }
  ];

  const stats = [
    { value: "7+", label: "Shot Types Tracked" },
    { value: "AI", label: "Powered Analysis" },
    { value: "PDF", label: "Export Reports" },
    { value: "100%", label: "Cloud Based" }
  ];

  return (
    <div className="min-h-screen bg-[#050505]">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-primary rounded flex items-center justify-center">
              <Target className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="font-heading text-xl font-bold tracking-tight">SQUASHSENSE</span>
          </Link>
          
          <div className="hidden md:flex items-center gap-8">
            <Link to="/history" className="nav-link text-muted-foreground hover:text-foreground">
              History
            </Link>
            <a href="#features" className="nav-link text-muted-foreground hover:text-foreground">
              Features
            </a>
          </div>
          
          <Link to="/upload">
            <Button 
              data-testid="nav-upload-btn"
              className="bg-primary text-primary-foreground hover:bg-primary/90 font-heading font-semibold tracking-wide"
            >
              Analyze Match
            </Button>
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-bg pt-32 pb-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-border bg-card/50">
                <span className="w-2 h-2 bg-primary rounded-full animate-pulse-glow"></span>
                <span className="text-sm text-muted-foreground">World's First AI Squash Analyzer</span>
              </div>
              
              <h1 className="font-heading text-5xl sm:text-6xl lg:text-7xl font-black tracking-tighter leading-[0.9]">
                DECODE YOUR
                <br />
                <span className="text-primary">SQUASH GAME</span>
              </h1>
              
              <p className="text-lg text-muted-foreground max-w-lg">
                Upload any squash match video and let AI analyze shots, rallies, movement patterns, 
                and swing mechanics. Get actionable insights to elevate your game.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4">
                <Link to="/upload">
                  <Button 
                    data-testid="hero-upload-btn"
                    size="lg" 
                    className="bg-primary text-primary-foreground hover:bg-primary/90 font-heading font-bold text-lg px-8 glow-hover"
                  >
                    <Play className="w-5 h-5 mr-2" />
                    Start Analysis
                  </Button>
                </Link>
                <Link to="/history">
                  <Button 
                    data-testid="hero-history-btn"
                    size="lg" 
                    variant="outline" 
                    className="border-border hover:border-primary/50 font-heading font-semibold"
                  >
                    View History
                    <ArrowRight className="w-5 h-5 ml-2" />
                  </Button>
                </Link>
              </div>
            </div>
            
            {/* Hero Visual */}
            <div className="relative">
              <div className="aspect-video rounded-lg overflow-hidden border border-border bg-card glass">
                <div className="absolute inset-0 grid-texture opacity-30"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center space-y-4">
                    <div className="w-20 h-20 mx-auto rounded-full bg-primary/20 border-2 border-primary flex items-center justify-center animate-pulse-glow">
                      <Play className="w-8 h-8 text-primary" />
                    </div>
                    <p className="text-muted-foreground">Upload your match video</p>
                  </div>
                </div>
              </div>
              
              {/* Floating stat cards */}
              <div className="absolute -bottom-6 -left-6 bg-card border border-border p-4 rounded-lg glass animate-slide-up" style={{animationDelay: '0.2s'}}>
                <div className="font-mono text-2xl font-bold text-primary">156</div>
                <div className="text-xs text-muted-foreground">Shots Analyzed</div>
              </div>
              
              <div className="absolute -top-6 -right-6 bg-card border border-border p-4 rounded-lg glass animate-slide-up" style={{animationDelay: '0.4s'}}>
                <div className="font-mono text-2xl font-bold text-[#00F0FF]">24</div>
                <div className="text-xs text-muted-foreground">Rallies Tracked</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="border-y border-border bg-card/30">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="font-heading text-4xl font-black text-primary">{stat.value}</div>
                <div className="text-sm text-muted-foreground mt-1">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="font-heading text-4xl sm:text-5xl font-black tracking-tight mb-4">
              POWERFUL <span className="text-primary">ANALYTICS</span>
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Our AI-powered analysis engine breaks down every aspect of your squash match
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className="stat-card rounded-lg card-shine animate-slide-up"
                style={{animationDelay: `${index * 0.1}s`}}
              >
                <div className="w-12 h-12 rounded-lg bg-primary/10 border border-primary/30 flex items-center justify-center text-primary mb-4">
                  {feature.icon}
                </div>
                <h3 className="font-heading text-xl font-bold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 px-6 bg-card/30 border-y border-border">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="font-heading text-4xl sm:text-5xl font-black tracking-tight mb-4">
              HOW IT <span className="text-primary">WORKS</span>
            </h2>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: "01", title: "Upload Video", desc: "Upload your squash match video in MP4, MOV, or WebM format" },
              { step: "02", title: "AI Analysis", desc: "Our GPT-5.2 powered AI analyzes frames for shots, movement, and patterns" },
              { step: "03", title: "Get Insights", desc: "Review detailed analytics and export reports in PDF or JSON" }
            ].map((item, index) => (
              <div key={index} className="relative">
                <div className="font-heading text-8xl font-black text-primary/10 absolute -top-8 -left-4">
                  {item.step}
                </div>
                <div className="relative pt-12 pl-4">
                  <h3 className="font-heading text-2xl font-bold mb-2">{item.title}</h3>
                  <p className="text-muted-foreground">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="font-heading text-4xl sm:text-5xl font-black tracking-tight mb-6">
            READY TO <span className="text-primary">ELEVATE</span> YOUR GAME?
          </h2>
          <p className="text-lg text-muted-foreground mb-8">
            Start analyzing your squash matches with AI-powered insights today.
          </p>
          <Link to="/upload">
            <Button 
              data-testid="cta-upload-btn"
              size="lg" 
              className="bg-primary text-primary-foreground hover:bg-primary/90 font-heading font-bold text-xl px-12 py-6 glow-hover"
            >
              Upload Your First Match
              <ArrowRight className="w-6 h-6 ml-2" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-8 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-primary rounded flex items-center justify-center">
              <Target className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="font-heading font-bold">SQUASHSENSE AI</span>
          </div>
          <p className="text-sm text-muted-foreground">
            World's First AI-Powered Squash Analysis Platform
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
