/**
 * Player identification — show a crop of each on-court player and let the user
 * name them (and optionally mark one as "me"). The names + colour signatures
 * are stored on the match and used to lock identity during analysis.
 */
import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { toast } from "sonner";
import { Button } from "./ui/button";
import { Loader2, Check, User } from "lucide-react";

const API = `${process.env.REACT_APP_BACKEND_URL}/api`;

export default function PlayerIdentifyModal({ matchId, existing, onClose, onSaved }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [players, setPlayers] = useState([]);   // [{slot, crop_b64, color_sig, box}]
  const [names, setNames] = useState({ 1: "", 2: "" });
  const [meSlot, setMeSlot] = useState(null);
  const [saving, setSaving] = useState(false);
  const poll = useRef(null);

  useEffect(() => {
    // Pre-fill from existing names if present
    if (existing) {
      setNames({ 1: existing.player1_name && existing.player1_name !== "Player 1" ? existing.player1_name : "",
                 2: existing.player2_name && existing.player2_name !== "Player 2" ? existing.player2_name : "" });
      if (existing.player1_is_me) setMeSlot(1);
      if (existing.player2_is_me) setMeSlot(2);
    }
    startIdentify();
    return () => { if (poll.current) clearInterval(poll.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matchId]);

  const startIdentify = async () => {
    setLoading(true); setError(null);
    try {
      await axios.post(`${API}/analysis/identify-players/${matchId}`);
      poll.current = setInterval(async () => {
        try {
          const r = await axios.get(`${API}/analysis/identify-players/${matchId}`);
          if (r.data.status === "done") {
            clearInterval(poll.current);
            setLoading(false);
            if (r.data.ok) setPlayers(r.data.players || []);
            else setError(r.data.error || "Could not identify players.");
          } else if (r.data.status === "failed") {
            clearInterval(poll.current); setLoading(false);
            setError(r.data.error || "Identification failed.");
          }
        } catch (e) { /* keep polling */ }
      }, 2500);
    } catch (e) {
      setLoading(false);
      setError(e?.response?.data?.detail || "Could not start identification.");
    }
  };

  const save = async () => {
    setSaving(true);
    try {
      const payload = {
        players: players.map((p) => ({
          slot: p.slot,
          name: names[p.slot] || `Player ${p.slot}`,
          is_me: meSlot === p.slot,
          crop_b64: p.crop_b64,
          color_sig: p.color_sig,
        })),
      };
      const r = await axios.post(`${API}/matches/${matchId}/save-players`, payload);
      toast.success("Players saved");
      onSaved && onSaved(r.data);
    } catch (e) {
      toast.error("Could not save players");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="bg-card border border-border rounded-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div>
            <h2 className="text-lg font-bold font-heading flex items-center gap-2">
              <User className="w-5 h-5 text-primary" /> Identify Players
            </h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              Name each player from their photo. Their shirt colour locks identity so they're never mixed up.
            </p>
          </div>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-xl px-2">✕</button>
        </div>

        <div className="p-5">
          {loading && (
            <div className="py-12 text-center">
              <Loader2 className="w-8 h-8 animate-spin text-primary mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">Finding a clear frame of both players…</p>
              <p className="text-[11px] text-muted-foreground mt-1">Takes ~20 seconds.</p>
            </div>
          )}

          {error && !loading && (
            <div className="py-8 text-center">
              <p className="text-sm text-red-400 mb-3">{error}</p>
              <Button variant="outline" onClick={startIdentify}>Try again</Button>
            </div>
          )}

          {!loading && !error && players.length >= 2 && (
            <>
              <div className="grid grid-cols-2 gap-4 mb-5">
                {players.map((p) => (
                  <div key={p.slot} className={`rounded-lg border p-3 transition-all ${
                    meSlot === p.slot ? "border-primary ring-1 ring-primary/40" : "border-border"}`}>
                    <div className="aspect-[2/3] bg-black rounded-md overflow-hidden mb-3 flex items-center justify-center">
                      {p.crop_b64
                        ? <img src={`data:image/jpeg;base64,${p.crop_b64}`} alt={`Player ${p.slot}`}
                            className="h-full object-contain" />
                        : <span className="text-xs text-muted-foreground">no image</span>}
                    </div>
                    <label className="text-[11px] text-muted-foreground">Name</label>
                    <input
                      value={names[p.slot]}
                      onChange={(e) => setNames((n) => ({ ...n, [p.slot]: e.target.value }))}
                      placeholder={`Player ${p.slot}`}
                      className="w-full mt-1 bg-background border border-border rounded px-2 py-1.5 text-sm" />
                    <button
                      onClick={() => setMeSlot(meSlot === p.slot ? null : p.slot)}
                      className={`mt-2 w-full text-xs rounded px-2 py-1.5 border transition-all ${
                        meSlot === p.slot
                          ? "bg-primary text-primary-foreground border-primary"
                          : "bg-background text-muted-foreground border-border hover:bg-muted"}`}>
                      {meSlot === p.slot ? "✓ This is me" : "This is me"}
                    </button>
                  </div>
                ))}
              </div>

              <div className="flex items-center justify-between">
                <button onClick={startIdentify} className="text-xs text-muted-foreground hover:text-foreground">
                  ↻ Find a different frame
                </button>
                <div className="flex gap-3">
                  <Button variant="outline" onClick={onClose}>Cancel</Button>
                  <Button onClick={save} disabled={saving}
                    className="bg-primary text-primary-foreground hover:bg-primary/90">
                    {saving ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Saving…</>
                      : <><Check className="w-4 h-4 mr-2" /> Save Players</>}
                  </Button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
