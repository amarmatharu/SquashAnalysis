"""
Audio-based rally segmentation.

Squash gives the clearest possible rally signal in its SOUND: every shot is a sharp
"thwack" (ball on wall/racket). A rally is a rhythm of thwacks; between rallies it
is quiet (or the ball is being bounced before a serve). This is far more robust
than visual ball tracking — it doesn't care about blur, occlusion, or calibration.

Pipeline:
  1. decode the audio (PyAV) → mono float at a working sample rate
  2. onset strength via high-frequency-weighted spectral flux (ball strikes are
     sharp, broadband/high-frequency transients)
  3. peak-pick → strike times
  4. cluster strikes into rallies (runs with small gaps; big gaps = between points)

Output matches the rally-segments schema (start_t/end_t/shots) so it drops into
the existing rally UI + the rules-engine scoreboard.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def extract_audio(video_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Decode the full audio track to a mono float32 array at target_sr."""
    import av

    container = av.open(video_path)
    if not container.streams.audio:
        container.close()
        return np.zeros(0, np.float32), target_sr
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=target_sr)
    chunks: List[np.ndarray] = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            arr = rf.to_ndarray().reshape(-1).astype(np.float32)
            chunks.append(arr)
    container.close()
    if not chunks:
        return np.zeros(0, np.float32), target_sr
    audio = np.concatenate(chunks) / 32768.0
    return audio, target_sr


def onset_strength(audio: np.ndarray, sr: int, hop: int = 256, win: int = 1024,
                   hf_cutoff_hz: float = 1500.0) -> Tuple[np.ndarray, float]:
    """High-frequency-weighted spectral flux. Returns (flux, frames_per_second)."""
    if len(audio) < win:
        return np.zeros(0), sr / hop
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    hf_weight = (freqs > hf_cutoff_hz).astype(np.float32)

    n_frames = 1 + (len(audio) - win) // hop
    # Build framed matrix (n_frames, win) via stride tricks for a vectorised FFT.
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = audio[idx] * window
    mag = np.abs(np.fft.rfft(frames, axis=1))            # (n_frames, win/2+1)
    diff = np.diff(mag, axis=0)
    diff[diff < 0] = 0.0
    flux = (diff * hf_weight[None, :]).sum(axis=1)
    flux = np.concatenate([[0.0], flux])                 # align length to n_frames
    return flux, sr / hop


def detect_strikes(audio: np.ndarray, sr: int,
                   min_gap_s: float = 0.12, k: float = 6.0) -> List[float]:
    """Detect ball-strike onset times (seconds) via adaptive peak-picking."""
    flux, fps = onset_strength(audio, sr)
    if len(flux) == 0:
        return []
    # Normalise + adaptive threshold (local mean + k*std over a ~1s window).
    flux = flux / (np.median(flux) + 1e-9)
    w = int(fps * 1.0)
    if w < 3:
        w = 3
    cs = np.cumsum(np.insert(flux, 0, 0))
    local_mean = (cs[w:] - cs[:-w]) / w
    local_mean = np.concatenate([np.full(w, local_mean[0] if len(local_mean) else 0), local_mean])[:len(flux)]
    csq = np.cumsum(np.insert(flux * flux, 0, 0))
    local_ms = (csq[w:] - csq[:-w]) / w
    local_ms = np.concatenate([np.full(w, local_ms[0] if len(local_ms) else 0), local_ms])[:len(flux)]
    local_std = np.sqrt(np.maximum(local_ms - local_mean ** 2, 1e-12))
    thresh = local_mean + k * local_std

    min_gap_frames = max(1, int(min_gap_s * fps))
    strikes: List[float] = []
    last = -10 ** 9
    for i in range(1, len(flux) - 1):
        if flux[i] > thresh[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]:
            if i - last >= min_gap_frames:
                strikes.append(i / fps)
                last = i
    return strikes


def segment_rallies_audio(
    video_path: str,
    start_s: float = 0.0,
    duration_s: float = 0.0,        # 0 = whole match
    gap_s: float = 3.0,             # silence gap that ends a rally
    min_strikes: int = 3,           # a real rally has several shots
    min_rally_s: float = 1.5,
    lead_s: float = 0.8,            # include the serve motion before the first strike
    tail_s: float = 1.0,            # include the ball settling after the last strike
    k: float = 6.0,                 # strike-detection selectivity (higher = only loud sharp strikes)
) -> Dict:
    """Segment a match into rallies from the ball-strike audio."""
    audio, sr = extract_audio(video_path)
    if len(audio) == 0:
        return {"error": "no audio track", "rallies": [], "num_rallies": 0}

    total_s = len(audio) / sr
    s0 = max(0.0, start_s)
    s1 = total_s if duration_s <= 0 else min(total_s, start_s + duration_s)
    seg = audio[int(s0 * sr):int(s1 * sr)]

    strikes = [s0 + t for t in detect_strikes(seg, sr, k=k)]

    # Cluster strikes into rallies.
    rallies: List[Dict] = []
    cur: List[float] = []
    for t in strikes:
        if cur and (t - cur[-1]) > gap_s:
            rallies.append(cur); cur = []
        cur.append(t)
    if cur:
        rallies.append(cur)

    out: List[Dict] = []
    for grp in rallies:
        if len(grp) < min_strikes:
            continue
        start_t = max(s0, grp[0] - lead_s)
        end_t = min(s1, grp[-1] + tail_s)
        if end_t - start_t < min_rally_s:
            continue
        out.append({
            "rally_id": len(out) + 1,
            "start_t": round(start_t, 2),
            "end_t": round(end_t, 2),
            "duration_s": round(end_t - start_t, 2),
            "shots": len(grp),               # audio strike count ≈ shot count
            "strike_times": [round(t, 2) for t in grp],
        })

    active_s = sum(r["duration_s"] for r in out)
    span = s1 - s0
    return {
        "method": "audio",
        "span_s": round(span, 1),
        "num_rallies": len(out),
        "total_strikes": len(strikes),
        "active_play_s": round(active_s, 1),
        "active_play_pct": round(active_s / span * 100, 1) if span else 0,
        "rallies": out,
    }


# ── serve-aware boundary refinement ───────────────────────────────────────────

def _in_service_box(cx: float, cy: float, margin: float = 0.7) -> bool:
    """Is a court position (metres) inside (or just at the edge of) a service box?
    Service boxes are 1.6 m squares in the back quarters, front edge on the short
    line (y = 5.49), outer edge on a side wall."""
    from .court import COURT_WIDTH, SHORT_LINE_Y, SERVICE_BOX
    y_ok = (SHORT_LINE_Y - margin) <= cy <= (SHORT_LINE_Y + SERVICE_BOX + margin)
    in_left = (-margin) <= cx <= (SERVICE_BOX + margin)
    in_right = (COURT_WIDTH - SERVICE_BOX - margin) <= cx <= (COURT_WIDTH + margin)
    return y_ok and (in_left or in_right)


def _serve_setup_present(video_path: str, t0: float, t1: float, model, detector) -> bool:
    """Sample frames in [t0, t1] and return True if a player's feet are in a
    service box — i.e. someone is setting up to serve (a real rally boundary)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sf, ef = int(max(0, t0) * fps), int(max(0, t1) * fps)
    if ef <= sf:
        cap.release()
        return False
    found = False
    for fi in np.linspace(sf, ef, 10).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        if w > 1280:
            f = cv2.resize(f, (1280, int(h * 1280 / w)))
        pf = detector.detect_frames([f], int(fi), fps)[0]
        for pb in pf.players:
            cx, cy = model.to_court((pb.feet_x, pb.feet_y))
            if _in_service_box(cx, cy):
                found = True
                break
        if found:
            break
    cap.release()
    return found


def refine_with_serves(video_path: str, rallies: List[Dict], calibration) -> List[Dict]:
    """Merge adjacent audio rallies whose boundary is NOT a real serve.

    For each gap between consecutive rallies, check whether a player is setting up
    in a service box right as the next cluster begins. If yes → real boundary
    (keep the split). If no → it was a within-rally lull → merge.
    """
    if not calibration or len(rallies) < 2:
        return rallies
    import cv2
    from .court import CourtModel, CourtCalibration
    from .players import get_player_detector

    cap = cv2.VideoCapture(video_path)
    nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    proc_w = min(nw, 1280)
    s = proc_w / nw if nw else 1.0

    def _sc(pt):
        return (pt[0] * s, pt[1] * s) if pt is not None else None

    proc_calib = CourtCalibration(
        front_left=_sc(calibration.front_left), front_right=_sc(calibration.front_right),
        back_right=_sc(calibration.back_right), back_left=_sc(calibration.back_left))
    try:
        model = CourtModel(proc_calib)
    except Exception:
        return rallies
    detector = get_player_detector()

    merged: List[Dict] = [dict(rallies[0])]
    for cur in rallies[1:]:
        # serve window: the seconds BEFORE the first strike, when the server is
        # settled in the box getting ready (they leave the box as they hit).
        serve = _serve_setup_present(video_path, cur["start_t"] - 2.5, cur["start_t"] + 0.8,
                                     model, detector)
        if serve:
            merged.append(dict(cur))
        else:
            prev = merged[-1]
            prev["end_t"] = cur["end_t"]
            prev["duration_s"] = round(prev["end_t"] - prev["start_t"], 2)
            prev["shots"] = prev.get("shots", 0) + cur.get("shots", 0)
            prev["strike_times"] = prev.get("strike_times", []) + cur.get("strike_times", [])
            prev["merged_lull"] = True

    for i, r in enumerate(merged):
        r["rally_id"] = i + 1
    return merged
