"""
TrackNet ball detector — scaffolding for the trained squash ball model.

TrackNet (Huang et al.) is a heatmap CNN built for small fast balls in racket
sports: it consumes a short stack of consecutive frames and outputs a per-pixel
ball-probability heatmap whose peak is the ball. It is the *appearance* model the
classical detector cannot be — the empirical test showed geometry alone can't tell
the ball from reflections/racket motion.

This module provides, behind the same window-level ``BallDetector`` interface:

    TrackNet               - compact TrackNetV2-style encoder/decoder (PyTorch).
    TrackNetBallDetector   - loads weights if present, runs batched inference;
                             if no weights yet, reports unavailable and yields no
                             candidates (so the pipeline safely falls back to the
                             classical detector).
    BallHeatmapDataset     - turns exported ball labels ({match_id, frame_index,
                             x, y}) into (3-frame stack -> gaussian heatmap) samples.
    train_tracknet         - training entrypoint; runs once enough labels exist.

Nothing here requires trained weights to import. Training needs the labelled
dataset the annotation tool produces (ideally on a GPU).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .ball import BallCandidate, BallDetector, PlayerBoxes

logger = logging.getLogger(__name__)

# Network input size (H, W). Frames are letterbox-free resized to this; ball
# coordinates are scaled accordingly. Kept modest so CPU inference is feasible.
DEFAULT_INPUT_HW = (288, 512)
N_STACK = 3  # consecutive frames fed together


def pick_device(prefer: Optional[str] = None) -> str:
    """Choose the best torch device: explicit > CUDA > Apple MPS > CPU."""
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _build_tracknet(input_channels: int, imagenet: bool = False):
    """Construct the ball heatmap network: a U-Net with an ImageNet-pretrained
    ResNet18 encoder (transfer learning).

    ``imagenet=True`` initialises the encoder from ImageNet weights (torchvision
    auto-downloads, ~45MB, cached) so training starts from real visual features
    and needs far fewer squash labels. ``imagenet=False`` builds the same
    architecture with random init — used when loading our own trained checkpoint
    (no download needed at inference). The first conv is widened from 3 to
    ``input_channels`` (9 = three stacked frames), tiling the pretrained RGB
    filters so the pretrained features still apply.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights

    def cbr(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)
        )

    class ResNetUNet(nn.Module):
        def __init__(self, in_ch):
            super().__init__()
            weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet else None
            rn = resnet18(weights=weights)

            # Widen the 3-channel stem conv to in_ch, tiling pretrained weights.
            conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
            if imagenet:
                with torch.no_grad():
                    reps = (in_ch + 2) // 3
                    w = rn.conv1.weight.repeat(1, reps, 1, 1)[:, :in_ch] / reps
                    conv1.weight.copy_(w)
            self.stem = nn.Sequential(conv1, rn.bn1, rn.relu)  # -> /2, 64
            self.maxpool = rn.maxpool                          # -> /4
            self.l1 = rn.layer1   # /4,  64
            self.l2 = rn.layer2   # /8,  128
            self.l3 = rn.layer3   # /16, 256
            self.l4 = rn.layer4   # /32, 512

            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.d4 = cbr(512 + 256, 256)
            self.d3 = cbr(256 + 128, 128)
            self.d2 = cbr(128 + 64, 64)
            self.d1 = cbr(64 + 64, 32)
            self.d0 = cbr(32, 16)
            self.head = nn.Conv2d(16, 1, 1)

        def encoder_parameters(self):
            for m in (self.stem, self.l1, self.l2, self.l3, self.l4):
                yield from m.parameters()

        def forward(self, x):
            s0 = self.stem(x)              # /2,  64
            x = self.maxpool(s0)           # /4
            s1 = self.l1(x)                # /4,  64
            s2 = self.l2(s1)               # /8,  128
            s3 = self.l3(s2)               # /16, 256
            b = self.l4(s3)                # /32, 512
            u = self.d4(torch.cat([self.up(b), s3], dim=1))   # /16
            u = self.d3(torch.cat([self.up(u), s2], dim=1))   # /8
            u = self.d2(torch.cat([self.up(u), s1], dim=1))   # /4
            u = self.d1(torch.cat([self.up(u), s0], dim=1))   # /2
            u = self.d0(self.up(u))                            # /1
            return torch.sigmoid(self.head(u))                 # (B,1,H,W)

    return ResNetUNet(input_channels)


def _weighted_bce_heatmap_loss(pred, gt, pos_weight: float = 250.0, eps: float = 1e-6):
    """Weighted binary cross-entropy for Gaussian ball heatmaps (TrackNet-style).

    The ball is ~1 pixel vs ~147k background pixels, so an unweighted loss is
    minimised by predicting "low everywhere" and ignoring the ball. We weight each
    pixel by ``1 + pos_weight * gt`` so that getting the ball region right dominates
    the loss, forcing the model to actually fire on the ball.
    """
    import torch

    pred = pred.clamp(eps, 1.0 - eps)
    weight = 1.0 + pos_weight * gt
    bce = -(gt * torch.log(pred) + (1.0 - gt) * torch.log(1.0 - pred))
    return (weight * bce).mean()


def gaussian_heatmap(h: int, w: int, cx: float, cy: float, sigma: float = 5.0) -> np.ndarray:
    """Single-channel gaussian centred at (cx, cy) — the training target."""
    ys, xs = np.mgrid[0:h, 0:w]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
    return g.astype(np.float32)


def heatmap_to_point(heatmap: np.ndarray, threshold: float) -> Optional[Tuple[float, float, float]]:
    """Peak of a heatmap → (x, y, score) in heatmap pixels, or None if too weak."""
    idx = int(np.argmax(heatmap))
    y, x = np.unravel_index(idx, heatmap.shape)
    score = float(heatmap[y, x])
    if score < threshold:
        return None
    return float(x), float(y), score


# ----------------------------------------------------------------------------
# Detector (window-level, same interface as MotionBallDetector)
# ----------------------------------------------------------------------------
class TrackNetBallDetector(BallDetector):
    def __init__(
        self,
        weights_path: Optional[str] = None,
        input_hw: Tuple[int, int] = DEFAULT_INPUT_HW,
        prob_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.input_hw = input_hw
        self.prob_threshold = prob_threshold
        self.weights_path = weights_path
        self._model = None
        self._device = device
        self.available = False
        if weights_path and os.path.exists(weights_path):
            self._load(weights_path)
        else:
            logger.warning(
                "TrackNet weights not found (%s); detector will yield no candidates "
                "until the model is trained on labelled ball data.",
                weights_path,
            )

    def _load(self, weights_path: str):
        try:
            import torch
            self._device = self._device or pick_device()
            model = _build_tracknet(N_STACK * 3)
            state = torch.load(weights_path, map_location=self._device)
            model.load_state_dict(state)
            model.eval().to(self._device)
            self._model = model
            self.available = True
            logger.info("TrackNet weights loaded from %s on %s", weights_path, self._device)
        except Exception as e:  # pragma: no cover - depends on real weights
            logger.error("Failed to load TrackNet weights: %s", e)
            self.available = False

    def detect_window(
        self,
        frames_bgr: List[np.ndarray],
        start_frame_index: int,
        fps: float,
        player_boxes_per_frame: Optional[List[PlayerBoxes]] = None,
    ) -> List[List[BallCandidate]]:
        n = len(frames_bgr)
        empty: List[List[BallCandidate]] = [[] for _ in range(n)]
        if not self.available or self._model is None or n < N_STACK:
            return empty

        import cv2
        import torch

        H, W = self.input_hw
        oh, ow = frames_bgr[0].shape[:2]
        sx, sy = ow / W, oh / H

        # Precompute resized, normalized frames.
        resized = [
            cv2.resize(f, (W, H)).astype(np.float32) / 255.0 for f in frames_bgr
        ]

        out = empty
        batch_idx: List[int] = []
        batch_inp: List[np.ndarray] = []
        # Center frame i uses stack [i-1, i, i+1].
        for i in range(1, n - 1):
            stack = np.concatenate(
                [resized[i - 1], resized[i], resized[i + 1]], axis=2
            )  # (H, W, 9)
            batch_inp.append(stack.transpose(2, 0, 1))  # (9, H, W)
            batch_idx.append(i)

        if not batch_inp:
            return out

        # Inference in mini-batches: a single huge batch overflows MPS tensor-dim
        # limits, and chunking keeps memory bounded on long mining windows.
        chunk = 16
        heatmaps_list = []
        with torch.no_grad():
            for c in range(0, len(batch_inp), chunk):
                xb = torch.from_numpy(np.stack(batch_inp[c:c + chunk])).to(self._device)
                heatmaps_list.append(self._model(xb).squeeze(1).cpu().numpy())
        heatmaps = np.concatenate(heatmaps_list, axis=0)  # (B, H, W)

        for k, i in enumerate(batch_idx):
            pt = heatmap_to_point(heatmaps[k], self.prob_threshold)
            if pt is None:
                continue
            hx, hy, score = pt
            out[i] = [
                BallCandidate(
                    frame_index=start_frame_index + i,
                    timestamp=(start_frame_index + i) / fps,
                    x=hx * sx, y=hy * sy, area=0.0, score=score,
                )
            ]
        return out


# ----------------------------------------------------------------------------
# Dataset adapter + training entrypoint
# ----------------------------------------------------------------------------
@dataclass
class _Sample:
    match_id: str
    frame_index: int
    x: float
    y: float


def _build_dataset_class():
    import torch
    from torch.utils.data import Dataset
    import cv2

    class BallHeatmapDataset(Dataset):
        """Maps exported ball labels to (frame-stack -> gaussian heatmap) samples.

        Frames are decoded from video **once** at construction and cached in RAM as
        the resized 9-channel uint8 stack, so every training epoch reads from memory
        instead of re-seeking 1080p video (the dominant cost). ~1.3 MB/sample.
        """

        def __init__(
            self,
            samples: List[Dict],
            video_resolver: Callable[[str], str],
            input_hw: Tuple[int, int] = DEFAULT_INPUT_HW,
            sigma: float = 5.0,
            augment: bool = False,
        ):
            self.input_hw = input_hw
            self.sigma = sigma
            self.augment = augment
            H, W = input_hw

            # Group by video so each file is opened once; decode + resize + cache.
            by_video: Dict[str, List[_Sample]] = {}
            for s in samples:
                by_video.setdefault(s["match_id"], []).append(
                    _Sample(s["match_id"], int(s["frame_index"]),
                            float(s["x"]), float(s["y"]))
                )

            # Cache each unique frame-stack ONCE (keyed by video+frame); items just
            # reference it. This keeps oversampled duplicates near-free in memory.
            self._cache: Dict[Tuple[str, int], np.ndarray] = {}   # (9,H,W) uint8
            self._items: List[Tuple[Tuple[str, int], float, float]] = []
            self._scale = (1.0, 1.0)
            for mid, slist in by_video.items():
                cap = cv2.VideoCapture(video_resolver(mid))
                for s in slist:
                    key = (mid, s.frame_index)
                    if key not in self._cache:
                        frames = []
                        for d in (-1, 0, 1):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, s.frame_index + d))
                            ret, f = cap.read()
                            frames.append(f if ret else (frames[-1] if frames else None))
                        if frames[1] is None:
                            continue
                        frames = [f if f is not None else frames[1] for f in frames]
                        oh, ow = frames[1].shape[:2]
                        stack = np.concatenate(
                            [cv2.resize(f, (W, H)) for f in frames], axis=2
                        ).transpose(2, 0, 1)  # (9,H,W) uint8
                        self._cache[key] = np.ascontiguousarray(stack)
                        self._scale = (W / ow, H / oh)
                    sx, sy = self._scale
                    self._items.append((key, s.x * sx, s.y * sy))
                cap.release()

        def __len__(self):
            return len(self._items)

        def __getitem__(self, idx):
            import random as _rnd
            H, W = self.input_hw
            key, cx, cy = self._items[idx]
            stack = self._cache[key].astype(np.float32)  # (9,H,W)

            if self.augment:
                # Horizontal flip — squash courts are left/right symmetric, so this
                # is a valid label-preserving doubling of the data.
                if _rnd.random() < 0.5:
                    stack = stack[:, :, ::-1]
                    cx = W - 1 - cx
                # Brightness / contrast — different courts & lighting.
                if _rnd.random() < 0.8:
                    alpha = _rnd.uniform(0.75, 1.25)   # contrast
                    beta = _rnd.uniform(-25, 25)       # brightness
                    stack = stack * alpha + beta
                # Per-frame colour jitter (applied identically to all 3 frames'
                # channels so motion cues stay consistent).
                if _rnd.random() < 0.5:
                    tint = np.array([_rnd.uniform(0.9, 1.1) for _ in range(3)] * 3,
                                    dtype=np.float32)[:, None, None]
                    stack = stack * tint
                stack = np.clip(stack, 0, 255)

            x = torch.from_numpy(np.ascontiguousarray(stack) / 255.0)
            target = gaussian_heatmap(H, W, cx, cy, self.sigma)[None, :, :]
            return x, torch.from_numpy(target)

    return BallHeatmapDataset


def train_tracknet(
    samples: List[Dict],
    video_resolver: Callable[[str], str],
    out_weights: str,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    input_hw: Tuple[int, int] = DEFAULT_INPUT_HW,
    device: Optional[str] = None,
    min_samples: int = 200,
    progress_cb: Optional[Callable[[int, int, float], None]] = None,
    pretrained: bool = True,
    freeze_encoder: bool = True,    # frozen ImageNet encoder = stable BN, less overfit
) -> Dict:
    """Train TrackNet on exported ball labels and save weights.

    Returns a small report. Refuses to train below ``min_samples`` — a model
    trained on a handful of points is worse than the classical fallback. Run this
    once the annotation tool has accumulated enough confirmed ball points
    (ideally on a GPU).
    """
    if len(samples) < min_samples:
        return {
            "trained": False,
            "reason": f"need >= {min_samples} labelled ball points, have {len(samples)}",
            "samples": len(samples),
        }

    import torch
    from torch.utils.data import DataLoader

    device = pick_device(device)
    Dataset = _build_dataset_class()
    ds = Dataset(samples, video_resolver, input_hw=input_hw, augment=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # Transfer learning: start the encoder from ImageNet features.
    model = _build_tracknet(N_STACK * 3, imagenet=pretrained).to(device)
    if pretrained and freeze_encoder:
        # With little data, freeze the pretrained encoder and train only the
        # decoder/head — fewer parameters to fit, so far fewer labels needed.
        for p in model.encoder_parameters():
            p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)

    model.train()
    history = []
    for ep in range(epochs):
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = _weighted_bce_heatmap_loss(pred, yb)  # imbalance-aware (not MSE)
            loss.backward()
            opt.step()
            total += float(loss.item())
        avg = total / max(1, len(dl))
        history.append(avg)
        logger.info("TrackNet epoch %d/%d loss=%.5f", ep + 1, epochs, avg)
        if progress_cb:
            try:
                progress_cb(ep + 1, epochs, avg)
            except Exception:
                pass

    os.makedirs(os.path.dirname(out_weights) or ".", exist_ok=True)
    torch.save(model.state_dict(), out_weights)
    return {
        "trained": True,
        "samples": len(samples),
        "epochs": epochs,
        "final_loss": history[-1] if history else None,
        "weights": out_weights,
        "device": device,
        "transfer_learning": bool(pretrained),
        "encoder_frozen": bool(pretrained and freeze_encoder),
    }
