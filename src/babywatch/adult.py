"""Adult-in-view gate: YOLO person boxes on the full frame; any person that is not the baby inside the crib ROI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths


@dataclass
class AdultRule:
    conf: float = 0.35
    max_inside_frac: float = 0.8  # a box with less of its area inside the ROI bbox than this is not "the baby in the crib"
    max_height_frac: float = 0.9  # a box taller than this fraction of the ROI height is an adult (leaning in / at the lens)
    hold_s: float = 5.0  # keep suppressing this long after the adult was last seen


def adult_score(boxes: np.ndarray, polygon: np.ndarray, rule: AdultRule = AdultRule()) -> float:
    """boxes: (N, 5) [x0, y0, x1, y1, conf] in work coords. Returns max conf of an adult-like box (0 if none)."""
    if len(boxes) == 0:
        return 0.0
    rx0, ry0 = polygon.min(0)
    rx1, ry1 = polygon.max(0)
    rh = ry1 - ry0
    best = 0.0
    for x0, y0, x1, y1, c in boxes:
        if c < rule.conf:
            continue
        area = max(x1 - x0, 1) * max(y1 - y0, 1)
        ix = max(0.0, min(x1, rx1) - max(x0, rx0))
        iy = max(0.0, min(y1, ry1) - max(y0, ry0))
        inside = ix * iy / area
        if inside < rule.max_inside_frac or (y1 - y0) > rule.max_height_frac * rh:
            best = max(best, float(c))
    return best


class AdultDetector:
    def __init__(self, weights: str | Path = paths.YOLO, imgsz: int = 960, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = device

    def persons(self, frames: list[np.ndarray], conf: float = 0.25) -> list[np.ndarray]:
        res = self.model.predict(frames, classes=[0], conf=conf, imgsz=self.imgsz, verbose=False,
                                 device=self.device)
        out = []
        for r in res:
            b = r.boxes
            out.append(np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]) if len(b) else np.zeros((0, 5)))
        return out


class AdultGate:
    """Online: True while an adult was seen within the last hold_s seconds."""

    def __init__(self, rule: AdultRule = AdultRule()):
        self.rule = rule
        self.last_seen: float | None = None

    def update(self, t: float, boxes: np.ndarray, polygon: np.ndarray) -> bool:
        if adult_score(boxes, polygon, self.rule) > 0:
            self.last_seen = t
        return self.last_seen is not None and t - self.last_seen <= self.rule.hold_s


def gate_offline(t: np.ndarray, scores: np.ndarray, hold_s: float) -> np.ndarray:
    """Boolean 'adult present (with hold)' per sample from per-sample adult scores."""
    out = np.zeros(len(t), bool)
    last = -1e9
    for i, (ti, s) in enumerate(zip(t, scores)):
        if s > 0:
            last = ti
        out[i] = ti - last <= hold_s
    return out
