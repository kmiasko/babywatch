"""Crib ROI: match a frame to the best reference scene (ORB + RANSAC homography), warp its polygon, crop."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import paths

MIN_INLIERS = 150  # real matches have 400+; occlusions (door, adult at lens) give <100
MASK_FILL = 114


@dataclass
class Scene:
    name: str
    ref_path: Path
    polygon: np.ndarray  # (N, 2) float32, WORK_SIZE coords
    kp: list
    desc: np.ndarray
    shape: tuple[int, int]


@dataclass
class Match:
    scene: str
    polygon: np.ndarray
    inliers: int
    ok: bool


def _prep(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _feature_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.full((h, w), 255, np.uint8)
    m[: int(h * 0.09), : int(w * 0.40)] = 0  # burnt-in timestamp
    return m


class SceneMatcher:
    def __init__(self, config: str | Path = paths.SCENES):
        config = Path(config)
        cfg = yaml.safe_load(config.read_text())
        self.orb = cv2.ORB_create(nfeatures=3000, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.scenes: list[Scene] = []
        for s in cfg["scenes"]:
            p = (config.parent.parent / s["ref"]) if not Path(s["ref"]).is_absolute() else Path(s["ref"])
            img = cv2.imread(str(p))
            if img is None:
                raise FileNotFoundError(p)
            g = _prep(img)
            kp, desc = self.orb.detectAndCompute(g, _feature_mask(g.shape))
            self.scenes.append(
                Scene(s["name"], p, np.asarray(s["polygon"], np.float32), kp, desc, g.shape)
            )

    def match(self, frame: np.ndarray) -> Match:
        g = _prep(frame)
        kp, desc = self.orb.detectAndCompute(g, _feature_mask(g.shape))
        best: Match | None = None
        for s in self.scenes:
            m = self._match_scene(s, kp, desc)
            if m and (best is None or m.inliers > best.inliers):
                best = m
        if best is None:
            s = self.scenes[0]
            return Match(s.name, s.polygon.copy(), 0, False)
        return best

    def _match_scene(self, s: Scene, kp, desc) -> Match | None:
        if desc is None or s.desc is None or len(kp) < 10:
            return None
        pairs = self.bf.knnMatch(s.desc, desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 10:
            return Match(s.name, s.polygon.copy(), len(good), False)
        src = np.array([s.kp[m.queryIdx].pt for m in good], np.float32)
        dst = np.array([kp[m.trainIdx].pt for m in good], np.float32)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        n = int(inl.sum()) if inl is not None else 0
        if H is None or n < MIN_INLIERS:
            return Match(s.name, s.polygon.copy(), n, False)
        poly = cv2.perspectiveTransform(s.polygon[None], H)[0]
        a0, a1 = cv2.contourArea(s.polygon), cv2.contourArea(poly)
        if not (0.5 < a1 / max(a0, 1) < 2.0):  # degenerate warp
            return Match(s.name, s.polygon.copy(), n, False)
        return Match(s.name, poly, n, True)


class RoiTracker:
    """Keeps a current polygon; re-matches periodically and keeps the last good match on failure."""

    def __init__(self, matcher: SceneMatcher, every_s: float = 10.0):
        self.matcher = matcher
        self.every_s = every_s
        self.last_t: float | None = None
        self.current: Match | None = None

    def update(self, t: float, frame: np.ndarray) -> Match:
        if self.current is None or self.last_t is None or t - self.last_t >= self.every_s:
            m = self.matcher.match(frame)
            self.last_t = t
            if m.ok or self.current is None:
                self.current = m
        return self.current


def crop(frame: np.ndarray, polygon: np.ndarray, out: int = 224, margin: float = 0.04) -> np.ndarray:
    """Crop polygon bbox (+margin), grey out pixels outside the polygon, resize to out x out (BGR)."""
    h, w = frame.shape[:2]
    x0, y0 = polygon.min(0)
    x1, y1 = polygon.max(0)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = int(max(x0 - mx, 0)), int(max(y0 - my, 0))
    x1, y1 = int(min(x1 + mx, w)), int(min(y1 + my, h))
    c = frame[y0:y1, x0:x1].copy()
    mask = np.zeros(c.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(polygon - [x0, y0]).astype(np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    c[mask == 0] = MASK_FILL
    return cv2.resize(c, (out, out), interpolation=cv2.INTER_AREA)
