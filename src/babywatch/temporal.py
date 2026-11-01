"""Temporal smoothing: embedding window pooling and a hysteresis state machine on P(upright)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def pool_windows(emb: np.ndarray, k: int) -> np.ndarray:
    """Causal mean over the last k rows (row i uses rows max(0,i-k+1)..i). k<=1 returns emb."""
    if k <= 1:
        return emb
    c = np.cumsum(np.vstack([np.zeros((1, emb.shape[1]), emb.dtype), emb]), axis=0)
    idx = np.arange(1, len(emb) + 1)
    lo = np.maximum(idx - k, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


class WindowPool:
    """Online version of pool_windows."""

    def __init__(self, k: int):
        self.buf: deque[np.ndarray] = deque(maxlen=max(k, 1))

    def __call__(self, e: np.ndarray) -> np.ndarray:
        self.buf.append(e)
        return np.mean(self.buf, axis=0)


@dataclass
class HysteresisParams:
    enter_p: float = 0.7
    exit_p: float = 0.3
    enter_s: float = 1.5
    exit_s: float = 2.0


class Hysteresis:
    """States: 'down' (lying/empty) and 'upright'. update() returns the new state on a transition."""

    def __init__(self, params: HysteresisParams = HysteresisParams(), state: str = "down"):
        self.p = params
        self.state = state
        self._since: float | None = None  # time the candidate condition started holding

    def update(self, t: float, prob: float) -> str | None:
        if self.state == "down":
            cond, need, nxt = prob > self.p.enter_p, self.p.enter_s, "upright"
        else:
            cond, need, nxt = prob < self.p.exit_p, self.p.exit_s, "down"
        if not cond:
            self._since = None
            return None
        if self._since is None:
            self._since = t
        if t - self._since >= need:
            self.state, self._since = nxt, None
            return nxt
        return None


def run_hysteresis(t: np.ndarray, p: np.ndarray, params: HysteresisParams) -> np.ndarray:
    """Offline: boolean upright state per sample."""
    h = Hysteresis(params)
    out = np.zeros(len(t), bool)
    for i, (ti, pi) in enumerate(zip(t, p)):
        h.update(float(ti), float(pi))
        out[i] = h.state == "upright"
    return out
