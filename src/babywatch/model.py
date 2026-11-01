"""Classifier head persistence and P(upright) prediction."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .temporal import HysteresisParams


@dataclass
class UprightModel:
    backbone: str
    img_size: int
    pool_k: int  # embedding window (samples at `fps`)
    fps: float
    head: Any  # sklearn estimator with predict_proba
    classes: list[str]
    hysteresis: HysteresisParams = field(default_factory=HysteresisParams)
    meta: dict = field(default_factory=dict)

    def p_upright(self, emb: np.ndarray) -> np.ndarray:
        emb = np.atleast_2d(emb)
        proba = self.head.predict_proba(emb)
        cls = list(self.head.classes_)
        return proba[:, cls.index("upright")]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["head"] = self.head  # keep the estimator object, not asdict's deep copy
        joblib.dump(d, path)

    @classmethod
    def load(cls, path: str | Path) -> "UprightModel":
        d = joblib.load(path)
        d["hysteresis"] = HysteresisParams(**d["hysteresis"])
        return cls(**d)
