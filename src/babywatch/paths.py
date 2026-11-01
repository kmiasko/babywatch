"""Data directory layout. Everything outside the wheel lives in $BABYWATCH_HOME (default ~/.babywatch)."""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path(os.environ.get("BABYWATCH_HOME", "~/.babywatch")).expanduser()
SCENES = HOME / "config" / "scenes.yaml"  # camera positions; their "ref" paths are relative to HOME
REFS = HOME / "refs"
MODELS = HOME / "models"
MODEL = MODELS / "upright.joblib"
YOLO = MODELS / "yolo11m.pt"
LABELS = HOME / "labels"
INTERVALS = LABELS / "intervals.csv"
ADULT_LABELS = LABELS / "adult.csv"
CACHE = HOME / "cache"
RECORDINGS = HOME / "recordings"
