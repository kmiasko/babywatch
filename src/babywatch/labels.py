"""Interval labels: labels/intervals.csv -> per-timestamp labels."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import paths

CLASSES = ["lying", "upright", "empty"]


def load_intervals(path: str | Path = paths.INTERVALS) -> dict[str, list[tuple[float, float, str]]]:
    out: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["video"]].append((float(r["start_s"]), float(r["end_s"]), r["label"].strip()))
    for v in out.values():
        v.sort()
    return dict(out)


def label_at(intervals: list[tuple[float, float, str]], t: np.ndarray, guard_s: float = 0.0) -> np.ndarray:
    """Label per timestamp ('' = unlabeled). Samples within guard_s of a label change become 'uncertain'."""
    lab = np.full(len(t), "", dtype=object)
    for a, b, l in intervals:
        lab[(t >= a) & (t < b)] = l
    if guard_s > 0:
        bounds = [a for i, (a, _, l) in enumerate(intervals) if i and intervals[i - 1][2] != l]
        for x in bounds:
            lab[np.abs(t - x) <= guard_s] = "uncertain"
    return lab
