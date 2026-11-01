"""YOLO person boxes on full frames for every clip -> cache/persons/<clip>.npz (t, boxes padded, n, poly)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from babywatch import paths
from babywatch.adult import AdultDetector
from babywatch.roi import RoiTracker, SceneMatcher
from babywatch.video import clip_id, iter_frames, list_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--weights", default=str(paths.YOLO))
    ap.add_argument("--out", default=str(paths.CACHE / "persons"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = AdultDetector(a.weights)
    matcher = SceneMatcher()
    for p in list_clips():
        cid = clip_id(p)
        tracker = RoiTracker(matcher, every_s=30)
        ts, polys, boxes, buf = [], [], [], []

        def flush():
            boxes.extend(det.persons(buf))
            buf.clear()

        for t, frame in iter_frames(p, fps=a.fps):
            ts.append(t)
            polys.append(tracker.update(t, frame).polygon)
            buf.append(frame)
            if len(buf) == 32:
                flush()
        flush()
        n = np.array([len(b) for b in boxes])
        pad = np.zeros((len(boxes), max(n.max(), 1), 5), np.float32)
        for i, b in enumerate(boxes):
            pad[i, : len(b)] = b
        np.savez_compressed(out / f"{cid}.npz", t=np.array(ts), boxes=pad, n=n, poly=np.array(polys))
        print(cid, len(ts), "frames", flush=True)


if __name__ == "__main__":
    main()
