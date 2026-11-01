"""Render each clip with its label and a timeline bar burnt in -> labels/review/<clip>.mp4.

Watch them, then fix labels/intervals.csv. Optional --pred cache/preds.npz overlays model P(upright).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np

from babywatch import paths
from babywatch.labels import label_at, load_intervals
from babywatch.roi import SceneMatcher
from babywatch.video import WORK_SIZE, clip_id, iter_frames, list_clips

COLORS = {"lying": (200, 120, 40), "upright": (40, 40, 230), "empty": (120, 120, 120),
          "uncertain": (0, 200, 230), "outside": (200, 0, 200), "": (40, 40, 40)}
BAR_H = 36


def timeline(intervals, dur: float, w: int) -> np.ndarray:
    t = (np.arange(w) + 0.5) / w * dur
    lab = label_at(intervals, t)
    bar = np.zeros((BAR_H, w, 3), np.uint8)
    for x, l in enumerate(lab):
        bar[:, x] = COLORS[l]
    for s in range(0, int(dur) + 1, 10):  # 10 s ticks
        x = int(s / dur * (w - 1))
        bar[: 6 if s % 60 else 12, x] = 255
    return bar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--labels", default=str(paths.INTERVALS))
    ap.add_argument("--pred", help="npz from babywatch train --dump-preds (keys <clip>_t, <clip>_p)")
    ap.add_argument("--out", default=str(paths.LABELS / "review"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    intervals = load_intervals(a.labels)
    preds = np.load(a.pred) if a.pred else None
    matcher = SceneMatcher()
    W, H = WORK_SIZE
    for path in a.clips or list_clips():
        cid = clip_id(path)
        iv = intervals.get(cid, [])
        with av.open(str(path)) as c:
            dur = float(c.duration / 1e6)
        bar = timeline(iv, dur, W)
        dst = out / f"{cid}.mp4"
        ff = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H + BAR_H}",
             "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
             "-pix_fmt", "yuv420p", str(dst)],
            stdin=subprocess.PIPE,
        )
        poly = None
        for t, frame in iter_frames(path, fps=a.fps):
            if poly is None:
                poly = matcher.match(frame).polygon.astype(np.int32)
            lab = label_at(iv, np.array([t]))[0] or "?"
            col = COLORS.get(lab, (255, 255, 255))
            cv2.polylines(frame, [poly], True, col, 2)
            cv2.rectangle(frame, (0, H - 44), (W, H), (0, 0, 0), -1)
            txt = f"{cid}  t={t:6.1f}s  label={lab}"
            if preds is not None and f"{cid}_t" in preds:
                pt, pp = preds[f"{cid}_t"], preds[f"{cid}_p"]
                p = float(pp[min(np.searchsorted(pt, t), len(pp) - 1)])
                txt += f"  P(upright)={p:.2f}"
                cv2.rectangle(frame, (W - 30, H - 44 - int(200 * p)), (W - 10, H - 44), (0, 0, 255), -1)
            cv2.putText(frame, txt, (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            b = bar.copy()
            x = int(t / dur * (W - 1))
            b[:, max(x - 1, 0) : x + 2] = 255
            ff.stdin.write(np.vstack([frame, b]).tobytes())
        ff.stdin.close()
        ff.wait()
        print(dst)


if __name__ == "__main__":
    main()
