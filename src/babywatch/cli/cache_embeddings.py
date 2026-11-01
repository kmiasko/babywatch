"""Decode every clip at --fps, crop the crib ROI, embed with each backbone config -> cache/<cfg>/<clip>.npz.

Also caches YOLO11 person detections in the ROI for the baseline -> cache/yolo/<clip>.npz.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from babywatch import paths
from babywatch.embed import Embedder
from babywatch.roi import RoiTracker, SceneMatcher, crop
from babywatch.video import clip_id, iter_frames, list_clips

CONFIGS = ["dinov2_b@224", "siglip2_b@224"]  # dinov2_b@448 also supported (slower, no better in CV)


def parse_cfg(c: str) -> tuple[str, int]:
    b, s = c.split("@")
    return b, int(s)


def decode_crops(path: Path, fps: float, size: int, matcher: SceneMatcher) -> tuple[np.ndarray, list[np.ndarray], list]:
    tracker = RoiTracker(matcher, every_s=30)
    ts, crops, polys = [], [], []
    for t, frame in iter_frames(path, fps=fps):
        m = tracker.update(t, frame)
        ts.append(t)
        crops.append(crop(frame, m.polygon, size))
        polys.append(m.polygon)
    return np.array(ts), crops, polys


def yolo_features(crops: list[np.ndarray], model) -> np.ndarray:
    """Per crop: [found, top_frac, bottom_frac, aspect(h/w), conf] of the most confident person."""
    feats = np.zeros((len(crops), 5), np.float32)
    feats[:, 1] = 1.0
    for i in range(0, len(crops), 32):
        res = model.predict(crops[i : i + 32], classes=[0], conf=0.15, verbose=False, imgsz=crops[0].shape[0])
        for j, r in enumerate(res):
            if len(r.boxes) == 0:
                continue
            k = int(r.boxes.conf.argmax())
            x0, y0, x1, y1 = r.boxes.xyxy[k].tolist()
            h = crops[0].shape[0]
            feats[i + j] = [1, y0 / h, y1 / h, (y1 - y0) / max(x1 - x0, 1), float(r.boxes.conf[k])]
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--yolo", default=str(paths.YOLO), help="'' to skip the baseline")
    ap.add_argument("--out", default=str(paths.CACHE))
    a = ap.parse_args()
    matcher = SceneMatcher()
    clips = list_clips()
    sizes = sorted({parse_cfg(c)[1] for c in a.configs} | ({448} if a.yolo else set()))
    crops_by_clip: dict[str, dict[int, tuple]] = {}
    for p in clips:
        cid = clip_id(p)
        crops_by_clip[cid] = {}
        for s in sizes:
            crops_by_clip[cid][s] = decode_crops(p, a.fps, s, matcher)
        print(f"decoded {cid}: {len(crops_by_clip[cid][sizes[0]][0])} frames", flush=True)

    for cfg in a.configs:
        b, s = parse_cfg(cfg)
        emb = Embedder(b, s)
        d = Path(a.out) / cfg
        d.mkdir(parents=True, exist_ok=True)
        for cid, by_size in crops_by_clip.items():
            ts, crops, polys = by_size[s]
            t0 = time.time()
            e = emb(crops)
            dt = time.time() - t0
            np.savez_compressed(d / f"{cid}.npz", t=ts, emb=e.astype(np.float16), poly=np.array(polys))
            print(f"{cfg} {cid}: {e.shape} {len(crops) / dt:.0f} img/s on {emb.device}", flush=True)
        del emb

    if a.yolo:
        from ultralytics import YOLO

        model = YOLO(a.yolo)
        d = Path(a.out) / "yolo"
        d.mkdir(parents=True, exist_ok=True)
        for cid, by_size in crops_by_clip.items():
            ts, crops, _ = by_size[448]
            np.savez_compressed(d / f"{cid}.npz", t=ts, feat=yolo_features(crops, model))
            print(f"yolo {cid} done", flush=True)


if __name__ == "__main__":
    main()
