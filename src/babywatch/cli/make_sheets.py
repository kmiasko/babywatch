"""Render ROI-crop contact sheets (1 fps, timestamps) for labeling -> labels/sheets/*.jpg.

Also writes labels/sheets/<clip>_roi.jpg: a full frame with the matched crib polygon.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from babywatch import paths
from babywatch.roi import RoiTracker, SceneMatcher, crop
from babywatch.video import clip_id, iter_frames, list_clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", help="default: all clips in $BABYWATCH_HOME/recordings")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--tile", type=int, default=200)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--clahe", action="store_true", help="contrast-enhance tiles (easier to read IR)")
    ap.add_argument("--out", default=str(paths.LABELS / "sheets"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    matcher = SceneMatcher()
    for path in a.clips or list_clips():
        cid = clip_id(path)
        tracker = RoiTracker(matcher, every_s=30)
        tiles: list[np.ndarray] = []
        first = True
        for t, frame in iter_frames(path, fps=a.fps):
            m = tracker.update(t, frame)
            if first:
                dbg = frame.copy()
                cv2.polylines(dbg, [m.polygon.astype(np.int32)], True, (0, 255, 0), 2)
                cv2.putText(dbg, f"{m.scene} inliers={m.inliers} ok={m.ok}", (10, 520),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(str(out / f"{cid}_roi.jpg"), dbg)
                print(cid, m.scene, m.inliers, m.ok)
                first = False
            c = crop(frame, m.polygon, a.tile)
            if a.clahe:
                lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)
                lab[..., 0] = cv2.createCLAHE(3.0, (4, 4)).apply(lab[..., 0])
                c = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            cv2.rectangle(c, (0, 0), (52, 20), (0, 0, 0), -1)
            cv2.putText(c, f"{int(t)}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            tiles.append(c)
        per = a.cols * a.rows
        for si in range(0, len(tiles), per):
            chunk = tiles[si : si + per]
            chunk += [np.zeros_like(tiles[0])] * (per - len(chunk))
            rows = [np.hstack(chunk[r * a.cols : (r + 1) * a.cols]) for r in range(a.rows)]
            cv2.imwrite(str(out / f"{cid}_{si // per:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
