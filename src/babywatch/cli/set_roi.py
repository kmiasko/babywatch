"""Click the crib polygon on a reference frame and store it in $BABYWATCH_HOME/config/scenes.yaml.

Include ~one standing-baby height above the top rail. Left click = add point, right click = undo,
Enter = save, Esc = cancel.
  babywatch set-roi --name pos_e_night --video "<clip>.mp4" --t 5      # grabs + saves refs/<name>.png
  babywatch set-roi --name pos_e_night --rtsp rtsp://...                # grab from the camera
  babywatch set-roi --name pos_a_night --ref refs/20261020-210452.png   # re-draw an existing scene
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from babywatch import paths
from babywatch.video import WORK_SIZE, iter_frames, read_frame_at


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref", help="existing reference image (work size)")
    g.add_argument("--video")
    g.add_argument("--rtsp")
    ap.add_argument("--t", type=float, default=5.0, help="time in --video to grab")
    ap.add_argument("--config", default=str(paths.SCENES))
    a = ap.parse_args()

    if a.ref:
        ref_path = Path(a.ref)
        img = cv2.imread(str(ref_path))
        if img is None:
            sys.exit(f"cannot read {ref_path}")
    else:
        img = read_frame_at(a.video, a.t) if a.video else next(iter_frames(a.rtsp, fps=None, reconnect=False))[1]
        ref_path = paths.REFS / f"{a.name}.png"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(ref_path), img)
    if (img.shape[1], img.shape[0]) != WORK_SIZE:
        img = cv2.resize(img, WORK_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(ref_path), img)

    pts: list[tuple[int, int]] = []

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif ev == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    win = "set_roi: L=add R=undo Enter=save Esc=cancel"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = img.copy()
        if pts:
            cv2.polylines(vis, [np.array(pts, np.int32)], len(pts) > 2, (0, 255, 0), 2)
            for p in pts:
                cv2.circle(vis, p, 4, (0, 0, 255), -1)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            sys.exit("cancelled")
        if k in (10, 13) and len(pts) >= 3:
            break
    cv2.destroyAllWindows()

    cfg_path = Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {"scenes": []}
    root = cfg_path.resolve().parent.parent  # scene "ref" paths are stored relative to the data dir
    ref = ref_path.resolve()
    ref_str = str(ref.relative_to(root)) if ref.is_relative_to(root) else str(ref)
    entry = {"name": a.name, "ref": ref_str, "polygon": [list(p) for p in pts]}
    cfg["scenes"] = [s for s in cfg["scenes"] if s["name"] != a.name] + [entry]
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
    print(f"saved scene {a.name} ({len(pts)} points) -> {cfg_path}")


if __name__ == "__main__":
    main()
