"""Watch an RTSP stream (or a file) and report when the baby is upright (sitting/standing).

Prints a JSON line whenever the notification state changes:
  {"ts": ..., "notify": true, "state": "upright", "adult": false, "p": 0.93}
notify = baby upright in the crib AND no adult in view (an adult suppresses it; see babywatch/adult.py).
  babywatch live --rtsp rtsp://user:pass@cam/stream [--show] [--log events.jsonl]
  babywatch live --file "<clip>.mp4" --realtime
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from babywatch import paths
from babywatch.adult import AdultDetector, AdultGate, adult_score
from babywatch.embed import Embedder
from babywatch.labels import load_intervals
from babywatch.model import UprightModel
from babywatch.roi import RoiTracker, SceneMatcher, crop
from babywatch.temporal import Hysteresis, WindowPool
from babywatch.video import clip_id, iter_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rtsp")
    src.add_argument("--file")
    ap.add_argument("--model", default=str(paths.MODEL))
    ap.add_argument("--scenes", default=str(paths.SCENES))
    ap.add_argument("--realtime", action="store_true", help="pace a file at native speed")
    ap.add_argument("--show", action="store_true", help="debug window with ROI and P(upright)")
    ap.add_argument("--log", help="append JSON lines here too")
    ap.add_argument("--yolo", default=str(paths.YOLO), help="person detector for the adult gate")
    ap.add_argument("--verbose", action="store_true", help="print P(upright) for every processed frame")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

    model = UprightModel.load(a.model)
    emb = Embedder(model.backbone, model.img_size)
    tracker = RoiTracker(SceneMatcher(a.scenes), every_s=10.0)
    pool = WindowPool(model.pool_k)
    hyst = Hysteresis(model.hysteresis)
    det = AdultDetector(a.yolo)
    gate = AdultGate()
    notify = False
    logf = open(a.log, "a") if a.log else None
    logging.info("model %s on %s, hysteresis %s", model.meta.get("name"), emb.device, model.hysteresis)

    def emit(rec: dict) -> None:
        line = json.dumps(rec)
        print(line, flush=True)
        if logf:
            logf.write(line + "\n")
            logf.flush()

    source = a.rtsp or a.file
    n, busy, scene = 0, 0.0, None
    for t, frame in iter_frames(source, fps=model.fps, realtime=a.realtime):
        t0 = time.perf_counter()
        m = tracker.update(t, frame)
        if m.scene != scene:
            logging.info("scene=%s inliers=%d ok=%s", m.scene, m.inliers, m.ok)
            scene = m.scene
        e = emb([crop(frame, m.polygon, model.img_size)])[0]
        p = float(model.p_upright(pool(e))[0])
        hyst.update(t, p)
        boxes = det.persons([frame])[0]
        adult = gate.update(t, boxes, m.polygon)
        new = (hyst.state == "upright") and not adult
        busy += time.perf_counter() - t0
        n += 1
        ts = datetime.fromtimestamp(t).isoformat(timespec="seconds") if a.rtsp else round(t, 1)
        rec = {"ts": ts, "notify": new, "state": hyst.state, "adult": adult, "p": round(p, 3)}
        if new != notify:
            emit(rec)
            notify = new
        elif a.verbose:
            print(json.dumps(rec), flush=True)
        if a.show:
            col = (0, 0, 255) if notify else (0, 200, 0)
            for x0, y0, x1, y1, c in boxes:
                ad = adult_score(np.array([[x0, y0, x1, y1, c]]), m.polygon) > 0
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255) if ad else (255, 200, 0), 2)
            cv2.polylines(frame, [m.polygon.astype(np.int32)], True, col, 2)
            cv2.rectangle(frame, (10, 60), (40, 260), (60, 60, 60), 1)
            cv2.rectangle(frame, (10, 260 - int(200 * p)), (40, 260), col, -1)
            cv2.putText(frame, f"{hyst.state} p={p:.2f}{' ADULT' if adult else ''}{' NOTIFY' if notify else ''}",
                        (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cv2.imshow("babywatch", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    logging.info("processed %d frames, %.1f ms/frame (%.0f fps capacity)", n, 1000 * busy / max(n, 1),
                 n / max(busy, 1e-9))
    if a.file:
        iv = load_intervals().get(clip_id(a.file), [])
        if iv:
            logging.info("labels: %s", " ".join(f"{s:.1f}-{e:.1f}:{l}" for s, e, l in iv))


if __name__ == "__main__":
    main()
