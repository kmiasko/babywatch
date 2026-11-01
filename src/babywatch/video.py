"""Frame reader for files and RTSP streams (PyAV), with fps subsampling and reconnect."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import av
import cv2
import numpy as np

from . import paths

log = logging.getLogger(__name__)

WORK_SIZE = (960, 540)  # canonical (w, h) all processing runs at


def clip_id(path: str | Path) -> str:
    """'20261020-210452_(0:00 - 5:40).mp4' -> '20261020-210452'."""
    return Path(path).name.split("_")[0]


def _duration(path: Path) -> float:
    with av.open(str(path)) as c:
        return float(c.duration or 0) / 1e6


def list_clips(root: str | Path = paths.RECORDINGS) -> list[Path]:
    """All *.mp4/*.mkv clips; if two share a clip id (an excerpt and its full recording), keep the longer."""
    best: dict[str, Path] = {}
    for p in sorted([*Path(root).glob("*.mp4"), *Path(root).glob("*.mkv")]):
        cid = clip_id(p)
        if cid not in best or _duration(p) > _duration(best[cid]):
            best[cid] = p
    return sorted(best.values())


def _to_work(img: np.ndarray, size: tuple[int, int] | None) -> np.ndarray:
    if size and (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _open(src: str, is_stream: bool):
    opts = {"rtsp_transport": "tcp", "timeout": "5000000"} if is_stream else {}
    c = av.open(src, options=opts, timeout=10.0 if is_stream else None)
    vs = c.streams.video[0]
    vs.thread_type = "AUTO"
    return c, vs


def iter_frames(
    src: str | Path,
    fps: float | None = 5.0,
    size: tuple[int, int] | None = WORK_SIZE,
    realtime: bool = False,
    reconnect: bool = True,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t, bgr_frame). For files t is media time in s; for streams it is unix time.

    fps=None yields every decoded frame. realtime=True paces a file at its native speed.
    """
    src = str(src)
    is_stream = "://" in src
    step = 1.0 / fps if fps else 0.0
    backoff = 1.0
    while True:
        try:
            container, vs = _open(src, is_stream)
        except av.error.FFmpegError as e:
            if not (is_stream and reconnect):
                raise
            log.warning("open failed (%s); retry in %.0fs", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        backoff = 1.0
        next_t = None
        wall0 = time.monotonic()
        try:
            for frame in container.decode(vs):
                if is_stream:
                    t = time.time()
                else:
                    if frame.pts is None:
                        continue
                    t = float(frame.pts * vs.time_base)
                if next_t is not None and t < next_t:
                    continue
                next_t = t + step if next_t is None else max(next_t + step, t - step)
                if realtime and not is_stream:
                    lag = t - (time.monotonic() - wall0)
                    if lag > 0:
                        time.sleep(lag)
                yield t, _to_work(frame.to_ndarray(format="bgr24"), size)
        except av.error.FFmpegError as e:
            if not (is_stream and reconnect):
                raise
            log.warning("stream error (%s); reconnecting", e)
        finally:
            container.close()
        if not is_stream:
            return
        if not reconnect:
            return
        log.warning("stream ended; reconnecting in %.0fs", backoff)
        time.sleep(backoff)


def read_frame_at(src: str | Path, t: float, size: tuple[int, int] | None = WORK_SIZE) -> np.ndarray:
    """Return the first frame at or after t seconds (files only)."""
    with av.open(str(src)) as c:
        vs = c.streams.video[0]
        c.seek(int(max(t - 2, 0) / vs.time_base), stream=vs, backward=True)
        for frame in c.decode(vs):
            if frame.pts is not None and frame.pts * vs.time_base >= t:
                return _to_work(frame.to_ndarray(format="bgr24"), size)
    raise ValueError(f"no frame at {t}s in {src}")
