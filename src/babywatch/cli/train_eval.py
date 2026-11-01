"""Leave-one-video-out CV over {backbone x head x pooling} + YOLO baselines; fit final model.

Prints frame-level (P/R/F1/PR-AUC) and event-level (latency, false alarms/h, missed episodes) metrics,
tunes hysteresis on the LOVO predictions, and saves the best pipeline to models/upright.joblib.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from babywatch import paths
from babywatch.adult import AdultRule, adult_score, gate_offline
from babywatch.labels import label_at, load_intervals
from babywatch.model import UprightModel
from babywatch.temporal import HysteresisParams, pool_windows, run_hysteresis

warnings.filterwarnings("ignore", category=ConvergenceWarning)
EVAL_CLASSES = ("lying", "upright", "empty")
TRAIN_CLASSES = ["lying", "upright", "empty"]  # "empty" dropped with --no-train-empty
RAIL_FRAC = 0.33  # approx. top-rail height as a fraction of the ROI crop (all scenes)


@dataclass
class Clip:
    cid: str
    t: np.ndarray
    x: np.ndarray  # features (pooled embeddings)
    lab: np.ndarray  # raw labels per sample ('' / lying / upright / empty / uncertain)
    lab_g: np.ndarray  # labels with transition guard applied
    adult_gt: np.ndarray  # adult in view (labels/adult.csv)
    adult_det: np.ndarray  # adult gate from YOLO person boxes (with hold)


def make_head(kind: str):
    if kind == "logreg":
        clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, max_iter=300, early_stopping=True,
                            n_iter_no_change=15, random_state=0)
    else:
        raise ValueError(kind)
    return make_pipeline(StandardScaler(), clf)


def load_clips(cfg: str, pool_k: int, intervals, guard: float, cache: Path) -> list[Clip]:
    clips = []
    for f in sorted((cache / cfg).glob("*.npz")):
        if f.stem not in intervals:
            continue
        d = np.load(f)
        t = d["t"]
        iv = intervals.get(f.stem, [])
        x = d["emb"].astype(np.float32) if "emb" in d else d["feat"]
        clips.append(Clip(f.stem, t, pool_windows(x, pool_k), label_at(iv, t), label_at(iv, t, guard),
                          *adult_arrays(f.stem, t, cache)))
    return clips


ADULT_LABELS = paths.ADULT_LABELS
ADULT_RULE = AdultRule()


def adult_arrays(cid: str, t: np.ndarray, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    gt = np.zeros(len(t), bool)
    with open(ADULT_LABELS) as fh:
        for r in csv.DictReader(fh):
            if r["video"] == cid:
                gt |= (t >= float(r["start_s"])) & (t < float(r["end_s"]))
    f = cache / "persons" / f"{cid}.npz"
    if not f.exists():
        return gt, np.zeros(len(t), bool)
    d = np.load(f)
    i = np.clip(np.searchsorted(d["t"], t - 1e-3), 0, len(d["t"]) - 1)
    sc = np.array([adult_score(d["boxes"][j, : d["n"][j]], d["poly"][j], ADULT_RULE) for j in i])
    return gt, gate_offline(t, sc, ADULT_RULE.hold_s)


def train_mask(c: Clip, classes=TRAIN_CLASSES) -> np.ndarray:
    return np.isin(c.lab_g, classes)


def eval_mask(c: Clip) -> np.ndarray:
    return np.isin(c.lab, EVAL_CLASSES)


def lovo_predict(clips: list[Clip], head: str, classes=TRAIN_CLASSES) -> dict[str, np.ndarray]:
    out = {}
    for i, c in enumerate(clips):
        tr = [o for j, o in enumerate(clips) if j != i]
        X = np.vstack([o.x[train_mask(o, classes)] for o in tr])
        y = np.concatenate([o.lab_g[train_mask(o, classes)] for o in tr])
        m = make_head(head).fit(X, y)
        out[c.cid] = m.predict_proba(c.x)[:, list(m.classes_).index("upright")]
    return out


def episodes(mask: np.ndarray, t: np.ndarray) -> list[tuple[float, float]]:
    eps, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = t[i]
        if not v and start is not None:
            eps.append((start, t[i]))
            start = None
    if start is not None:
        eps.append((start, t[-1] + 1e-3))
    return eps


def event_metrics(clips: list[Clip], preds: dict[str, np.ndarray], hp: HysteresisParams) -> dict:
    lat, missed, n_eps, fa, down_s = [], 0, 0, 0, 0.0
    tp = fp = fn = 0
    for c in clips:
        p = preds[c.cid]
        st = run_hysteresis(c.t, p, hp)
        dt = np.median(np.diff(c.t))
        st = st & ~c.adult_det  # notification = upright and no adult in view
        ev = eval_mask(c)
        gt = (c.lab == "upright") & ~c.adult_gt  # should notify
        down_s += float((ev & ~gt).sum() * dt)
        tp += int((st & gt & ev).sum())
        fp += int((st & ~gt & ev).sum())
        fn += int((~st & gt & ev).sum())
        # true episodes: upright runs, merged across uncertain gaps
        care = gt | ~ev  # a notification episode touching only should-not-notify samples is a false alarm
        for a, b in episodes(gt, c.t):
            n_eps += 1
            hit = np.where(st & (c.t >= a) & (c.t < b + 2.0))[0]
            if len(hit):
                lat.append(max(0.0, c.t[hit[0]] - a))
            else:
                missed += 1
        for a, b in episodes(st, c.t):
            if not np.any(care & (c.t >= a - 1.0) & (c.t < b)):
                fa += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "sm_f1": 2 * prec * rec / max(prec + rec, 1e-9), "sm_p": prec, "sm_r": rec,
        "lat_med": float(np.median(lat)) if lat else float("nan"), "lat_max": float(np.max(lat)) if lat else float("nan"),
        "missed": missed, "episodes": n_eps, "fa": fa, "fa_per_h": fa / max(down_s / 3600, 1e-9),
    }


def frame_metrics(clips: list[Clip], preds: dict[str, np.ndarray], thr: float = 0.5) -> dict:
    ys, ps, f1s = [], [], []
    for c in clips:
        ev = eval_mask(c)
        y = c.lab[ev] == "upright"
        p = preds[c.cid][ev]
        ys.append(y)
        ps.append(p)
        f1s.append(f1_score(y, p > thr, zero_division=1.0) if y.any() else float(((p > thr) == y).mean()))
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {"P": precision_score(y, p > thr, zero_division=0), "R": recall_score(y, p > thr),
            "F1": f1_score(y, p > thr), "PR_AUC": average_precision_score(y, p), "F1_fold_mean": float(np.mean(f1s)),
            "F1_folds": f1s}


def tune_hysteresis(clips, preds) -> tuple[HysteresisParams, dict]:
    best, best_key, best_m = None, None, None
    for ep, xp, es, xs in itertools.product([0.5, 0.6, 0.7, 0.8, 0.9], [0.2, 0.3, 0.4], [0.4, 0.8, 1.0, 1.5],
                                            [1.0, 2.0, 3.0]):
        hp = HysteresisParams(ep, xp, es, xs)
        m = event_metrics(clips, preds, hp)
        # ties at 2-decimal F1 go to the more conservative setting (it was tuned on only a few clips)
        key = (m["missed"], m["fa"], -round(m["sm_f1"], 2), -ep, -es, xp, -xs)
        if best_key is None or key < best_key:
            best, best_key, best_m = hp, key, m
    assert best is not None and best_m is not None
    return best, best_m


def yolo_baselines(intervals, guard, cache: Path) -> list[tuple[str, dict, dict]]:
    clips = load_clips("yolo", 1, intervals, guard, cache)
    rows = []
    for name, score in [
        ("yolo:top-above-rail", lambda f: np.where(f[:, 0] > 0, RAIL_FRAC - f[:, 1], -1.0)),
        ("yolo:aspect>1", lambda f: np.where(f[:, 0] > 0, f[:, 3] - 1.0, -1.0)),
    ]:
        raw = {c.cid: score(c.x) for c in clips}
        preds = {k: 1 / (1 + np.exp(-20 * v)) for k, v in raw.items()}  # monotone squash, thr 0.5 == score 0
        rows.append((name, frame_metrics(clips, preds), event_metrics(clips, preds, HysteresisParams())))
    preds = lovo_predict(clips, "logreg")
    rows.append(("yolo:feat+logreg", frame_metrics(clips, preds), event_metrics(clips, preds, HysteresisParams())))
    return rows


def fmt_row(name, fm, em) -> str:
    return (f"{name:34s} {fm['P']:.3f} {fm['R']:.3f} {fm['F1']:.3f} {fm['F1_fold_mean']:.3f} {fm['PR_AUC']:.3f} | "
            f"{em['sm_f1']:.3f} {em['lat_med']:5.1f} {em['lat_max']:5.1f} {em['fa_per_h']:6.1f} "
            f"{em['missed']}/{em['episodes']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=["dinov2_b@224", "siglip2_b@224"])
    ap.add_argument("--heads", nargs="*", default=["logreg", "mlp"])
    ap.add_argument("--pools", nargs="*", type=int, default=[1, 5, 10], help="window in samples (5 fps)")
    ap.add_argument("--guard", type=float, default=0.5, help="drop +-s around label transitions for training")
    ap.add_argument("--labels", default=str(paths.INTERVALS))
    ap.add_argument("--cache", default=str(paths.CACHE))
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--train-empty", choices=["yes", "no", "both"], default="both",
                    help="train on empty-crib / baby-outside frames as class 'empty' (never scored either way)")
    ap.add_argument("--exclude", nargs="*", default=[], help="clip ids to leave out entirely (held-out test)")
    ap.add_argument("--out", default=str(paths.MODEL))
    ap.add_argument("--dump-preds", default=str(paths.CACHE / "preds.npz"), help="LOVO preds of the chosen pipeline")
    a = ap.parse_args()
    intervals = {k: [(s, e, "empty" if l == "outside" else l) for s, e, l in v]
                 for k, v in load_intervals(a.labels).items() if k not in a.exclude}
    cache = Path(a.cache)

    hdr = (f"{'pipeline':34s} {'P':5s} {'R':5s} {'F1':5s} {'F1fm':5s} {'PRAUC':5s} | "
           f"{'smF1':5s} {'latM':5s} {'latX':5s} {'FA/h':6s} miss")
    print("frame metrics at p>0.5 (pooled over folds; F1fm = mean of per-fold F1) | after default hysteresis")
    print(hdr)
    results = []
    empties = {"yes": [True], "no": [False], "both": [True, False]}[a.train_empty]
    for cfg, head, k, te in itertools.product(a.configs, a.heads, a.pools, empties):
        clips = load_clips(cfg, k, intervals, a.guard, cache)
        classes = TRAIN_CLASSES if te else ["lying", "upright"]
        preds = lovo_predict(clips, head, classes)
        fm = frame_metrics(clips, preds)
        em = event_metrics(clips, preds, HysteresisParams())
        name = f"{cfg}+{head}+pool{k}" + ("" if te else "+noempty")
        results.append((name, cfg, head, k, clips, preds, fm, em, classes))
        print(fmt_row(name, fm, em), flush=True)
    print("-- baselines")
    for name, fm, em in yolo_baselines(intervals, a.guard, cache):
        print(fmt_row(name, fm, em))

    # pick: best mean per-fold F1, ties -> PR-AUC
    best = max(results, key=lambda r: (round(r[6]["F1_fold_mean"], 3), r[6]["PR_AUC"]))
    name, cfg, head, k, clips, preds, fm, _, classes = best
    hp, em = tune_hysteresis(clips, preds)
    print(f"\nchosen: {name}   per-fold F1: " + " ".join(f"{c.cid}={f:.3f}" for c, f in zip(clips, fm["F1_folds"])))
    print(f"tuned hysteresis: {asdict(hp)}")
    print(f"after tuned hysteresis: F1={em['sm_f1']:.3f} P={em['sm_p']:.3f} R={em['sm_r']:.3f} "
          f"latency median={em['lat_med']:.1f}s max={em['lat_max']:.1f}s FA/h={em['fa_per_h']:.1f} "
          f"missed={em['missed']}/{em['episodes']}")
    ok = em["sm_f1"] >= 0.95 and em["missed"] == 0 and em["lat_med"] <= 3.0
    print("targets (F1>=0.95, 0 missed, median latency<=3s):", "MET" if ok else "NOT MET")

    if a.dump_preds:
        np.savez(a.dump_preds, **{f"{c.cid}_t": c.t for c in clips}, **{f"{c.cid}_p": preds[c.cid] for c in clips})
    X = np.vstack([c.x[train_mask(c, classes)] for c in clips])
    y = np.concatenate([c.lab_g[train_mask(c, classes)] for c in clips])
    m = make_head(head).fit(X, y)
    b, s = cfg.split("@")
    UprightModel(b, int(s), k, a.fps, m, list(m.classes_), hp,
                 {"cv": {kk: v for kk, v in fm.items() if kk != "F1_folds"}, "cv_events": em, "name": name}).save(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
