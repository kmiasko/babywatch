# babywatch: architecture and usage

babywatch watches a nanny-cam stream and sends a notification when the baby is **upright in the crib**
(sitting or standing). It sends nothing while the baby lies down, while the crib is empty, or while an adult is
in view.

- [1. What it decides](#1-what-it-decides)
- [2. Architecture](#2-architecture)
- [3. Data directory](#3-data-directory)
- [4. Installation](#4-installation)
- [5. Usage](#5-usage)
- [6. Retraining](#6-retraining)
- [7. Evaluation and current results](#7-evaluation-and-current-results)
- [8. Known limitations](#8-known-limitations)

## 1. What it decides

For every processed frame (5 per second):

```
notify = (baby state is "upright" after smoothing) AND NOT (adult seen in the last 5 s)
```

| Situation | Classifier output | Adult gate | Notification |
|---|---|---|---|
| Baby lying in the crib | lying | – | no |
| Baby sits or stands in the crib, nobody else in view | upright | off | **yes** |
| Baby upright, adult in the room (sofa, leaning in, at the lens) | upright | on | no |
| Crib empty, adult walking around | empty | on/off | no |
| Baby out of the crib (in arms, on the floor) | empty | on | no |
| Door or object blocks the view of the crib | empty | – | no |

Crawling and getting up are treated as transitions. They were not used to train or score the model.

## 2. Architecture

### 2.1 Pipeline

```
 RTSP / file
     │  PyAV decode, subsample to 5 fps, resize to 960×540            babywatch/video.py
     ▼
 ┌──────────────┐  ORB features + RANSAC homography against the      babywatch/roi.py
 │ Scene match  │  reference frame of each camera position;
 │ (every 10 s) │  warps that position's crib polygon onto the frame
 └──────┬───────┘
        │ crib polygon
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
 ┌──────────────┐ crop polygon bbox, grey out   ┌──────────────┐ YOLO11m person boxes on the   babywatch/adult.py
 │ ROI crop     │ pixels outside it, 224×224     │ Person detect│ full frame
 └──────┬───────┘                                └──────┬───────┘
        ▼                                               ▼
 ┌──────────────┐ frozen DINOv2 ViT-B/14         ┌──────────────┐ a box is an "adult" if <80 % of
 │ Embedding    │ (CLS + mean patch, 1536-d),    │ Adult rule   │ it lies inside the crib area or
 │              │ fp16 on CUDA                   │              │ it is taller than 90 % of it
 └──────┬───────┘   babywatch/embed.py           └──────┬───────┘
        ▼                                               ▼
 ┌──────────────┐ mean over the last 1 s         ┌──────────────┐ adult = seen within the last 5 s
 │ Window pool  │ (5 embeddings)                 │ Hold         │
 └──────┬───────┘   babywatch/temporal.py        └──────┬───────┘
        ▼                                               │
 ┌──────────────┐ MLP (256 hidden), classes      │
 │ Classifier   │ lying / upright / empty        │
 │              │ → P(upright)                   │
 └──────┬───────┘   babywatch/model.py           │
        ▼                                               │
 ┌──────────────┐ enter "upright" at p > 0.9 held 1 s,  │
 │ Hysteresis   │ leave at p < 0.2 held 1 s             │
 └──────┬───────┘   babywatch/temporal.py               │
        └──────────────────────┬────────────────────────┘
                               ▼
                notify = upright AND NOT adult
                → JSON line on every change            babywatch/cli/live.py
```

### 2.2 Components

The project uses the src layout. Module paths in the diagram above are relative to `src/`.

```
record/
├── pyproject.toml          package metadata, `babywatch` entry point, build config (hatchling)
├── requirements.lock       pinned dependency versions for installation
├── uv.lock                 uv lockfile for development
├── README.md
├── docs/architecture-and-usage.md
└── src/babywatch/          the package
    ├── __main__.py         `python -m babywatch`
    ├── adult.py  embed.py  labels.py  model.py  paths.py  roi.py  temporal.py  video.py
    └── cli/                one module per `babywatch <command>`
```

| Module | Responsibility | Key details |
|---|---|---|
| `src/babywatch/video.py` | Frame source | PyAV decoding of files and RTSP (TCP). Subsamples to a target fps and resizes to `WORK_SIZE` = 960×540. RTSP reconnects with backoff of 1 s, doubling up to 30 s. `list_clips()` finds recordings; if two files share a clip ID, it keeps the longer one. |
| `src/babywatch/roi.py` | Crib region | `SceneMatcher` holds one reference frame and polygon per camera position. It matches the live frame with ORB (3000 features, CLAHE, burnt-in timestamp masked) and a RANSAC homography. A match needs ≥150 inliers; real matches give 400+, occlusions <100. `RoiTracker` re-matches every 10 s and keeps the last good polygon on failure. `crop()` cuts the polygon's bounding box, greys out everything outside the polygon and resizes to a square. |
| `src/babywatch/embed.py` | Features | `Embedder` wraps timm backbones: `dinov2_b` (`vit_base_patch14_dinov2.lvd142m`, CLS + mean patch tokens) and `siglip2_b` (`vit_base_patch16_siglip_224.v2_webli`, attention-pool output). Weights download from the Hugging Face hub on first use. |
| `src/babywatch/temporal.py` | Smoothing | `WindowPool` / `pool_windows` average the last k embeddings (causal). `Hysteresis` is a two-state machine (`down` / `upright`) with separate enter and exit thresholds and minimum durations. |
| `src/babywatch/model.py` | Trained artifact | `UprightModel`: backbone name, crop size, pooling window, fps, the sklearn head, hysteresis parameters and CV metadata, saved with joblib. |
| `src/babywatch/adult.py` | Adult gate | `AdultDetector` runs YOLO11m (person class, conf ≥ 0.25) on the full frame. `adult_score()` applies `AdultRule`: conf ≥ 0.35, and either <80 % of the box inside the crib bbox or box height >90 % of the crib bbox height. `AdultGate` holds the suppression for 5 s after the last adult box. |
| `src/babywatch/labels.py` | Labels | Reads `intervals.csv` and converts intervals into per-timestamp labels, with an optional guard band around label changes. |
| `src/babywatch/paths.py` | Data layout | All default paths under `$BABYWATCH_HOME` (default `~/.babywatch`). |
| `src/babywatch/cli/` | Commands | `babywatch <command>`. See [Usage](#5-usage). |

### 2.3 Why this design

- **Crop first, then classify.** The crib is small in the frame (about 150×80 px in the 640×360 stream). Classifying the whole frame would mostly see the room. The crop also removes most of the sofa and door area.
- **Frozen foundation-model features with a small head.** There are only ~14 min of unique labeled video. DINOv2 features separate lying from upright with a few hundred labels, even in grainy IR with the baby behind bars. Generic pose models (YOLO-pose), built for adults in clear views, do much worse. The YOLO "head above rail" baseline missed 4–8 of 11 episodes.
- **Separate adult gate instead of a learned class.** "Is an adult in view?" is what generic person detectors are good at, and it needs no training data from this room. The rule only has to tell the baby (a box inside the crib area) from everyone else.
- **Homography to reference frames instead of a fixed pixel ROI.** The camera has been in four positions. Matching each frame to stored references finds the crib again after the camera is nudged or moved, and refuses to match when the view is blocked.

### 2.4 Throughput

On an RTX 4080 SUPER, `babywatch live` takes about 16 ms per frame: embedding ~4 ms, YOLO11m at 960 px about 11 ms, the rest ROI and decoding. That is roughly 60 fps of capacity against the 5 fps it needs.

## 3. Data directory

The wheel contains only code. Everything else lives in `~/.babywatch`, or in `$BABYWATCH_HOME` if set (`set -x BABYWATCH_HOME /path`).

```
~/.babywatch/
├── config/scenes.yaml      camera positions: name, reference frame, crib polygon
├── refs/*.png              reference frames (960×540)
├── models/
│   ├── upright.joblib      trained classifier + hysteresis (babywatch train)
│   └── yolo11m.pt          person detector (downloaded by ultralytics if missing)
├── labels/                 training only
│   ├── intervals.csv       video,start_s,end_s,label
│   ├── adult.csv           video,start_s,end_s   (adult in view)
│   ├── sheets/             contact sheets (babywatch sheets)
│   └── review/             review videos (babywatch review)
├── cache/                  training only: embeddings, person boxes, CV predictions
└── recordings/             training only: *.mp4, *.mkv
```

Only `config/`, `refs/` and `models/` are needed to run `babywatch live`.

**`scenes.yaml`.** The polygon is in 960×540 coordinates. It covers the crib and about one standing-baby height above the top rail. `ref` paths are relative to the data directory.

```yaml
scenes:
  - name: pos_a_night
    ref: refs/20261020-210452.png
    polygon: [[290, 170], [655, 170], [655, 535], [290, 535]]
```

**Labels.**
- **File naming.** Recordings are named `<clip-id>_<anything>.mp4|mkv`. The clip ID (the part before the first `_`) is the `video` column in the CSVs.
- **`intervals.csv` labels:**
  - `lying`: baby lying in the crib
  - `upright`: sitting or standing in the crib
  - `empty`: no baby in the crib
  - `outside`: baby visible but out of the crib; trained as `empty`
  - `uncertain`: transitions, crawling, hand-offs; not used for training or scoring
- **`adult.csv`** lists the intervals where an adult is anywhere in the camera view. It is used only for scoring.

## 4. Installation

Requirements:
- Linux with an NVIDIA GPU and a driver that supports CUDA 12.8
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg`, used only by `babywatch review`

From the source directory:

```fish
uv build
uv tool install ./dist/babywatch-0.1.0-py3-none-any.whl --python 3.12 -c requirements.lock \
    --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```

- **What the install does.** It installs into `~/.local/share/uv/tools/babywatch` and puts `babywatch` in `~/.local/bin`.
- **Why `-c requirements.lock`.** It pins the tested versions (torch 2.11 + CUDA 12.8). Without it, the resolver can pick a torch build for CUDA 13.
- **Network.** The CUDA runtime wheels come from `pypi.nvidia.com`.

| Task | Command |
|---|---|
| Upgrade after a rebuild | `uv tool install --force ./dist/babywatch-0.1.0-py3-none-any.whl -c requirements.lock --index https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match` |
| Uninstall | `uv tool uninstall babywatch` (does not touch `~/.babywatch`) |
| Development | `uv sync` in the source directory, then `uv run babywatch …` |

## 5. Usage

```fish
babywatch            # list commands
babywatch <command> --help
```

### 5.1 Watch the camera

```fish
babywatch live --rtsp 'rtsp://USER:PASS@CAMERA/stream' --log ~/.babywatch/events.jsonl
```

It prints one JSON line on stdout, and appends it to `--log`, each time the notification state changes:

```json
{"ts": "2026-11-01T21:14:03", "notify": true,  "state": "upright", "adult": false, "p": 0.998}
{"ts": "2026-11-01T21:15:40", "notify": false, "state": "upright", "adult": true,  "p": 0.991}
```

| Field | Meaning |
|---|---|
| `ts` | Wall-clock time for RTSP; seconds into the file for `--file` |
| `notify` | **The value to act on.** `true` = baby upright in the crib and no adult in view |
| `state` | Smoothed classifier state: `upright` or `down` |
| `adult` | Adult seen within the last 5 s |
| `p` | P(upright) for this frame |

Options:

| Option | Effect |
|---|---|
| `--file PATH` | Read a recording instead of RTSP |
| `--realtime` | With `--file`, play at native speed |
| `--show` | Debug window: crib polygon, P(upright) bar, person boxes (red = adult). `q` closes it |
| `--verbose` | Print a line for every processed frame, not only changes |
| `--model`, `--scenes`, `--yolo` | Override the paths under `~/.babywatch` |

Logs (scene matches, reconnects, throughput) go to stderr.

**Sending notifications.** babywatch prints events; it does not deliver them. Pipe the output into whatever sends the message. For example, with [ntfy](https://ntfy.sh):

```fish
babywatch live --rtsp $CAM_URL | while read -l line
    if string match -q '*"notify": true*' -- $line
        curl -s -d "Baby is up" ntfy.sh/your-topic
    end
end
```

### 5.2 Add or fix a camera position

The camera position must be in `scenes.yaml`, or the crib will not be found. If no reference matches, `live` falls back to the first scene's polygon, logged as `ok=False`.

```fish
# grab a frame from the camera and click the polygon
babywatch set-roi --name pos_e_night --rtsp 'rtsp://USER:PASS@CAMERA/stream'
# or from a recording at t = 5 s
babywatch set-roi --name pos_e_night --video ~/.babywatch/recordings/clip.mp4 --t 5
# redraw the polygon of an existing scene
babywatch set-roi --name pos_a_night --ref ~/.babywatch/refs/20261020-210452.png
```

- Left click adds a point, right click undoes the last one, Enter saves, Esc cancels.
- Include about one standing-baby height above the top rail.
- Keep the polygon off the floor and sofa in front of the crib where possible.

Adding a scene does not require retraining. Retraining with footage from the new position improves accuracy there.

## 6. Retraining

Do this after recording new clips: new nights, a new camera position, situations the model gets wrong.

1. **Add recordings.** Copy the clips into `~/.babywatch/recordings/`.
2. **Label postures.**
   - Run `babywatch sheets` to create contact sheets of ROI crops with timestamps in `labels/sheets/`. Use `--fps 1 --clahe` for night footage.
   - Add rows to `labels/intervals.csv`.
3. **Label adults.** Add rows to `labels/adult.csv` for every interval where an adult is anywhere in the view.
4. **Check the labels.** Run `babywatch review --pred ~/.babywatch/cache/preds.npz` to render `labels/review/<clip>.mp4`. It shows the label, a timeline bar, and the latest cross-validated P(upright). Fix the CSV and repeat.
5. **Cache features.**
   - `babywatch cache-embeddings` embeds ROI crops at 5 fps with each backbone into `cache/<backbone>@<size>/`, and caches YOLO features for the baseline.
   - `babywatch cache-persons` stores full-frame person boxes for the adult gate in `cache/persons/`.
6. **Train.** Run `babywatch train`.

`babywatch train`:
- runs leave-one-video-out cross-validation over backbone × head (`logreg`, `mlp`) × pooling window (1, 5, 10 samples), with and without the `empty` class (`--train-empty`), plus the YOLO baselines
- picks the pipeline with the best mean per-fold F1, then tunes the hysteresis on its held-out predictions
- refits on all clips and writes `models/upright.joblib`
- writes the held-out predictions to `cache/preds.npz`

Useful options:

| Option | Effect |
|---|---|
| `--configs dinov2_b@224 siglip2_b@224` | Backbones to compare. Must be cached first |
| `--heads mlp --pools 5 --train-empty yes` | Fix the pipeline; only the hysteresis is tuned |
| `--exclude CLIP_ID` | Leave a clip out entirely, to test on it with `live --file --model …` |
| `--out PATH` | Write the model elsewhere |
| `--guard 0.5` | Seconds dropped around label changes for training |

`live` reads the model on start, so restart it after retraining.

## 7. Evaluation and current results

The cross-validation is leave-one-video-out: each fold trains on all but one clip and scores on the unseen clip. This measures how well the model handles a new night or camera position, not neighbouring frames of the same clip.

Metrics:
- **Frame level.** Precision, recall, F1 and PR-AUC of P(upright) > 0.5 on frames labeled `lying`, `upright` or `empty`.
- **Event level.** Applies hysteresis and the adult gate. Ground truth "should notify" = `upright` and no adult in `adult.csv`.
  - F1 is computed over frames.
  - Onset latency is measured from the start of each should-notify episode.
  - A false alarm is a notification episode that touches only should-not-notify frames.
- **Targets.** Event F1 ≥ 0.95, no missed episodes, median latency ≤ 3 s.

Current model (6 clips, 11 should-notify episodes): DINOv2-B @224, 1 s window, MLP, trained with the `empty` class.

| Metric | Value |
|---|---|
| Event F1 / precision / recall | 0.966 / 0.970 / 0.962 |
| Missed episodes | 0 / 11 |
| False alarms per hour | 0 |
| Onset latency, median / max | 1.4 s / 3.4 s |
| Best YOLO-only baseline, event F1 | 0.63 (4 of 11 missed) |

Training data:

| Clip ID | Length | Camera position | Light | Content |
|---|---|---|---|---|
| 20261020-210452 | 5:40 | pos_a | IR night | lying, several sit/stand episodes |
| 20261025-091538 | 10:00 | pos_b | day | standing/sitting with adult on the sofa, adult picks baby up, empty crib with adult around, door closes |
| 20261026-195104 | 2:03 | pos_c | IR night | lying → standing |
| 20261026-200106 | 2:00 | pos_c | IR night | sitting → lying |
| 20261029-105212 | 9:57 | pos_d | day | lying → standing, adult picks baby up, baby with adult outside the crib, door closes |
| 20261029-143547 | 2:34 | pos_d | day | lying → sitting |

## 8. Known limitations

- **Small dataset.**
  - The numbers above come from about 32 minutes of video from 6 recordings.
  - The weakest held-out clip is the IR night clip 20261020-210452 (frame F1 0.87).
  - More night recordings will help the most.
- **No adult footage at night.** The adult gate has only been tested in daylight. YOLO is less reliable on grainy IR.
- **A blocked view means no notification.** If a door or object covers the crib, the classifier reports `empty`, so a baby standing behind it is missed. A separate "camera blocked" state would need its own detection.
- **A person standing still for a long time inside the crib area** could be taken for the baby. In practice an adult leaning in is taller than the crib area and is caught by the height rule.
- **Video glitches.** Corrupted HEVC frames (grey or colour blocks lasting a few seconds) produce random predictions. The 1 s hysteresis absorbs short glitches, not long ones.
- **The hysteresis thresholds were tuned on the same held-out predictions they are scored on.** Expect slightly worse numbers on new footage.
- **One camera.** `live` handles a single stream. Run one process per camera.
