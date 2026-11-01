"""`babywatch <command> [args]` dispatcher. Run a command with --help for its options.

Data (config, refs, models, labels, cache, recordings) lives in $BABYWATCH_HOME, default ~/.babywatch.
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "live": ("live", "watch an RTSP stream or a file and print notification changes"),
    "set-roi": ("set_roi", "click the crib polygon for a camera position"),
    "sheets": ("make_sheets", "render ROI contact sheets for labeling"),
    "review": ("review_labels", "render clips with labels (and predictions) burnt in"),
    "cache-embeddings": ("cache_embeddings", "embed ROI crops of all clips"),
    "cache-persons": ("cache_persons", "YOLO person boxes on full frames of all clips"),
    "train": ("train_eval", "leave-one-video-out CV, then fit the model"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: babywatch <command> [args]\n\ncommands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:18s} {desc}")
        sys.exit(0 if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help") else 2)
    cmd = sys.argv.pop(1)
    sys.argv[0] = f"babywatch {cmd}"
    importlib.import_module(f"babywatch.cli.{COMMANDS[cmd][0]}").main()


if __name__ == "__main__":
    main()
