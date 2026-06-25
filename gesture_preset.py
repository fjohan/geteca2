#!/usr/bin/env python3
"""Run gesture_text_candidates.py with coarse candidate-volume presets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PRESETS = {
    "high": {
        "description": "Permissive settings intended to return many candidates.",
        "args": [
            "--merge-gap",
            "-1",
            "--gesture-threshold",
            "0.01",
            "--min-face-visible",
            "0.20",
            "--min-hand-visible",
            "0.0",
            "--min-hand-size",
            "0",
            "--min-finger-score",
            "0.0",
            "--reason-threshold",
            "0.40",
            "--max-duration",
            "20",
            "--top-k",
            "100",
        ],
    },
    "mid": {
        "description": "Balanced settings intended to return a medium-sized review set.",
        "args": [
            "--merge-gap",
            "-1",
            "--gesture-threshold",
            "auto",
            "--min-face-visible",
            "0.50",
            "--min-hand-visible",
            "0.10",
            "--min-hand-size",
            "35",
            "--min-finger-score",
            "0.10",
            "--reason-threshold",
            "0.55",
            "--max-duration",
            "12",
            "--top-k",
            "50",
        ],
    },
    "low": {
        "description": "Strict settings intended to return fewer, cleaner candidates.",
        "args": [
            "--merge-gap",
            "-1",
            "--gesture-threshold",
            "auto",
            "--min-face-visible",
            "0.80",
            "--min-hand-visible",
            "0.45",
            "--min-hand-size",
            "65",
            "--min-finger-score",
            "0.35",
            "--reason-threshold",
            "0.70",
            "--max-duration",
            "10",
            "--top-k",
            "25",
        ],
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=sorted(PRESETS), help="Candidate-volume preset.")
    parser.add_argument("--url", help="YouTube URL to pass through.")
    parser.add_argument("--video", help="Local video path to pass through.")
    parser.add_argument("--subtitles", help="Local subtitle path to pass through.")
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    parser.add_argument("--lang", default="en", help="Transcript/subtitle language, currently en or sv for text reasons.")
    parser.add_argument("--sample-fps", default="2")
    parser.add_argument("--make-clips", default="true")
    parser.add_argument("--overlay-stickman", default="false")
    parser.add_argument("--timeline-chart", default="true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args, extra_args = build_parser().parse_known_args()
    if not args.url and not args.video:
        raise SystemExit("Provide --url or --video.")
    if args.video and not args.subtitles:
        raise SystemExit("Local video mode requires --subtitles.")

    script = Path(__file__).with_name("gesture_text_candidates.py")
    out_dir = args.out or Path(f"./run_{Path(args.video).stem if args.video else 'youtube'}_{args.preset}")
    cmd = [sys.executable, str(script)]
    if args.url:
        cmd += ["--url", args.url]
    if args.video:
        cmd += ["--video", args.video]
    if args.subtitles:
        cmd += ["--subtitles", args.subtitles]
    cmd += [
        "--out",
        str(out_dir),
        "--lang",
        args.lang,
        "--sample-fps",
        args.sample_fps,
        "--make-clips",
        args.make_clips,
        "--overlay-stickman",
        args.overlay_stickman,
        "--timeline-chart",
        args.timeline_chart,
    ]
    cmd += PRESETS[args.preset]["args"]
    if args.force:
        cmd.append("--force")
    if extra_args:
        extra = extra_args[1:] if extra_args and extra_args[0] == "--" else extra_args
        cmd += extra

    print(f"Preset: {args.preset} - {PRESETS[args.preset]['description']}", file=sys.stderr)
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
