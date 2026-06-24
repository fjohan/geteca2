# Gesture/Text Candidate Finder

This CLI finds timestamp ranges where transcript text and visible hand or arm gestures co-occur. It is intended for human review, not definitive gesture annotation.

## Requirements

Install Python dependencies in this virtual environment:

```bash
bin/python -m pip install -r requirements.txt
```

The CLI also requires these executables on `PATH`:

```bash
ffmpeg -version
ffprobe -version
yt-dlp --version
```

## YouTube Usage

```bash
bin/python gesture_text_candidates.py \
  --url "https://www.youtube.com/watch?v=..." \
  --out ./corpus_run \
  --make-clips true \
  --top-k 30
```

The tool downloads the video, subtitles, and metadata with `yt-dlp`, analyzes sampled video frames with MediaPipe Holistic, aligns gesture activity with subtitle windows, and writes review outputs.

## Local Test Mode

```bash
bin/python gesture_text_candidates.py \
  --video local.mp4 \
  --subtitles local.vtt \
  --out test_out
```

This mode avoids YouTube and is useful for quick local checks.

## Outputs

The output directory contains:

- `candidates.csv`
- `candidates.json`
- `text_segments.csv`
- `frame_features.csv`
- `index.html`
- `clips/` when `--make-clips true`

Candidate columns include rank, timestamp range, text, gesture score statistics, hand visibility percentages, and clip path when generated.

## Options

```text
--url URL
--video local.mp4
--subtitles local.vtt
--out DIR
--lang en
--whisper false
--min-duration 1.0
--max-duration 12.0
--pre-roll 0.5
--post-roll 0.75
--gesture-threshold auto
--top-k 50
--make-clips false
--clip-format mp4
--keep-video false
--sample-fps 10
--merge-gap 0.75
--force
```

`--gesture-threshold auto` uses the 80th percentile of nonzero smoothed gesture scores. You can also pass a numeric threshold such as `0.35`.

`--merge-gap -1` preserves the original VTT/SRT cue segmentation. This is useful for Whisper output that already has review-sized segments.

## Whisper

If a video has no subtitles, `--whisper true` attempts to use the optional `openai-whisper` package. Install it separately if needed:

```bash
bin/python -m pip install openai-whisper
```
