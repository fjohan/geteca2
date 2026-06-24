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
  --overlay-stickman true \
  --timeline-chart true \
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

When `--make-clips true --overlay-stickman true` is used, review clips are re-rendered with a MediaPipe arm/hand stickman overlay. The overlay draws shoulder-elbow-wrist lines plus detected hand landmarks. This affects only the review clips, not scoring or candidate detection.

When `--timeline-chart true` is used, `index.html` includes a whole-video timeline chart. Time runs left to right, each bar is a candidate, and bar height is the normalized candidate score. If clips were generated, clicking a bar opens the corresponding clip; otherwise it jumps to the candidate row in the table.

## Scoring

The tool first samples video frames, detects pose/hand landmarks with MediaPipe, and writes per-frame features to `frame_features.csv`.

Frame-level `gesture_score` is based on:

- normalized wrist speed
- normalized wrist acceleration
- visible hands or visible pose
- whether hands are above shoulders
- whether hands are extended far from the torso

Scores are smoothed over roughly `0.7s` so a single noisy frame has less influence.

For each subtitle window, the tool computes overlapping gesture statistics:

- `mean_gesture_score`
- `max_gesture_score`
- `percent_frames_gesturing`
- `left_hand_visible_percent`
- `right_hand_visible_percent`

Candidate score is currently:

```python
candidate_score = (
    max_gesture_score * 0.35
    + mean_gesture_score * 0.25
    + percent_frames_gesturing * 0.20
    + visible_percent * 0.10
    + text_density * 0.05
    + duration_factor * 0.05
)
```

`visible_percent` is the strongest of left-hand visibility, right-hand visibility, or partial pose visibility for the segment.

`text_density` is a capped speech-rate proxy based on words per second.

`duration_factor` mildly favors longer windows up to `--max-duration`.

`--gesture-threshold` controls which frames count as gesturing for `percent_frames_gesturing`. It does not directly set a minimum `candidate_score`, and it does not override duration filters. `--gesture-threshold auto` uses the 80th percentile of nonzero smoothed gesture scores. A numeric value such as `0.01` makes the gesturing-frame test more permissive.

By default, adjacent subtitle cues are merged when the gap is `<= --merge-gap`, default `0.75s`. For Whisper VTT/SRT files that already have good review-sized segments, use:

```bash
--merge-gap -1
```

This preserves the original VTT/SRT cue segmentation.

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
--overlay-stickman false
--timeline-chart false
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
