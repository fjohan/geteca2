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

## Presets

`gesture_preset.py` wraps the main CLI with coarse candidate-volume presets:

- `high`: permissive, likely many candidates.
- `mid`: balanced, likely a medium review set.
- `low`: strict, likely fewer cleaner candidates.

Example:

```bash
bin/python gesture_preset.py high \
  --video classic-carbonara.mp4 \
  --subtitles classic-carbonara.vtt \
  --lang en \
  --out ./run_carbonara_high
```

The wrapper defaults to `--sample-fps 2`, `--make-clips true`, and `--timeline-chart true`.

You can override preset values after `--`:

```bash
bin/python gesture_preset.py mid \
  --video classic-carbonara.mp4 \
  --subtitles classic-carbonara.vtt \
  --lang en \
  -- --top-k 20 --min-face-visible 0.3
```

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

The tool first samples video frames, detects face/pose/hand landmarks with MediaPipe, and writes per-frame features to `frame_features.csv`.

The original frame-level `gesture_score` is preserved. It is based on:

- normalized wrist speed
- normalized wrist acceleration
- visible hands or visible pose
- whether hands are above shoulders
- whether hands are extended far from the torso

The newer ranking model adds independent frame scores:

- `face_visibility_score`: face landmarks, estimated face size, and a rough frontal/three-quarter yaw score.
- `hand_visibility_score`: detected hand landmarks and estimated hand size.
- `finger_visibility_score`: hand size plus whether finger joint spacing is large enough to inspect articulation.
- `arm_motion_score`: wrist velocity, wrist acceleration, elbow angular velocity, and trajectory energy.
- `hand_motion_score`: palm orientation change, hand openness change, finger spread change, and pinch/open-close change.

MediaPipe Holistic's `solutions` API does not expose reliable per-landmark confidence for every face/hand point, so tracking confidence is approximated from landmark presence, size, and stability-derived motion signals.

Scores are smoothed over roughly `0.7s` so a single noisy frame has less influence.

For each subtitle window, the tool computes overlapping segment statistics:

- `mean_gesture_score`
- `max_gesture_score`
- `percent_frames_gesturing`
- `left_hand_visible_percent`
- `right_hand_visible_percent`
- `face_visibility`
- `hand_visibility`
- `finger_visibility`
- `arm_motion`
- `hand_motion`
- `speech_duration`
- `mean_hand_size`

Candidate score is currently:

```python
gesture_component = mean([mean_gesture_score, max_gesture_score, percent_frames_gesturing])

candidate_score = (
    gesture_component * gesture_weight
    + face_visibility * face_weight
    + hand_visibility * hand_weight
    + arm_motion * arm_weight
    + finger_visibility * finger_weight
) / total_weight
```

Default weights:

- `--gesture-weight 0.30`
- `--face-weight 0.25`
- `--hand-weight 0.20`
- `--arm-weight 0.15`
- `--finger-weight 0.10`

Candidate filters:

- `--min-face-visible 0.80`
- `--min-hand-visible 0.0`
- `--min-hand-size 0.0`
- `--min-finger-score 0.0`

The default face filter is intentionally strict: it prioritizes clips where the active speaker's face is clearly visible. Lower `--min-face-visible` when working with cooking overhead shots, instrument closeups, or other videos where hands matter more than face.

`--gesture-threshold` controls which frames count as gesturing for `percent_frames_gesturing`. It does not directly set a minimum `candidate_score`, and it does not override duration filters. `--gesture-threshold auto` uses the 80th percentile of nonzero smoothed gesture scores. A numeric value such as `0.01` makes the gesturing-frame test more permissive.

### Reason Labels

After numeric scoring, the tool runs a transparent heuristic reason-label layer. This does not replace the numeric scores; it explains why a candidate may be interesting.

Each candidate can include:

- `primary_reason`
- `reason_labels`
- `reason_confidences_json`
- `evidence_summary`
- structured `reason_confidences` and `reason_evidence` in `candidates.json`

Implemented labels:

- `FACE_VISIBLE`
- `HANDS_VISIBLE`
- `LARGE_ARM_MOVEMENT`
- `BEAT_GESTURE`
- `POINTING_LIKE`
- `OPEN_PALM`
- `FINGER_ARTICULATION`
- `TWO_HANDED_GESTURE`
- `HAND_NEAR_FACE`
- `DEICTIC_TEXT_WITH_GESTURE`
- `EMPHASIS_TEXT_WITH_GESTURE`
- `CONTRAST_TEXT_WITH_GESTURE`

Reason labels are controlled by:

- `--reason-threshold 0.6`
- `--enable-text-reasons true`
- `--enable-finger-reasons true`
- `--reason-labels all`
- `--top-reasons 5`

Text-based reason labels use `--lang`. English (`en`) and Swedish (`sv`) marker sets are currently supported. Visual reason labels are language-independent.

Examples:

```bash
--lang en
--lang sv
```

`index.html` shows reason badges, the primary reason, evidence summaries, a reason-label filter, and a sortable reason-confidence column.

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
--min-face-visible 0.80
--min-hand-visible 0.0
--min-hand-size 0.0
--min-finger-score 0.0
--gesture-weight 0.30
--face-weight 0.25
--hand-weight 0.20
--arm-weight 0.15
--finger-weight 0.10
--reason-threshold 0.6
--enable-text-reasons true
--enable-finger-reasons true
--reason-labels all
--top-reasons 5
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
