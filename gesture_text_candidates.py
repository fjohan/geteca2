#!/usr/bin/env python3
"""Find transcript ranges where speech/text and visible gestures co-occur."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import mediapipe as mp
import numpy as np


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
SUBTITLE_EXTS = {".vtt", ".srt"}


@dataclass
class TextSegment:
    start: float
    end: float
    text: str


@dataclass
class Candidate:
    rank: int
    video_id: str
    start: float
    end: float
    duration: float
    text: str
    candidate_score: float
    mean_gesture_score: float
    max_gesture_score: float
    percent_frames_gesturing: float
    left_hand_visible_percent: float
    right_hand_visible_percent: float
    clip_path: str = ""


def log(message: str) -> None:
    print(message, file=sys.stderr)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    log("+ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def require_executable(name: str) -> str:
    venv_path = Path(sys.executable).parent / name
    if venv_path.exists() and os.access(venv_path, os.X_OK):
        return str(venv_path)
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Required executable not found on PATH: {name}")
    return path


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_timecode(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        raise ValueError(f"Invalid subtitle timecode: {value}")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_caption_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\\.*?\}", "", text)
    text = text.replace("&nbsp;", " ")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_subtitles(path: Path) -> list[TextSegment]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\ufeff", "")
    raw = re.sub(r"^WEBVTT.*?(?:\n\n|\r\n\r\n)", "", raw, flags=re.DOTALL)
    segments: list[TextSegment] = []
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n"))
    time_re = re.compile(
        r"(?P<start>\d{1,2}:\d{2}(?::\d{2})?[\.,]\d{3})\s+-->\s+"
        r"(?P<end>\d{1,2}:\d{2}(?::\d{2})?[\.,]\d{3})"
    )
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        match = time_re.search(lines[time_index])
        if not match:
            continue
        text = clean_caption_text(" ".join(lines[time_index + 1 :]))
        if text:
            segments.append(
                TextSegment(
                    start=parse_timecode(match.group("start")),
                    end=parse_timecode(match.group("end")),
                    text=text,
                )
            )
    return dedupe_subtitle_segments(segments)


def dedupe_subtitle_segments(segments: list[TextSegment]) -> list[TextSegment]:
    """Collapse common VTT auto-caption duplicates."""
    deduped: list[TextSegment] = []
    seen: set[tuple[float, float, str]] = set()
    for segment in segments:
        key = (round(segment.start, 2), round(segment.end, 2), segment.text)
        if key not in seen:
            seen.add(key)
            deduped.append(segment)
    return deduped


def merge_text_segments(segments: list[TextSegment], max_gap: float = 0.75) -> list[TextSegment]:
    if not segments:
        return []
    merged: list[TextSegment] = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if segment.start - previous.end <= max_gap:
            merged[-1] = TextSegment(previous.start, max(previous.end, segment.end), f"{previous.text} {segment.text}")
        else:
            merged.append(segment)
    return merged


def ffprobe_duration(video_path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ]
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def find_downloaded_files(download_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    video_files = sorted(p for p in download_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    subtitle_files = sorted(p for p in download_dir.iterdir() if p.suffix.lower() in SUBTITLE_EXTS)
    info_files = sorted(download_dir.glob("*.info.json"))
    return (
        max(video_files, key=lambda p: p.stat().st_size) if video_files else None,
        subtitle_files[0] if subtitle_files else None,
        info_files[0] if info_files else None,
    )


def download_youtube(url: str, out_dir: Path, lang: str, force: bool) -> tuple[Path, Path | None, str]:
    yt_dlp = require_executable("yt-dlp")
    download_dir = out_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    existing_video, existing_subtitle, existing_info = find_downloaded_files(download_dir)
    if existing_video and not force:
        return existing_video, existing_subtitle, load_video_id(existing_info, existing_video)

    template = str(download_dir / "%(id)s.%(ext)s")
    cmd = [
        yt_dlp,
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        lang,
        "--sub-format",
        "vtt/srt",
        "-f",
        "bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4][vcodec^=avc1]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        template,
        url,
    ]
    if force:
        cmd.insert(1, "--force-overwrites")
    run(cmd)
    video_path, subtitle_path, info_path = find_downloaded_files(download_dir)
    if not video_path:
        raise SystemExit("yt-dlp completed but no downloaded video file was found.")
    return video_path, subtitle_path, load_video_id(info_path, video_path)


def load_video_id(info_path: Path | None, video_path: Path) -> str:
    if info_path and info_path.exists():
        try:
            data = json.loads(info_path.read_text(encoding="utf-8"))
            return str(data.get("id") or video_path.stem)
        except json.JSONDecodeError:
            pass
    return video_path.stem.split(".")[0]


def maybe_transcribe_with_whisper(video_path: Path, out_dir: Path, lang: str) -> Path | None:
    try:
        import whisper  # type: ignore
    except ImportError:
        log("Whisper mode requested, but the 'whisper' package is not installed. Continuing without subtitles.")
        return None
    model_name = os.environ.get("WHISPER_MODEL", "base")
    log(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    result = model.transcribe(str(video_path), language=lang)
    subtitle_path = out_dir / "whisper_segments.srt"
    with subtitle_path.open("w", encoding="utf-8") as handle:
        for i, segment in enumerate(result.get("segments", []), start=1):
            handle.write(f"{i}\n")
            handle.write(f"{format_srt_time(segment['start'])} --> {format_srt_time(segment['end'])}\n")
            handle.write(clean_caption_text(segment.get("text", "")) + "\n\n")
    return subtitle_path


def format_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def point(landmarks, idx: int) -> tuple[float, float, float] | None:
    if not landmarks:
        return None
    lm = landmarks.landmark[idx]
    return (float(lm.x), float(lm.y), float(getattr(lm, "visibility", 1.0)))


def distance(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def elbow_angle(
    shoulder: tuple[float, float, float] | None,
    elbow: tuple[float, float, float] | None,
    wrist: tuple[float, float, float] | None,
) -> float:
    if shoulder is None or elbow is None or wrist is None:
        return 0.0
    a = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
    b = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def analyze_gestures(video_path: Path, out_dir: Path, sample_fps: float, force: bool) -> list[dict[str, float]]:
    features_path = out_dir / "frame_features.csv"
    if features_path.exists() and not force:
        return read_frame_features(features_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(native_fps / sample_fps)))
    pose = mp.solutions.pose.PoseLandmark
    rows: list[dict[str, float]] = []
    previous_wrists: dict[str, tuple[float, float, float] | None] = {"left": None, "right": None}
    previous_speeds: dict[str, float] = {"left": 0.0, "right": 0.0}
    frame_idx = 0

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            timestamp = frame_idx / native_fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            try:
                result = holistic.process(rgb)
            except Exception as exc:  # MediaPipe can fail on corrupt frames.
                log(f"MediaPipe skipped frame at {timestamp:.2f}s: {exc}")
                frame_idx += 1
                continue

            pose_landmarks = result.pose_landmarks
            left_hand_visible = 1.0 if result.left_hand_landmarks else 0.0
            right_hand_visible = 1.0 if result.right_hand_landmarks else 0.0
            left_shoulder = point(pose_landmarks, pose.LEFT_SHOULDER.value)
            right_shoulder = point(pose_landmarks, pose.RIGHT_SHOULDER.value)
            left_elbow = point(pose_landmarks, pose.LEFT_ELBOW.value)
            right_elbow = point(pose_landmarks, pose.RIGHT_ELBOW.value)
            left_wrist = point(pose_landmarks, pose.LEFT_WRIST.value)
            right_wrist = point(pose_landmarks, pose.RIGHT_WRIST.value)
            torso_size = max(distance(left_shoulder, right_shoulder), 0.08)
            row = frame_feature_row(
                timestamp,
                pose_landmarks,
                left_hand_visible,
                right_hand_visible,
                left_shoulder,
                right_shoulder,
                left_elbow,
                right_elbow,
                left_wrist,
                right_wrist,
                torso_size,
                previous_wrists,
                previous_speeds,
                1.0 / max(sample_fps, 0.1),
            )
            rows.append(row)
            previous_wrists["left"] = left_wrist
            previous_wrists["right"] = right_wrist
            previous_speeds["left"] = row["left_wrist_speed"]
            previous_speeds["right"] = row["right_wrist_speed"]
            frame_idx += 1
    cap.release()
    smooth_scores(rows, max(1, int(round(sample_fps * 0.7))))
    write_frame_features(features_path, rows)
    return rows


def frame_feature_row(
    timestamp: float,
    pose_landmarks,
    left_hand_visible: float,
    right_hand_visible: float,
    left_shoulder,
    right_shoulder,
    left_elbow,
    right_elbow,
    left_wrist,
    right_wrist,
    torso_size: float,
    previous_wrists: dict[str, tuple[float, float, float] | None],
    previous_speeds: dict[str, float],
    dt: float,
) -> dict[str, float]:
    pose_visible = 1.0 if pose_landmarks else 0.0
    left_speed = distance(left_wrist, previous_wrists["left"]) / max(dt * torso_size, 0.001)
    right_speed = distance(right_wrist, previous_wrists["right"]) / max(dt * torso_size, 0.001)
    left_accel = abs(left_speed - previous_speeds["left"]) / max(dt, 0.001)
    right_accel = abs(right_speed - previous_speeds["right"]) / max(dt, 0.001)
    left_above = 1.0 if left_wrist and left_shoulder and left_wrist[1] < left_shoulder[1] else 0.0
    right_above = 1.0 if right_wrist and right_shoulder and right_wrist[1] < right_shoulder[1] else 0.0
    left_far = 1.0 if distance(left_wrist, left_shoulder) / torso_size > 1.1 else 0.0
    right_far = 1.0 if distance(right_wrist, right_shoulder) / torso_size > 1.1 else 0.0
    motion_energy = min(1.0, (left_speed + right_speed) / 8.0) + min(1.0, (left_accel + right_accel) / 60.0) * 0.4
    visibility = max(left_hand_visible, right_hand_visible, pose_visible * 0.5)
    posture = max(left_above, right_above) * 0.2 + max(left_far, right_far) * 0.2
    raw_score = min(1.0, motion_energy * 0.65 + visibility * 0.25 + posture)
    return {
        "timestamp": timestamp,
        "pose_visible": pose_visible,
        "left_hand_visible": left_hand_visible,
        "right_hand_visible": right_hand_visible,
        "left_wrist_x": left_wrist[0] if left_wrist else 0.0,
        "left_wrist_y": left_wrist[1] if left_wrist else 0.0,
        "right_wrist_x": right_wrist[0] if right_wrist else 0.0,
        "right_wrist_y": right_wrist[1] if right_wrist else 0.0,
        "left_elbow_x": left_elbow[0] if left_elbow else 0.0,
        "left_elbow_y": left_elbow[1] if left_elbow else 0.0,
        "right_elbow_x": right_elbow[0] if right_elbow else 0.0,
        "right_elbow_y": right_elbow[1] if right_elbow else 0.0,
        "left_shoulder_x": left_shoulder[0] if left_shoulder else 0.0,
        "left_shoulder_y": left_shoulder[1] if left_shoulder else 0.0,
        "right_shoulder_x": right_shoulder[0] if right_shoulder else 0.0,
        "right_shoulder_y": right_shoulder[1] if right_shoulder else 0.0,
        "left_wrist_speed": left_speed,
        "right_wrist_speed": right_speed,
        "left_wrist_acceleration": left_accel,
        "right_wrist_acceleration": right_accel,
        "hand_above_shoulder_left": left_above,
        "hand_above_shoulder_right": right_above,
        "hand_far_from_torso_left": left_far,
        "hand_far_from_torso_right": right_far,
        "elbow_angle_left": elbow_angle(left_shoulder, left_elbow, left_wrist),
        "elbow_angle_right": elbow_angle(right_shoulder, right_elbow, right_wrist),
        "hand_motion_energy": motion_energy,
        "gesture_score_raw": raw_score,
        "gesture_score": raw_score,
    }


def smooth_scores(rows: list[dict[str, float]], window: int) -> None:
    if not rows:
        return
    scores = np.array([row["gesture_score_raw"] for row in rows], dtype=float)
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(scores, kernel, mode="same")
    for row, score in zip(rows, smoothed):
        row["gesture_score"] = float(score)


def read_frame_features(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def write_frame_features(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gesture_threshold(rows: list[dict[str, float]], threshold: str) -> float:
    scores = np.array([row["gesture_score"] for row in rows if row["gesture_score"] > 0], dtype=float)
    if not len(scores):
        return 1.0
    if threshold == "auto":
        return float(np.percentile(scores, 80))
    return float(threshold)


def build_candidates(
    text_segments: list[TextSegment],
    frame_rows: list[dict[str, float]],
    video_id: str,
    min_duration: float,
    max_duration: float,
    threshold_arg: str,
    top_k: int,
) -> list[Candidate]:
    if not text_segments or not frame_rows:
        return []
    threshold = gesture_threshold(frame_rows, threshold_arg)
    candidates: list[Candidate] = []
    for segment in text_segments:
        duration = segment.end - segment.start
        if duration < min_duration or duration > max_duration:
            continue
        overlapping = [row for row in frame_rows if segment.start <= row["timestamp"] <= segment.end]
        if not overlapping:
            continue
        scores = np.array([row["gesture_score"] for row in overlapping], dtype=float)
        gesturing = scores >= threshold
        left_visible = np.array([row["left_hand_visible"] for row in overlapping], dtype=float)
        right_visible = np.array([row["right_hand_visible"] for row in overlapping], dtype=float)
        pose_visible = np.array([row["pose_visible"] for row in overlapping], dtype=float)
        visible_percent = float(max(left_visible.mean(), right_visible.mean(), pose_visible.mean() * 0.5))
        percent_gesturing = float(gesturing.mean())
        if visible_percent <= 0 or percent_gesturing <= 0:
            continue
        token_count = max(1, len(segment.text.split()))
        text_density = min(1.0, token_count / max(duration, 0.1) / 4.0)
        duration_factor = min(1.0, duration / max_duration)
        mean_score = float(scores.mean())
        max_score = float(scores.max())
        candidate_score = (
            max_score * 0.35
            + mean_score * 0.25
            + percent_gesturing * 0.2
            + visible_percent * 0.1
            + text_density * 0.05
            + duration_factor * 0.05
        )
        candidates.append(
            Candidate(
                rank=0,
                video_id=video_id,
                start=segment.start,
                end=segment.end,
                duration=duration,
                text=segment.text,
                candidate_score=candidate_score,
                mean_gesture_score=mean_score,
                max_gesture_score=max_score,
                percent_frames_gesturing=percent_gesturing,
                left_hand_visible_percent=float(left_visible.mean()),
                right_hand_visible_percent=float(right_visible.mean()),
            )
        )
    candidates.sort(key=lambda item: item.candidate_score, reverse=True)
    for index, candidate in enumerate(candidates[:top_k], start=1):
        candidate.rank = index
    return candidates[:top_k]


def write_text_segments(out_dir: Path, segments: list[TextSegment]) -> None:
    with (out_dir / "text_segments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_time", "end_time", "text", "token_count", "speech_rate_proxy"])
        writer.writeheader()
        for segment in segments:
            duration = max(segment.end - segment.start, 0.1)
            token_count = len(segment.text.split())
            writer.writerow(
                {
                    "start_time": f"{segment.start:.3f}",
                    "end_time": f"{segment.end:.3f}",
                    "text": segment.text,
                    "token_count": token_count,
                    "speech_rate_proxy": f"{token_count / duration:.3f}",
                }
            )


def candidate_as_dict(candidate: Candidate) -> dict[str, str | int | float]:
    return {
        "rank": candidate.rank,
        "video_id": candidate.video_id,
        "start": round(candidate.start, 3),
        "end": round(candidate.end, 3),
        "duration": round(candidate.duration, 3),
        "text": candidate.text,
        "candidate_score": round(candidate.candidate_score, 6),
        "mean_gesture_score": round(candidate.mean_gesture_score, 6),
        "max_gesture_score": round(candidate.max_gesture_score, 6),
        "percent_frames_gesturing": round(candidate.percent_frames_gesturing, 6),
        "left_hand_visible_percent": round(candidate.left_hand_visible_percent, 6),
        "right_hand_visible_percent": round(candidate.right_hand_visible_percent, 6),
        "clip_path": candidate.clip_path,
    }


def write_candidates(out_dir: Path, candidates: list[Candidate]) -> None:
    rows = [candidate_as_dict(candidate) for candidate in candidates]
    fieldnames = [
        "rank",
        "video_id",
        "start",
        "end",
        "duration",
        "text",
        "candidate_score",
        "mean_gesture_score",
        "max_gesture_score",
        "percent_frames_gesturing",
        "left_hand_visible_percent",
        "right_hand_visible_percent",
        "clip_path",
    ]
    with (out_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def make_clips(
    video_path: Path,
    out_dir: Path,
    candidates: list[Candidate],
    pre_roll: float,
    post_roll: float,
    clip_format: str,
    overlay_stickman: bool,
) -> None:
    require_executable("ffmpeg")
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    duration = ffprobe_duration(video_path)
    for candidate in candidates:
        start = max(0.0, candidate.start - pre_roll)
        end = candidate.end + post_roll
        if duration:
            end = min(duration, end)
        clip_path = clips_dir / f"rank_{candidate.rank:03d}_{start:.2f}_{end:.2f}.{clip_format}"
        if overlay_stickman:
            render_stickman_clip(video_path, clip_path, start, end)
            candidate.clip_path = str(clip_path.relative_to(out_dir))
            continue
        stream_copy = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(video_path),
            "-c",
            "copy",
            str(clip_path),
        ]
        result = run(stream_copy, check=False)
        if result.returncode != 0 or not clip_path.exists() or clip_path.stat().st_size == 0:
            reencode = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-to",
                f"{end:.3f}",
                "-i",
                str(video_path),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(clip_path),
            ]
            run(reencode)
        candidate.clip_path = str(clip_path.relative_to(out_dir))


def render_stickman_clip(video_path: Path, clip_path: Path, start: float, end: float) -> None:
    temp_video = clip_path.with_name(clip_path.stem + ".overlay_tmp.mp4")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video for overlay rendering: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Could not create overlay clip: {temp_video}")

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp > end:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = holistic.process(rgb)
            draw_stickman_overlay(frame, result)
            writer.write(frame)
    writer.release()
    cap.release()

    mux = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(video_path),
        "-i",
        str(temp_video),
        "-map",
        "1:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(clip_path),
    ]
    run(mux)
    try:
        temp_video.unlink()
    except OSError:
        pass


def draw_stickman_overlay(frame, result) -> None:
    height, width = frame.shape[:2]
    pose_landmarks = result.pose_landmarks
    drawing = mp.solutions.drawing_utils
    hands = mp.solutions.hands
    style = drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
    hand_style = drawing.DrawingSpec(color=(0, 180, 255), thickness=1, circle_radius=1)

    if pose_landmarks:
        pose = mp.solutions.pose.PoseLandmark
        arm_connections = [
            (pose.LEFT_SHOULDER.value, pose.LEFT_ELBOW.value),
            (pose.LEFT_ELBOW.value, pose.LEFT_WRIST.value),
            (pose.RIGHT_SHOULDER.value, pose.RIGHT_ELBOW.value),
            (pose.RIGHT_ELBOW.value, pose.RIGHT_WRIST.value),
            (pose.LEFT_SHOULDER.value, pose.RIGHT_SHOULDER.value),
        ]
        for start_idx, end_idx in arm_connections:
            draw_pose_connection(frame, pose_landmarks, start_idx, end_idx, width, height)
    if result.left_hand_landmarks:
        drawing.draw_landmarks(frame, result.left_hand_landmarks, hands.HAND_CONNECTIONS, hand_style, hand_style)
    if result.right_hand_landmarks:
        drawing.draw_landmarks(frame, result.right_hand_landmarks, hands.HAND_CONNECTIONS, hand_style, hand_style)


def draw_pose_connection(frame, landmarks, start_idx: int, end_idx: int, width: int, height: int) -> None:
    a = landmarks.landmark[start_idx]
    b = landmarks.landmark[end_idx]
    if getattr(a, "visibility", 1.0) < 0.35 or getattr(b, "visibility", 1.0) < 0.35:
        return
    ax, ay = int(a.x * width), int(a.y * height)
    bx, by = int(b.x * width), int(b.y * height)
    cv2.line(frame, (ax, ay), (bx, by), (0, 255, 255), 3)
    cv2.circle(frame, (ax, ay), 5, (0, 180, 255), -1)
    cv2.circle(frame, (bx, by), 5, (0, 180, 255), -1)


def build_timeline_chart(candidates: list[Candidate], video_duration: float) -> str:
    if not candidates or video_duration <= 0:
        return ""
    bars = []
    max_score = max((candidate.candidate_score for candidate in candidates), default=1.0) or 1.0
    for candidate in candidates:
        left = max(0.0, min(100.0, candidate.start / video_duration * 100.0))
        width = max(0.35, min(3.0, candidate.duration / video_duration * 100.0))
        height = max(4.0, candidate.candidate_score / max_score * 100.0)
        href = candidate.clip_path or f"#candidate-{candidate.rank}"
        label = (
            f"Rank {candidate.rank}: {candidate.start:.2f}-{candidate.end:.2f}s, "
            f"score {candidate.candidate_score:.3f}"
        )
        bars.append(
            f'<a class="timeline-bar" href="{html.escape(href)}" title="{html.escape(label)}" '
            f'style="left: {left:.4f}%; width: {width:.4f}%; height: {height:.2f}%;">'
            f'<span>{candidate.rank}</span></a>'
        )
    return f"""
  <section class="timeline-section">
    <h2>Timeline</h2>
    <div class="timeline-chart" aria-label="Candidate scores over video time">
      {''.join(bars)}
    </div>
    <div class="timeline-axis">
      <span>0:00</span>
      <span>{format_mmss(video_duration)}</span>
    </div>
  </section>
"""


def format_mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def write_html_report(
    out_dir: Path,
    candidates: list[Candidate],
    video_duration: float = 0.0,
    timeline_chart: bool = False,
) -> None:
    rows = []
    for candidate in candidates:
        media = ""
        if candidate.clip_path:
            media = f'<video src="{html.escape(candidate.clip_path)}" controls width="360"></video>'
        rows.append(
            f'<tr id="candidate-{candidate.rank}">'
            f"<td>{candidate.rank}</td>"
            f"<td>{candidate.start:.2f}-{candidate.end:.2f}</td>"
            f"<td>{candidate.candidate_score:.3f}</td>"
            f"<td>{html.escape(candidate.text)}</td>"
            f"<td>{media}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Gesture/Text Candidates</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 0 0 16px; }}
    .timeline-section {{ margin: 24px 0; }}
    .timeline-chart {{
      position: relative;
      height: 180px;
      border-left: 1px solid #bbb;
      border-bottom: 1px solid #bbb;
      background: linear-gradient(to top, #f5f5f5 1px, transparent 1px) 0 0 / 100% 25%;
    }}
    .timeline-bar {{
      position: absolute;
      bottom: 0;
      min-width: 5px;
      display: block;
      background: #1d75b9;
      border-radius: 3px 3px 0 0;
      opacity: 0.85;
      text-decoration: none;
    }}
    .timeline-bar:hover, .timeline-bar:focus {{ opacity: 1; background: #d35400; }}
    .timeline-bar span {{
      position: absolute;
      left: 50%;
      bottom: calc(100% + 4px);
      transform: translateX(-50%);
      color: #333;
      font-size: 11px;
      line-height: 1;
      white-space: nowrap;
    }}
    .timeline-axis {{ display: flex; justify-content: space-between; color: #666; font-size: 12px; margin-top: 6px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Gesture/Text Candidates</h1>
  {build_timeline_chart(candidates, video_duration) if timeline_chart else ""}
  <table>
    <thead><tr><th>Rank</th><th>Time</th><th>Score</th><th>Text</th><th>Clip</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    (out_dir / "index.html").write_text(document, encoding="utf-8")


def copy_or_keep_video(video_path: Path, out_dir: Path, keep_video: bool) -> None:
    if keep_video:
        return
    downloads = out_dir / "downloads"
    try:
        if downloads in video_path.parents and video_path.exists():
            video_path.unlink()
    except OSError as exc:
        log(f"Could not remove downloaded video: {exc}")


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, str]:
    if args.video:
        video_path = Path(args.video).expanduser().resolve()
        if not video_path.exists():
            raise SystemExit(f"Local video not found: {video_path}")
        subtitle_path = Path(args.subtitles).expanduser().resolve() if args.subtitles else None
        if subtitle_path and not subtitle_path.exists():
            raise SystemExit(f"Subtitle file not found: {subtitle_path}")
        return video_path, subtitle_path, video_path.stem
    if not args.url:
        raise SystemExit("Provide --url YOUTUBE_URL or --video local.mp4")
    return download_youtube(args.url, args.out, args.lang, args.force)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="YouTube URL to process")
    parser.add_argument("--video", help="Local video path for test/offline mode")
    parser.add_argument("--subtitles", help="Local .vtt or .srt path for test/offline mode")
    parser.add_argument("--out", type=Path, default=Path("./output"))
    parser.add_argument("--lang", default="en")
    parser.add_argument("--whisper", type=parse_bool, default=False)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=12.0)
    parser.add_argument("--pre-roll", type=float, default=0.5)
    parser.add_argument("--post-roll", type=float, default=0.75)
    parser.add_argument("--gesture-threshold", default="auto")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--make-clips", type=parse_bool, default=False)
    parser.add_argument(
        "--overlay-stickman",
        type=parse_bool,
        default=False,
        help="Draw MediaPipe arm/hand stickman overlays on generated review clips.",
    )
    parser.add_argument(
        "--timeline-chart",
        type=parse_bool,
        default=False,
        help="Add a clickable score-over-time bar chart to index.html.",
    )
    parser.add_argument("--clip-format", default="mp4")
    parser.add_argument("--keep-video", type=parse_bool, default=False)
    parser.add_argument("--sample-fps", type=float, default=10.0)
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.75,
        help="Merge adjacent subtitle cues with gaps up to this many seconds. Use -1 to keep original cue segmentation.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download/reprocess cached intermediates")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    require_executable("ffmpeg")
    require_executable("ffprobe")
    video_path, subtitle_path, video_id = resolve_inputs(args)
    if not subtitle_path and args.whisper:
        subtitle_path = maybe_transcribe_with_whisper(video_path, args.out, args.lang)
    if not subtitle_path:
        raise SystemExit("No subtitles found. Re-run with --subtitles local.vtt or --whisper true.")

    log(f"Parsing subtitles: {subtitle_path}")
    parsed_segments = parse_subtitles(subtitle_path)
    if args.merge_gap >= 0:
        text_segments = merge_text_segments(parsed_segments, max_gap=args.merge_gap)
    else:
        text_segments = parsed_segments
    write_text_segments(args.out, text_segments)
    log(f"Text windows: {len(text_segments)}")

    log(f"Analyzing gestures at {args.sample_fps:g} fps: {video_path}")
    frame_rows = analyze_gestures(video_path, args.out, args.sample_fps, args.force)
    log(f"Frame feature rows: {len(frame_rows)}")

    candidates = build_candidates(
        text_segments,
        frame_rows,
        video_id,
        args.min_duration,
        args.max_duration,
        args.gesture_threshold,
        args.top_k,
    )
    if args.make_clips and candidates:
        make_clips(
            video_path,
            args.out,
            candidates,
            args.pre_roll,
            args.post_roll,
            args.clip_format,
            args.overlay_stickman,
        )
    write_candidates(args.out, candidates)
    report_duration = ffprobe_duration(video_path) if args.timeline_chart else 0.0
    write_html_report(args.out, candidates, report_duration, args.timeline_chart)
    copy_or_keep_video(video_path, args.out, args.keep_video)
    log(f"Wrote {len(candidates)} candidates to {args.out / 'candidates.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
