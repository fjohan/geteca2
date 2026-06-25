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
from dataclasses import dataclass, field
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
    face_visibility: float = 0.0
    hand_visibility: float = 0.0
    finger_visibility: float = 0.0
    arm_motion: float = 0.0
    hand_motion: float = 0.0
    speech_duration: float = 0.0
    mean_hand_size: float = 0.0
    primary_reason: str = ""
    reason_labels: list[str] = field(default_factory=list)
    reason_confidences: dict[str, float] = field(default_factory=dict)
    reason_evidence: dict[str, dict[str, float | str]] = field(default_factory=dict)
    evidence_summary: str = ""
    clip_path: str = ""


@dataclass
class RankingWeights:
    gesture: float = 0.30
    face: float = 0.25
    hand: float = 0.20
    arm: float = 0.15
    finger: float = 0.10


@dataclass
class CandidateFilters:
    min_face_visible: float = 0.80
    min_hand_visible: float = 0.0
    min_hand_size: float = 0.0
    min_finger_score: float = 0.0


@dataclass
class ReasonConfig:
    threshold: float = 0.60
    enable_text_reasons: bool = True
    enable_finger_reasons: bool = True
    labels: set[str] | None = None
    top_reasons: int = 5
    language: str = "en"


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


def parse_reason_labels(value: str) -> set[str] | None:
    normalized = value.strip()
    if not normalized or normalized.lower() == "all":
        return None
    return {item.strip().upper() for item in normalized.split(",") if item.strip()}


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
        cached_rows = read_frame_features(features_path)
        if frame_rows_have_scoring_columns(cached_rows):
            return cached_rows
        log("Cached frame_features.csv lacks new scoring columns; regenerating.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(native_fps / sample_fps)))
    pose = mp.solutions.pose.PoseLandmark
    rows: list[dict[str, float]] = []
    previous_wrists: dict[str, tuple[float, float, float] | None] = {"left": None, "right": None}
    previous_speeds: dict[str, float] = {"left": 0.0, "right": 0.0}
    previous_elbow_angles: dict[str, float] = {"left": 0.0, "right": 0.0}
    previous_hand_metrics: dict[str, dict[str, float] | None] = {"left": None, "right": None}
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
                frame.shape[1],
                frame.shape[0],
                result.face_landmarks,
                result.left_hand_landmarks,
                result.right_hand_landmarks,
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
                previous_elbow_angles,
                previous_hand_metrics,
                1.0 / max(sample_fps, 0.1),
            )
            rows.append(row)
            previous_wrists["left"] = left_wrist
            previous_wrists["right"] = right_wrist
            previous_speeds["left"] = row["left_wrist_speed"]
            previous_speeds["right"] = row["right_wrist_speed"]
            previous_elbow_angles["left"] = row["elbow_angle_left"]
            previous_elbow_angles["right"] = row["elbow_angle_right"]
            previous_hand_metrics["left"] = {
                "orientation": row["left_hand_orientation"],
                "openness": row["left_hand_openness"],
                "spread": row["left_finger_spread"],
                "pinch": row["left_pinch_distance"],
            }
            previous_hand_metrics["right"] = {
                "orientation": row["right_hand_orientation"],
                "openness": row["right_hand_openness"],
                "spread": row["right_finger_spread"],
                "pinch": row["right_pinch_distance"],
            }
            frame_idx += 1
    cap.release()
    smooth_scores(rows, max(1, int(round(sample_fps * 0.7))))
    write_frame_features(features_path, rows)
    return rows


def frame_feature_row(
    timestamp: float,
    frame_width: int,
    frame_height: int,
    face_landmarks,
    left_hand_landmarks,
    right_hand_landmarks,
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
    previous_elbow_angles: dict[str, float],
    previous_hand_metrics: dict[str, dict[str, float] | None],
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
    left_elbow_angle = elbow_angle(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = elbow_angle(right_shoulder, right_elbow, right_wrist)
    left_elbow_angular_velocity = abs(left_elbow_angle - previous_elbow_angles["left"]) / max(dt, 0.001)
    right_elbow_angular_velocity = abs(right_elbow_angle - previous_elbow_angles["right"]) / max(dt, 0.001)
    motion_energy = min(1.0, (left_speed + right_speed) / 8.0) + min(1.0, (left_accel + right_accel) / 60.0) * 0.4
    visibility = max(left_hand_visible, right_hand_visible, pose_visible * 0.5)
    posture = max(left_above, right_above) * 0.2 + max(left_far, right_far) * 0.2
    raw_score = min(1.0, motion_energy * 0.65 + visibility * 0.25 + posture)
    face_scores = compute_face_scores(face_landmarks, frame_width, frame_height)
    left_hand_scores = compute_hand_scores(left_hand_landmarks, frame_width, frame_height)
    right_hand_scores = compute_hand_scores(right_hand_landmarks, frame_width, frame_height)
    left_hand_motion = compute_hand_motion_score(left_hand_scores, previous_hand_metrics["left"], dt)
    right_hand_motion = compute_hand_motion_score(right_hand_scores, previous_hand_metrics["right"], dt)
    arm_motion_score = compute_arm_motion_score(
        left_speed,
        right_speed,
        left_accel,
        right_accel,
        left_elbow_angular_velocity,
        right_elbow_angular_velocity,
    )
    hand_visibility_score = max(left_hand_scores["visibility"], right_hand_scores["visibility"])
    finger_visibility_score = max(left_hand_scores["finger_visibility"], right_hand_scores["finger_visibility"])
    hand_motion_score = max(left_hand_motion, right_hand_motion)
    hand_size = max(left_hand_scores["size"], right_hand_scores["size"])
    return {
        "timestamp": timestamp,
        "pose_visible": pose_visible,
        "face_visibility_score": face_scores["visibility"],
        "face_landmark_percent": face_scores["landmark_percent"],
        "face_size": face_scores["size"],
        "face_yaw_score": face_scores["yaw_score"],
        "hand_visibility_score": hand_visibility_score,
        "finger_visibility_score": finger_visibility_score,
        "arm_motion_score": arm_motion_score,
        "hand_motion_score": hand_motion_score,
        "mean_hand_size": hand_size,
        "left_hand_size": left_hand_scores["size"],
        "right_hand_size": right_hand_scores["size"],
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
        "elbow_angle_left": left_elbow_angle,
        "elbow_angle_right": right_elbow_angle,
        "elbow_angular_velocity_left": left_elbow_angular_velocity,
        "elbow_angular_velocity_right": right_elbow_angular_velocity,
        "left_hand_orientation": left_hand_scores["orientation"],
        "right_hand_orientation": right_hand_scores["orientation"],
        "left_hand_openness": left_hand_scores["openness"],
        "right_hand_openness": right_hand_scores["openness"],
        "left_finger_spread": left_hand_scores["finger_spread"],
        "right_finger_spread": right_hand_scores["finger_spread"],
        "left_pinch_distance": left_hand_scores["pinch_distance"],
        "right_pinch_distance": right_hand_scores["pinch_distance"],
        "hand_motion_energy": motion_energy,
        "gesture_score_raw": raw_score,
        "gesture_score": raw_score,
    }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def landmark_bbox(landmarks) -> tuple[float, float, float, float] | None:
    if not landmarks:
        return None
    xs = [float(lm.x) for lm in landmarks.landmark]
    ys = [float(lm.y) for lm in landmarks.landmark]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def compute_face_scores(face_landmarks, frame_width: int, frame_height: int) -> dict[str, float]:
    if not face_landmarks:
        return {"visibility": 0.0, "landmark_percent": 0.0, "size": 0.0, "yaw_score": 0.0}
    landmark_count = len(face_landmarks.landmark)
    landmark_percent = clamp01(landmark_count / 468.0)
    bbox = landmark_bbox(face_landmarks)
    if not bbox:
        return {"visibility": 0.0, "landmark_percent": landmark_percent, "size": 0.0, "yaw_score": 0.0}
    min_x, min_y, max_x, max_y = bbox
    face_area_fraction = max(0.0, max_x - min_x) * max(0.0, max_y - min_y)
    size_score = clamp01(face_area_fraction / 0.08)
    yaw_score = estimate_face_yaw_score(face_landmarks)
    visibility = clamp01(0.45 * landmark_percent + 0.35 * size_score + 0.20 * yaw_score)
    return {
        "visibility": visibility,
        "landmark_percent": landmark_percent,
        "size": face_area_fraction * frame_width * frame_height,
        "yaw_score": yaw_score,
    }


def estimate_face_yaw_score(face_landmarks) -> float:
    try:
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        nose = face_landmarks.landmark[1]
    except IndexError:
        return 1.0
    eye_mid_x = (left_eye.x + right_eye.x) / 2.0
    eye_span = abs(right_eye.x - left_eye.x)
    if eye_span <= 0:
        return 1.0
    normalized_offset = abs(nose.x - eye_mid_x) / eye_span
    return clamp01(1.0 - normalized_offset / 0.55)


def compute_hand_scores(hand_landmarks, frame_width: int, frame_height: int) -> dict[str, float]:
    empty = {
        "visibility": 0.0,
        "finger_visibility": 0.0,
        "size": 0.0,
        "orientation": 0.0,
        "openness": 0.0,
        "finger_spread": 0.0,
        "pinch_distance": 0.0,
    }
    if not hand_landmarks:
        return empty
    landmark_count = len(hand_landmarks.landmark)
    landmark_percent = clamp01(landmark_count / 21.0)
    bbox = landmark_bbox(hand_landmarks)
    if not bbox:
        return empty
    min_x, min_y, max_x, max_y = bbox
    bbox_w = max(0.0, max_x - min_x) * frame_width
    bbox_h = max(0.0, max_y - min_y) * frame_height
    size_px = max(bbox_w, bbox_h)
    size_score = clamp01(size_px / 90.0)
    joint_distance_score = compute_finger_joint_distance_score(hand_landmarks, frame_width, frame_height)
    visibility = clamp01(0.55 * landmark_percent + 0.45 * size_score)
    finger_visibility = clamp01(0.45 * landmark_percent + 0.35 * size_score + 0.20 * joint_distance_score)
    return {
        "visibility": visibility,
        "finger_visibility": finger_visibility,
        "size": size_px,
        "orientation": compute_hand_orientation(hand_landmarks),
        "openness": compute_hand_openness(hand_landmarks),
        "finger_spread": compute_finger_spread(hand_landmarks),
        "pinch_distance": normalized_landmark_distance(hand_landmarks, 4, 8),
    }


def normalized_landmark_distance(landmarks, a_idx: int, b_idx: int) -> float:
    try:
        a = landmarks.landmark[a_idx]
        b = landmarks.landmark[b_idx]
    except IndexError:
        return 0.0
    return math.hypot(a.x - b.x, a.y - b.y)


def compute_finger_joint_distance_score(hand_landmarks, frame_width: int, frame_height: int) -> float:
    pairs = [
        (5, 6),
        (6, 7),
        (7, 8),
        (9, 10),
        (10, 11),
        (11, 12),
        (13, 14),
        (14, 15),
        (15, 16),
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    distances = []
    for a_idx, b_idx in pairs:
        try:
            a = hand_landmarks.landmark[a_idx]
            b = hand_landmarks.landmark[b_idx]
        except IndexError:
            continue
        distances.append(math.hypot((a.x - b.x) * frame_width, (a.y - b.y) * frame_height))
    if not distances:
        return 0.0
    return clamp01(float(np.mean(distances)) / 18.0)


def compute_hand_orientation(hand_landmarks) -> float:
    try:
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
    except IndexError:
        return 0.0
    return math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)


def compute_hand_openness(hand_landmarks) -> float:
    fingertips = [4, 8, 12, 16, 20]
    try:
        wrist = hand_landmarks.landmark[0]
    except IndexError:
        return 0.0
    distances = []
    for idx in fingertips:
        try:
            tip = hand_landmarks.landmark[idx]
        except IndexError:
            continue
        distances.append(math.hypot(tip.x - wrist.x, tip.y - wrist.y))
    return float(np.mean(distances)) if distances else 0.0


def compute_finger_spread(hand_landmarks) -> float:
    fingertip_pairs = [(4, 8), (8, 12), (12, 16), (16, 20)]
    distances = [normalized_landmark_distance(hand_landmarks, a_idx, b_idx) for a_idx, b_idx in fingertip_pairs]
    return float(np.mean(distances)) if distances else 0.0


def angle_delta(a: float, b: float) -> float:
    delta = abs(a - b)
    return min(delta, abs((2 * math.pi) - delta))


def compute_hand_motion_score(current: dict[str, float], previous: dict[str, float] | None, dt: float) -> float:
    if not previous or current["visibility"] <= 0:
        return 0.0
    orientation_change = angle_delta(current["orientation"], previous["orientation"]) / max(dt, 0.001)
    openness_change = abs(current["openness"] - previous["openness"]) / max(dt, 0.001)
    spread_change = abs(current["finger_spread"] - previous["spread"]) / max(dt, 0.001)
    pinch_change = abs(current["pinch_distance"] - previous["pinch"]) / max(dt, 0.001)
    return clamp01(
        min(1.0, orientation_change / 4.0) * 0.35
        + min(1.0, openness_change / 0.8) * 0.25
        + min(1.0, spread_change / 0.5) * 0.20
        + min(1.0, pinch_change / 0.5) * 0.20
    )


def compute_arm_motion_score(
    left_speed: float,
    right_speed: float,
    left_accel: float,
    right_accel: float,
    left_elbow_angular_velocity: float,
    right_elbow_angular_velocity: float,
) -> float:
    wrist_velocity = min(1.0, (left_speed + right_speed) / 8.0)
    wrist_acceleration = min(1.0, (left_accel + right_accel) / 60.0)
    elbow_velocity = min(1.0, (left_elbow_angular_velocity + right_elbow_angular_velocity) / 240.0)
    trajectory_energy = min(1.0, max(left_speed, right_speed) / 4.0)
    return clamp01(
        wrist_velocity * 0.35
        + wrist_acceleration * 0.25
        + elbow_velocity * 0.25
        + trajectory_energy * 0.15
    )


def smooth_scores(rows: list[dict[str, float]], window: int) -> None:
    if not rows:
        return
    smooth_columns = [
        ("gesture_score_raw", "gesture_score"),
        ("face_visibility_score", "face_visibility_score"),
        ("hand_visibility_score", "hand_visibility_score"),
        ("finger_visibility_score", "finger_visibility_score"),
        ("arm_motion_score", "arm_motion_score"),
        ("hand_motion_score", "hand_motion_score"),
    ]
    kernel = np.ones(window, dtype=float) / window
    for source, target in smooth_columns:
        scores = np.array([row[source] for row in rows], dtype=float)
        smoothed = np.convolve(scores, kernel, mode="same")
        for row, score in zip(rows, smoothed):
            row[target] = float(score)


def read_frame_features(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def frame_rows_have_scoring_columns(rows: list[dict[str, float]]) -> bool:
    if not rows:
        return False
    required = {
        "face_visibility_score",
        "hand_visibility_score",
        "finger_visibility_score",
        "arm_motion_score",
        "hand_motion_score",
        "mean_hand_size",
    }
    return required.issubset(rows[0].keys())


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


class ReasonDetector:
    name = "BASE"
    requires_text = False
    requires_finger = False

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        raise NotImplementedError

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        return {}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return f"{self.name}: confidence {confidence:.2f}."


class FaceVisibleDetector(ReasonDetector):
    name = "FACE_VISIBLE"

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        return candidate.face_visibility

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        return {"mean_face_visibility": candidate.face_visibility}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return f"Face visible for most of the segment; mean face score {confidence:.2f}."


class HandsVisibleDetector(ReasonDetector):
    name = "HANDS_VISIBLE"

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        return candidate.hand_visibility

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        return {
            "mean_hand_visibility": candidate.hand_visibility,
            "left_hand_visible_percent": candidate.left_hand_visible_percent,
            "right_hand_visible_percent": candidate.right_hand_visible_percent,
        }

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return (
            f"Hands visible; left {float(evidence['left_hand_visible_percent']):.0%}, "
            f"right {float(evidence['right_hand_visible_percent']):.0%} of frames."
        )


class LargeArmMovementDetector(ReasonDetector):
    name = "LARGE_ARM_MOVEMENT"

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        return clamp01(max(candidate.arm_motion, candidate.mean_gesture_score))

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        high_motion = frame_percent(frames, "arm_motion_score", 0.45)
        max_speed = max((row["left_wrist_speed"] + row["right_wrist_speed"] for row in frames), default=0.0)
        return {"high_motion_percent": high_motion, "max_combined_wrist_speed": max_speed}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return (
            f"High wrist/elbow movement over {float(evidence['high_motion_percent']):.0%} of frames; "
            f"max wrist speed {float(evidence['max_combined_wrist_speed']):.2f} torso-widths/s."
        )


class BeatGestureDetector(ReasonDetector):
    name = "BEAT_GESTURE"

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        peaks = motion_peak_count(frames)
        if candidate.duration <= 0:
            return 0.0
        peak_density = peaks / max(candidate.duration, 1.0)
        return clamp01(min(1.0, peaks / 4.0) * 0.65 + min(1.0, peak_density / 1.2) * 0.35)

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        return {"motion_peaks": motion_peak_count(frames), "duration": candidate.duration}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return f"Detected {int(evidence['motion_peaks'])} motion peaks aligned with speech over {float(evidence['duration']):.1f} seconds."


class PointingLikeDetector(ReasonDetector):
    name = "POINTING_LIKE"
    requires_finger = True

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        pointing_frames = 0
        for row in frames:
            if row["finger_visibility_score"] < 0.45:
                continue
            if max(row["left_pinch_distance"], row["right_pinch_distance"]) > 0.12 and row["mean_hand_size"] >= 45:
                pointing_frames += 1
        return clamp01(pointing_frames / max(len(frames), 1) * 1.4)

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        return {"pointing_like_percent": self.score(candidate, frames, text) / 1.4}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return f"Index/thumb separation and visible hand shape suggest pointing in {float(evidence['pointing_like_percent']):.0%} of frames."


class OpenPalmDetector(ReasonDetector):
    name = "OPEN_PALM"
    requires_finger = True

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        open_frames = 0
        for row in frames:
            openness = max(row["left_hand_openness"], row["right_hand_openness"])
            spread = max(row["left_finger_spread"], row["right_finger_spread"])
            if row["finger_visibility_score"] >= 0.45 and openness > 0.22 and spread > 0.09:
                open_frames += 1
        return clamp01(open_frames / max(len(frames), 1) * 1.3)

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        return {"open_palm_percent": self.score(candidate, frames, text) / 1.3}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return f"Fingers appear extended/spread with visible palm in {float(evidence['open_palm_percent']):.0%} of frames."


class FingerArticulationDetector(ReasonDetector):
    name = "FINGER_ARTICULATION"
    requires_finger = True

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        return clamp01(candidate.finger_visibility * 0.55 + candidate.hand_motion * 0.45)

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        return {"finger_visibility": candidate.finger_visibility, "hand_motion": candidate.hand_motion}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return "Finger spread and openness changed while finger landmarks remained visible."


class TwoHandedGestureDetector(ReasonDetector):
    name = "TWO_HANDED_GESTURE"

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        both_visible = sum(1 for row in frames if row["left_hand_visible"] > 0 and row["right_hand_visible"] > 0)
        both_percent = both_visible / max(len(frames), 1)
        return clamp01(both_percent * 0.65 + candidate.hand_motion * 0.35)

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        both_visible = sum(1 for row in frames if row["left_hand_visible"] > 0 and row["right_hand_visible"] > 0)
        return {"both_hands_visible_percent": both_visible / max(len(frames), 1), "hand_motion": candidate.hand_motion}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return f"Both hands visible in {float(evidence['both_hands_visible_percent']):.0%} of frames with hand motion present."


class HandNearFaceDetector(ReasonDetector):
    name = "HAND_NEAR_FACE"

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        return hand_near_face_percent(frames)

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        return {"hand_near_face_percent": hand_near_face_percent(frames)}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return f"Hand approaches face/head region in {float(evidence['hand_near_face_percent']):.0%} of frames."


class TextGestureDetector(ReasonDetector):
    requires_text = True

    def __init__(self, name: str, words: set[str]) -> None:
        self.name = name
        self.words = words

    def score(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> float:
        hits = matching_text_markers(text, self.words)
        if not hits:
            return 0.0
        return clamp01(candidate.percent_frames_gesturing * 0.55 + candidate.mean_gesture_score * 0.35 + min(1.0, len(hits) / 2.0) * 0.10)

    def evidence(self, candidate: Candidate, frames: list[dict[str, float]], text: str) -> dict[str, float | str]:
        return {"markers": ", ".join(matching_text_markers(text, self.words)), "gesture_percent": candidate.percent_frames_gesturing}

    def summary(self, confidence: float, evidence: dict[str, float | str], text: str) -> str:
        return f"Transcript contains '{evidence['markers']}' during high gesture activity."


TEXT_REASON_MARKERS = {
    "en": {
        "DEICTIC_TEXT_WITH_GESTURE": {
            "this",
            "that",
            "here",
            "there",
            "these",
            "those",
            "look",
            "see",
        },
        "EMPHASIS_TEXT_WITH_GESTURE": {
            "really",
            "very",
            "important",
            "exactly",
            "never",
            "always",
            "must",
            "key",
        },
        "CONTRAST_TEXT_WITH_GESTURE": {
            "but",
            "however",
            "on the other hand",
            "whereas",
            "instead",
        },
    },
    "sv": {
        "DEICTIC_TEXT_WITH_GESTURE": {
            "den här",
            "det här",
            "den där",
            "det där",
            "dessa",
            "de här",
            "de där",
            "här",
            "där",
            "detta",
            "denna",
            "titta",
            "kolla",
            "se",
            "ser",
        },
        "EMPHASIS_TEXT_WITH_GESTURE": {
            "verkligen",
            "väldigt",
            "viktigt",
            "exakt",
            "aldrig",
            "alltid",
            "måste",
            "nyckel",
            "helt",
            "riktigt",
            "jätte",
        },
        "CONTRAST_TEXT_WITH_GESTURE": {
            "men",
            "dock",
            "däremot",
            "å andra sidan",
            "medan",
            "istället",
            "i stället",
        },
    },
}


def text_reason_markers_for_language(language: str) -> dict[str, set[str]]:
    normalized = language.lower().split("-")[0]
    if normalized in TEXT_REASON_MARKERS:
        return TEXT_REASON_MARKERS[normalized]
    return TEXT_REASON_MARKERS["en"]


def build_reason_detectors(config: ReasonConfig) -> list[ReasonDetector]:
    text_markers = text_reason_markers_for_language(config.language)
    detectors: list[ReasonDetector] = [
        FaceVisibleDetector(),
        HandsVisibleDetector(),
        LargeArmMovementDetector(),
        BeatGestureDetector(),
        PointingLikeDetector(),
        OpenPalmDetector(),
        FingerArticulationDetector(),
        TwoHandedGestureDetector(),
        HandNearFaceDetector(),
        TextGestureDetector("DEICTIC_TEXT_WITH_GESTURE", text_markers["DEICTIC_TEXT_WITH_GESTURE"]),
        TextGestureDetector("EMPHASIS_TEXT_WITH_GESTURE", text_markers["EMPHASIS_TEXT_WITH_GESTURE"]),
        TextGestureDetector("CONTRAST_TEXT_WITH_GESTURE", text_markers["CONTRAST_TEXT_WITH_GESTURE"]),
    ]
    filtered = []
    for detector in detectors:
        if config.labels is not None and detector.name not in config.labels:
            continue
        if detector.requires_text and not config.enable_text_reasons:
            continue
        if detector.requires_finger and not config.enable_finger_reasons:
            continue
        filtered.append(detector)
    return filtered


def apply_reason_labels(candidate: Candidate, frames: list[dict[str, float]], config: ReasonConfig) -> None:
    scored: list[tuple[str, float, dict[str, float | str], str]] = []
    for detector in build_reason_detectors(config):
        confidence = clamp01(detector.score(candidate, frames, candidate.text))
        evidence = detector.evidence(candidate, frames, candidate.text)
        if confidence >= config.threshold:
            scored.append((detector.name, confidence, evidence, detector.summary(confidence, evidence, candidate.text)))
    scored.sort(key=lambda item: item[1], reverse=True)
    selected = scored[: max(1, config.top_reasons)]
    candidate.reason_labels = [name for name, _, _, _ in selected]
    candidate.primary_reason = selected[0][0] if selected else ""
    candidate.reason_confidences = {name: round(confidence, 6) for name, confidence, _, _ in selected}
    candidate.reason_evidence = {name: evidence for name, _, evidence, _ in selected}
    candidate.evidence_summary = " ".join(summary for _, _, _, summary in selected[:3])


def frame_percent(frames: list[dict[str, float]], key: str, threshold: float) -> float:
    return sum(1 for row in frames if row[key] >= threshold) / max(len(frames), 1)


def motion_peak_count(frames: list[dict[str, float]]) -> int:
    if len(frames) < 3:
        return 0
    values = [row["arm_motion_score"] for row in frames]
    peaks = 0
    for idx in range(1, len(values) - 1):
        if values[idx] > 0.25 and values[idx] >= values[idx - 1] and values[idx] > values[idx + 1]:
            peaks += 1
    return peaks


def hand_near_face_percent(frames: list[dict[str, float]]) -> float:
    near = 0
    eligible = 0
    for row in frames:
        face_visible = row["face_visibility_score"] > 0.4
        if not face_visible:
            continue
        face_x = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2.0
        face_y = min(row["left_shoulder_y"], row["right_shoulder_y"]) - 0.18
        for side in ("left", "right"):
            if row[f"{side}_hand_visible"] <= 0:
                continue
            eligible += 1
            dist = math.hypot(row[f"{side}_wrist_x"] - face_x, row[f"{side}_wrist_y"] - face_y)
            if dist < 0.28:
                near += 1
    return near / max(eligible, 1)


def matching_text_markers(text: str, markers: set[str]) -> list[str]:
    normalized = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    compact = re.sub(r"\s+", " ", normalized)
    padded = f" {compact} "
    return sorted(marker for marker in markers if f" {marker} " in padded)


def build_candidates(
    text_segments: list[TextSegment],
    frame_rows: list[dict[str, float]],
    video_id: str,
    min_duration: float,
    max_duration: float,
    threshold_arg: str,
    top_k: int,
    weights: RankingWeights,
    filters: CandidateFilters,
    reason_config: ReasonConfig,
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
        if not segment.text.strip():
            continue
        scores = np.array([row["gesture_score"] for row in overlapping], dtype=float)
        gesturing = scores >= threshold
        left_visible = np.array([row["left_hand_visible"] for row in overlapping], dtype=float)
        right_visible = np.array([row["right_hand_visible"] for row in overlapping], dtype=float)
        pose_visible = np.array([row["pose_visible"] for row in overlapping], dtype=float)
        face_visibility = float(np.mean([row["face_visibility_score"] for row in overlapping]))
        hand_visibility = float(np.mean([row["hand_visibility_score"] for row in overlapping]))
        finger_visibility = float(np.mean([row["finger_visibility_score"] for row in overlapping]))
        arm_motion = float(np.mean([row["arm_motion_score"] for row in overlapping]))
        hand_motion = float(np.mean([row["hand_motion_score"] for row in overlapping]))
        mean_hand_size = float(np.mean([row["mean_hand_size"] for row in overlapping]))
        visible_percent = float(max(left_visible.mean(), right_visible.mean(), pose_visible.mean() * 0.5))
        percent_gesturing = float(gesturing.mean())
        if visible_percent <= 0 or percent_gesturing <= 0:
            continue
        if face_visibility < filters.min_face_visible:
            continue
        if hand_visibility <= filters.min_hand_visible:
            continue
        if mean_hand_size < filters.min_hand_size:
            continue
        if finger_visibility < filters.min_finger_score:
            continue
        token_count = max(1, len(segment.text.split()))
        mean_score = float(scores.mean())
        max_score = float(scores.max())
        gesture_component = float(np.mean([mean_score, max_score, percent_gesturing]))
        total_weight = max(
            0.001,
            weights.gesture + weights.face + weights.hand + weights.arm + weights.finger,
        )
        candidate_score = (
            gesture_component * weights.gesture
            + face_visibility * weights.face
            + hand_visibility * weights.hand
            + arm_motion * weights.arm
            + finger_visibility * weights.finger
        ) / total_weight
        candidate = Candidate(
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
            face_visibility=face_visibility,
            hand_visibility=hand_visibility,
            finger_visibility=finger_visibility,
            arm_motion=arm_motion,
            hand_motion=hand_motion,
            speech_duration=duration,
            mean_hand_size=mean_hand_size,
        )
        apply_reason_labels(candidate, overlapping, reason_config)
        candidates.append(candidate)
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
        "face_visibility": round(candidate.face_visibility, 6),
        "hand_visibility": round(candidate.hand_visibility, 6),
        "finger_visibility": round(candidate.finger_visibility, 6),
        "arm_motion": round(candidate.arm_motion, 6),
        "hand_motion": round(candidate.hand_motion, 6),
        "speech_duration": round(candidate.speech_duration, 3),
        "mean_hand_size": round(candidate.mean_hand_size, 3),
        "primary_reason": candidate.primary_reason,
        "reason_labels": "|".join(candidate.reason_labels),
        "reason_labels_list": candidate.reason_labels,
        "reason_confidences": candidate.reason_confidences,
        "reason_confidences_json": json.dumps(candidate.reason_confidences, sort_keys=True),
        "reason_evidence": candidate.reason_evidence,
        "evidence_summary": candidate.evidence_summary,
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
        "face_visibility",
        "hand_visibility",
        "finger_visibility",
        "arm_motion",
        "hand_motion",
        "speech_duration",
        "mean_hand_size",
        "primary_reason",
        "reason_labels",
        "reason_confidences_json",
        "evidence_summary",
        "clip_path",
    ]
    with (out_dir / "candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])
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
    reason_options = sorted({label for candidate in candidates for label in candidate.reason_labels})
    for candidate in candidates:
        media = ""
        if candidate.clip_path:
            media = f'<video src="{html.escape(candidate.clip_path)}" controls width="360"></video>'
        badges = " ".join(f'<span class="badge">{html.escape(label)}</span>' for label in candidate.reason_labels)
        primary_confidence = candidate.reason_confidences.get(candidate.primary_reason, 0.0)
        reason_cell = (
            f'<strong>{html.escape(candidate.primary_reason or "NONE")}</strong> '
            f'<span class="reason-confidence">{primary_confidence:.3f}</span>'
            f'<div class="badges">{badges}</div>'
        )
        transcript_cell = (
            f"{html.escape(candidate.text)}"
            f'<div class="evidence-summary">{html.escape(candidate.evidence_summary)}</div>'
        )
        rows.append(
            f'<tr id="candidate-{candidate.rank}" data-reasons="{html.escape("|".join(candidate.reason_labels))}">'
            f"<td>{candidate.rank}</td>"
            f"<td>{candidate.start:.2f}-{candidate.end:.2f}</td>"
            f"<td>{candidate.candidate_score:.3f}</td>"
            f"<td>{reason_cell}</td>"
            f"<td>{primary_confidence:.3f}</td>"
            f"<td>{candidate.face_visibility:.3f}</td>"
            f"<td>{candidate.hand_visibility:.3f}</td>"
            f"<td>{candidate.finger_visibility:.3f}</td>"
            f"<td>{candidate.arm_motion:.3f}</td>"
            f"<td>{candidate.hand_motion:.3f}</td>"
            f"<td>{transcript_cell}</td>"
            f"<td>{media}</td>"
            "</tr>"
        )
    reason_filter_options = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)}</option>' for label in reason_options
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
    .controls {{ margin: 16px 0; }}
    .badge {{ display: inline-block; margin: 4px 4px 0 0; padding: 2px 6px; border-radius: 4px; background: #e8f0f7; color: #134d78; font-size: 12px; }}
    .reason-confidence {{ color: #666; font-size: 12px; margin-left: 4px; }}
    .evidence-summary {{ color: #555; font-size: 13px; margin-top: 6px; max-width: 680px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ text-align: left; cursor: pointer; user-select: none; }}
    th.no-sort {{ cursor: default; }}
  </style>
</head>
<body>
  <h1>Gesture/Text Candidates</h1>
  {build_timeline_chart(candidates, video_duration) if timeline_chart else ""}
  <div class="controls">
    <label for="reason-filter">Reason</label>
    <select id="reason-filter">
      <option value="">All reasons</option>
      {reason_filter_options}
    </select>
  </div>
  <table id="candidates-table">
    <thead><tr><th>Rank</th><th>Time</th><th>Score</th><th class="no-sort">Reason</th><th>Reason Conf.</th><th>Face</th><th>Hand</th><th>Finger</th><th>Arm</th><th>Hand Motion</th><th class="no-sort">Text</th><th class="no-sort">Clip</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <script>
    document.querySelectorAll('#candidates-table th:not(.no-sort)').forEach((th, index) => {{
      th.addEventListener('click', () => {{
        const table = th.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const direction = th.dataset.sort === 'desc' ? 1 : -1;
        rows.sort((a, b) => {{
          const av = parseFloat(a.children[index].textContent) || 0;
          const bv = parseFloat(b.children[index].textContent) || 0;
          return (av - bv) * direction;
        }});
        table.querySelectorAll('th').forEach(header => delete header.dataset.sort);
        th.dataset.sort = direction === 1 ? 'asc' : 'desc';
        rows.forEach(row => tbody.appendChild(row));
      }});
    }});
    document.querySelector('#reason-filter').addEventListener('change', (event) => {{
      const selected = event.target.value;
      document.querySelectorAll('#candidates-table tbody tr').forEach((row) => {{
        const labels = row.dataset.reasons ? row.dataset.reasons.split('|') : [];
        row.style.display = !selected || labels.includes(selected) ? '' : 'none';
      }});
    }});
  </script>
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
    parser.add_argument("--min-face-visible", type=float, default=0.80)
    parser.add_argument("--min-hand-visible", type=float, default=0.0)
    parser.add_argument("--min-hand-size", type=float, default=0.0)
    parser.add_argument("--min-finger-score", type=float, default=0.0)
    parser.add_argument("--gesture-weight", type=float, default=0.30)
    parser.add_argument("--face-weight", type=float, default=0.25)
    parser.add_argument("--hand-weight", type=float, default=0.20)
    parser.add_argument("--arm-weight", type=float, default=0.15)
    parser.add_argument("--finger-weight", type=float, default=0.10)
    parser.add_argument("--reason-threshold", type=float, default=0.60)
    parser.add_argument("--enable-text-reasons", type=parse_bool, default=True)
    parser.add_argument("--enable-finger-reasons", type=parse_bool, default=True)
    parser.add_argument("--reason-labels", default="all", help="Comma-separated reason labels to run, or 'all'.")
    parser.add_argument("--top-reasons", type=int, default=5)
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
        RankingWeights(
            gesture=args.gesture_weight,
            face=args.face_weight,
            hand=args.hand_weight,
            arm=args.arm_weight,
            finger=args.finger_weight,
        ),
        CandidateFilters(
            min_face_visible=args.min_face_visible,
            min_hand_visible=args.min_hand_visible,
            min_hand_size=args.min_hand_size,
            min_finger_score=args.min_finger_score,
        ),
        ReasonConfig(
            threshold=args.reason_threshold,
            enable_text_reasons=args.enable_text_reasons,
            enable_finger_reasons=args.enable_finger_reasons,
            labels=parse_reason_labels(args.reason_labels),
            top_reasons=args.top_reasons,
            language=args.lang,
        ),
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
