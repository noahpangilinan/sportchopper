"""Turn a smoothed per-sample activity score into kept 'live play' segments."""

from dataclasses import dataclass
from typing import List


@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def smooth(scores: List[float], window: int = 3) -> List[float]:
    if window <= 1 or len(scores) <= 1:
        return list(scores)
    half = window // 2
    smoothed = []
    for i in range(len(scores)):
        lo = max(0, i - half)
        hi = min(len(scores), i + half + 1)
        smoothed.append(sum(scores[lo:hi]) / (hi - lo))
    return smoothed


def hysteresis_segments(
    timestamps: List[float],
    scores: List[float],
    enter_threshold: float,
    exit_threshold: float,
) -> List[Segment]:
    """Enter a segment once score crosses enter_threshold, stay in it until
    score drops below exit_threshold. Two thresholds (rather than one) avoid
    flicker from momentary detection dropout/occlusion."""
    segments: List[Segment] = []
    active = False
    start_ts = None

    for ts, score in zip(timestamps, scores):
        if not active and score >= enter_threshold:
            active = True
            start_ts = ts
        elif active and score < exit_threshold:
            active = False
            segments.append(Segment(start_ts, ts))

    if active and start_ts is not None:
        segments.append(Segment(start_ts, timestamps[-1]))

    return segments


def merge_close_segments(segments: List[Segment], min_gap: float) -> List[Segment]:
    if not segments:
        return []
    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.start - merged[-1].end <= min_gap:
            merged[-1] = Segment(merged[-1].start, seg.end)
        else:
            merged.append(seg)
    return merged


def drop_short_segments(segments: List[Segment], min_point_len: float) -> List[Segment]:
    return [s for s in segments if s.duration >= min_point_len]


def pad_segments(segments: List[Segment], pad: float, video_duration: float) -> List[Segment]:
    return [
        Segment(max(0.0, s.start - pad), min(video_duration, s.end + pad))
        for s in segments
    ]


def merge_overlapping(segments: List[Segment]) -> List[Segment]:
    """Padding can make adjacent kept segments overlap; merge those back together."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s.start)
    merged = [ordered[0]]
    for seg in ordered[1:]:
        if seg.start <= merged[-1].end:
            merged[-1] = Segment(merged[-1].start, max(merged[-1].end, seg.end))
        else:
            merged.append(seg)
    return merged


def build_segments(
    timestamps: List[float],
    scores: List[float],
    video_duration: float,
    enter_threshold: float,
    exit_threshold: float,
    min_point_len: float,
    min_gap: float,
    pad: float,
    smoothing_window: int = 3,
) -> List[Segment]:
    smoothed = smooth(scores, smoothing_window)
    raw = hysteresis_segments(timestamps, smoothed, enter_threshold, exit_threshold)
    merged = merge_close_segments(raw, min_gap)
    kept = drop_short_segments(merged, min_point_len)
    padded = pad_segments(kept, pad, video_duration)
    return merge_overlapping(padded)
