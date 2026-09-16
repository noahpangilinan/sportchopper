"""Video frame sampling utilities built on OpenCV."""

from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int


def get_video_info(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return VideoInfo(
        fps=fps,
        frame_count=frame_count,
        duration=frame_count / fps,
        width=width,
        height=height,
    )


def iter_sampled_frames(path: str, sample_fps: float) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (timestamp_seconds, frame) pairs sampled at ~sample_fps, regardless
    of the source video's native frame rate."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(source_fps / sample_fps))

        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                yield frame_index / source_fps, frame
            frame_index += 1
    finally:
        cap.release()
