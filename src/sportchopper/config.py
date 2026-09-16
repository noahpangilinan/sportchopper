"""Tunable defaults for the detection/segmentation pipeline, per sport."""

from dataclasses import dataclass

# COCO class ids used by the pretrained YOLOv8 weights.
COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32


@dataclass
class SportPreset:
    sample_fps: float
    conf_threshold: float
    ball_weight: float
    player_motion_weight: float
    enter_threshold: float
    exit_threshold: float
    min_point_len: float
    min_gap: float
    pad: float


PRESETS = {
    "volleyball": SportPreset(
        sample_fps=5.0,
        conf_threshold=0.30,
        ball_weight=0.6,
        player_motion_weight=0.4,
        enter_threshold=0.55,
        exit_threshold=0.35,
        min_point_len=2.0,
        min_gap=1.5,
        pad=1.0,
    ),
    "tennis": SportPreset(
        sample_fps=5.0,
        conf_threshold=0.25,
        ball_weight=0.7,
        player_motion_weight=0.3,
        enter_threshold=0.5,
        exit_threshold=0.3,
        min_point_len=1.5,
        min_gap=2.0,
        pad=1.0,
    ),
    "pickleball": SportPreset(
        sample_fps=5.0,
        conf_threshold=0.25,
        ball_weight=0.7,
        player_motion_weight=0.3,
        enter_threshold=0.5,
        exit_threshold=0.3,
        min_point_len=1.0,
        min_gap=1.5,
        pad=0.8,
    ),
}


def get_preset(sport: str) -> SportPreset:
    try:
        return PRESETS[sport]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sport {sport!r}. Choose one of: {', '.join(PRESETS)}"
        ) from exc
