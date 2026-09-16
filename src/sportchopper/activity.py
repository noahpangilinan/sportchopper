"""Combine tracked ball/player motion into a single 0-1 'live play' score."""

from .config import SportPreset
from .tracking import TrackedFrame

# Reference scales used to normalize raw pixel/sec speeds into 0-1 components.
# These are heuristic starting points and will likely need tuning against
# real footage/resolution (see --debug-plot in the CLI).
BALL_SPEED_REFERENCE = 2000.0  # px/s considered "fast" (ball clearly in play)
PLAYER_MOTION_REFERENCE = 300.0  # px/s considered "active" player movement


def compute_score(tracked: TrackedFrame, preset: SportPreset) -> float:
    ball_component = 0.0
    if tracked.ball_present:
        speed_component = min(tracked.ball_speed / BALL_SPEED_REFERENCE, 1.0)
        ball_component = 0.5 + 0.5 * speed_component

    player_component = min(tracked.player_motion / PLAYER_MOTION_REFERENCE, 1.0)

    score = (
        preset.ball_weight * ball_component
        + preset.player_motion_weight * player_component
    )
    return max(0.0, min(1.0, score))
