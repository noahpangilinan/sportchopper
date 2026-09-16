"""Simple nearest-centroid tracking across sampled frames.

This is not a full multi-object tracker (no ID re-acquisition, no Kalman
filtering) - it's just enough continuity between consecutive *sampled*
frames to estimate ball speed and aggregate player motion, which is all the
activity signal in activity.py needs.
"""

from dataclasses import dataclass
from typing import List, Optional

from .detection import Detection


@dataclass
class TrackedFrame:
    timestamp: float
    ball_present: bool
    ball_speed: float  # pixels/sec; 0 if no ball or no previous ball to compare against
    num_people: int
    player_motion: float  # average pixels/sec centroid movement of matched people


class ActivityTracker:
    """Call `update()` once per sampled frame, in increasing timestamp order."""

    def __init__(self, match_radius: float = 150.0):
        self._match_radius = match_radius
        self._prev_ball: Optional[Detection] = None
        self._prev_ts: Optional[float] = None
        self._prev_people: List[Detection] = []

    def update(
        self, timestamp: float, people: List[Detection], balls: List[Detection]
    ) -> TrackedFrame:
        dt = (timestamp - self._prev_ts) if self._prev_ts is not None else None

        ball = max(balls, key=lambda d: d.conf) if balls else None
        ball_speed = 0.0
        if ball is not None and self._prev_ball is not None and dt:
            dist = ((ball.cx - self._prev_ball.cx) ** 2 + (ball.cy - self._prev_ball.cy) ** 2) ** 0.5
            ball_speed = dist / dt

        player_motion = 0.0
        if people and self._prev_people and dt:
            total = 0.0
            matched = 0
            for det in people:
                nearest = min(
                    self._prev_people,
                    key=lambda p: (p.cx - det.cx) ** 2 + (p.cy - det.cy) ** 2,
                )
                dist = ((det.cx - nearest.cx) ** 2 + (det.cy - nearest.cy) ** 2) ** 0.5
                if dist <= self._match_radius:
                    total += dist / dt
                    matched += 1
            if matched:
                player_motion = total / matched

        result = TrackedFrame(
            timestamp=timestamp,
            ball_present=ball is not None,
            ball_speed=ball_speed,
            num_people=len(people),
            player_motion=player_motion,
        )

        self._prev_ball = ball
        self._prev_ts = timestamp
        self._prev_people = people
        return result
