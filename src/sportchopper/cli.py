"""Command-line entry point: `sportchopper chop <input> -o <output> [options]`"""

import argparse
import json
import sys
from typing import List, Optional, Tuple

from tqdm import tqdm

from .activity import compute_score
from .config import SportPreset, get_preset
from .cutter import cut_and_concat
from .detection import Detector
from .segments import Segment, build_segments
from .tracking import ActivityTracker
from .video import VideoInfo, get_video_info, iter_sampled_frames


def _analyze(
    input_path: str, preset: SportPreset, conf_override: Optional[float]
) -> Tuple[VideoInfo, List[float], List[float]]:
    conf = conf_override if conf_override is not None else preset.conf_threshold
    detector = Detector(conf_threshold=conf)
    tracker = ActivityTracker()

    info = get_video_info(input_path)
    timestamps: List[float] = []
    scores: List[float] = []

    total = int(info.duration * preset.sample_fps) + 1
    for ts, frame in tqdm(
        iter_sampled_frames(input_path, preset.sample_fps),
        total=total,
        desc="Analyzing",
        unit="frame",
    ):
        people, balls = detector.detect(frame)
        tracked = tracker.update(ts, people, balls)
        timestamps.append(ts)
        scores.append(compute_score(tracked, preset))

    return info, timestamps, scores


def _save_debug_plot(
    timestamps: List[float],
    scores: List[float],
    segments: List[Segment],
    preset: SportPreset,
    path: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestamps, scores, label="activity score", linewidth=1)
    ax.axhline(preset.enter_threshold, color="green", linestyle="--", label="enter threshold")
    ax.axhline(preset.exit_threshold, color="red", linestyle="--", label="exit threshold")
    for seg in segments:
        ax.axvspan(seg.start, seg.end, color="green", alpha=0.15)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("activity score")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sportchopper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chop = subparsers.add_parser("chop", help="Cut dead time between points out of a video")
    chop.add_argument("input", help="Path to source video")
    chop.add_argument("-o", "--output", default="output.mp4", help="Path to write the trimmed video")
    chop.add_argument("--sport", default="volleyball", choices=["volleyball", "tennis", "pickleball"])
    chop.add_argument("--sample-fps", type=float, default=None, help="Override analysis frame rate")
    chop.add_argument("--min-point-len", type=float, default=None, help="Override minimum point duration (s)")
    chop.add_argument("--min-gap", type=float, default=None, help="Override minimum gap to merge across (s)")
    chop.add_argument("--pad", type=float, default=None, help="Override padding around each kept segment (s)")
    chop.add_argument("--conf", type=float, default=None, help="Override detector confidence threshold")
    chop.add_argument("--dry-run", action="store_true", help="Print detected segments without cutting")
    chop.add_argument("--debug-plot", metavar="PATH", default=None, help="Save an activity-score plot to PATH")

    args = parser.parse_args(argv)

    preset = get_preset(args.sport)
    if args.sample_fps is not None:
        preset.sample_fps = args.sample_fps
    if args.min_point_len is not None:
        preset.min_point_len = args.min_point_len
    if args.min_gap is not None:
        preset.min_gap = args.min_gap
    if args.pad is not None:
        preset.pad = args.pad

    info, timestamps, scores = _analyze(args.input, preset, args.conf)

    segments = build_segments(
        timestamps=timestamps,
        scores=scores,
        video_duration=info.duration,
        enter_threshold=preset.enter_threshold,
        exit_threshold=preset.exit_threshold,
        min_point_len=preset.min_point_len,
        min_gap=preset.min_gap,
        pad=preset.pad,
    )

    if args.debug_plot:
        _save_debug_plot(timestamps, scores, segments, preset, args.debug_plot)

    if not segments:
        print(
            "No points detected - try lowering --conf, or the enter/exit thresholds in config.py.",
            file=sys.stderr,
        )
        return 1

    total_kept = sum(s.duration for s in segments)
    print(f"Detected {len(segments)} point(s), keeping {total_kept:.1f}s of {info.duration:.1f}s source.")

    if args.dry_run:
        print(json.dumps([{"start": s.start, "end": s.end} for s in segments], indent=2))
        return 0

    cut_and_concat(args.input, segments, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
