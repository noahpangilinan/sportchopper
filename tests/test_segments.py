from sportchopper.segments import (
    Segment,
    build_segments,
    drop_short_segments,
    hysteresis_segments,
    merge_close_segments,
    merge_overlapping,
    pad_segments,
    smooth,
)


def test_smooth_averages_neighbors():
    assert smooth([0, 0, 1, 0, 0], window=3) == [0.0, 1 / 3, 1 / 3, 1 / 3, 0.0]


def test_smooth_passthrough_when_window_is_one():
    assert smooth([0.1, 0.9, 0.2], window=1) == [0.1, 0.9, 0.2]


def test_hysteresis_segments_basic_rise_and_fall():
    timestamps = [0, 1, 2, 3, 4, 5]
    scores = [0, 0.6, 0.6, 0.6, 0.2, 0.2]
    segments = hysteresis_segments(timestamps, scores, enter_threshold=0.5, exit_threshold=0.3)
    assert segments == [Segment(1, 4)]


def test_hysteresis_segments_stays_open_until_end_of_clip():
    timestamps = [0, 1, 2, 3]
    scores = [0, 0.6, 0.6, 0.6]
    segments = hysteresis_segments(timestamps, scores, enter_threshold=0.5, exit_threshold=0.3)
    assert segments == [Segment(1, 3)]


def test_hysteresis_ignores_dips_that_dont_cross_exit_threshold():
    timestamps = [0, 1, 2, 3, 4]
    scores = [0.6, 0.6, 0.4, 0.6, 0.2]  # dip to 0.4 stays above exit_threshold=0.3
    segments = hysteresis_segments(timestamps, scores, enter_threshold=0.5, exit_threshold=0.3)
    assert segments == [Segment(0, 4)]


def test_merge_close_segments_bridges_short_gap():
    segments = [Segment(0, 2), Segment(2.5, 4)]
    assert merge_close_segments(segments, min_gap=1.0) == [Segment(0, 4)]


def test_merge_close_segments_keeps_far_apart_segments_separate():
    segments = [Segment(0, 2), Segment(5, 7)]
    assert merge_close_segments(segments, min_gap=1.0) == [Segment(0, 2), Segment(5, 7)]


def test_drop_short_segments_filters_out_blips():
    segments = [Segment(0, 0.5), Segment(2, 6)]
    assert drop_short_segments(segments, min_point_len=2.0) == [Segment(2, 6)]


def test_pad_segments_clamps_to_video_bounds():
    segments = [Segment(0.5, 9.5)]
    assert pad_segments(segments, pad=1.0, video_duration=10.0) == [Segment(0.0, 10.0)]


def test_merge_overlapping_combines_padded_segments():
    segments = [Segment(0, 3), Segment(2, 5)]
    assert merge_overlapping(segments) == [Segment(0, 5)]


def test_build_segments_end_to_end():
    timestamps = [i * 0.5 for i in range(20)]  # 0..9.5s
    scores = [0.1] * 4 + [0.7] * 8 + [0.1] * 8  # dead / point / dead
    segments = build_segments(
        timestamps=timestamps,
        scores=scores,
        video_duration=10.0,
        enter_threshold=0.5,
        exit_threshold=0.3,
        min_point_len=1.0,
        min_gap=0.5,
        pad=0.5,
        smoothing_window=1,
    )
    assert len(segments) == 1
    assert segments[0].start < 2.0
    assert segments[0].end > 5.5


def test_build_segments_drops_short_blip_and_keeps_real_point():
    timestamps = [i * 0.5 for i in range(20)]
    scores = [0.1, 0.6] + [0.1] * 4 + [0.7] * 8 + [0.1] * 6  # a 0.5s blip, then a real point
    segments = build_segments(
        timestamps=timestamps,
        scores=scores,
        video_duration=10.0,
        enter_threshold=0.5,
        exit_threshold=0.3,
        min_point_len=1.5,
        min_gap=0.5,
        pad=0.0,
        smoothing_window=1,
    )
    assert len(segments) == 1
