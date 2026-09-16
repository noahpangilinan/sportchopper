from sportchopper.activity import compute_score
from sportchopper.config import get_preset
from sportchopper.tracking import TrackedFrame


def test_high_ball_speed_and_player_motion_score_high():
    preset = get_preset("volleyball")
    tracked = TrackedFrame(
        timestamp=1.0, ball_present=True, ball_speed=3000.0, num_people=4, player_motion=400.0
    )
    assert compute_score(tracked, preset) > 0.8


def test_no_ball_and_static_players_score_low():
    preset = get_preset("volleyball")
    tracked = TrackedFrame(
        timestamp=1.0, ball_present=False, ball_speed=0.0, num_people=4, player_motion=10.0
    )
    assert compute_score(tracked, preset) < 0.2


def test_ball_present_but_stationary_scores_moderately():
    preset = get_preset("volleyball")
    tracked = TrackedFrame(
        timestamp=1.0, ball_present=True, ball_speed=0.0, num_people=2, player_motion=0.0
    )
    score = compute_score(tracked, preset)
    assert 0.2 < score < 0.5


def test_score_is_clamped_to_unit_range():
    preset = get_preset("volleyball")
    tracked = TrackedFrame(
        timestamp=1.0, ball_present=True, ball_speed=999_999.0, num_people=10, player_motion=999_999.0
    )
    assert compute_score(tracked, preset) <= 1.0
