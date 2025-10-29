from balance_game.physics import BalancePhysics
from balance_game.types import Keypoints2D, Point2D, StabilizerConfig


def test_balance_stable_after_initial_frame():
    cfg = StabilizerConfig(max_tilt_deg=10.0, max_jerk=1.0, clear_seconds=1.0)
    physics = BalancePhysics(cfg)

    kps = Keypoints2D(
        nose=Point2D(100, 50),
        left_shoulder=Point2D(80, 100),
        right_shoulder=Point2D(120, 100),
    )

    # 1フレーム目は prev が無いため不安定
    stable, metrics = physics.update(kps, 0.033)
    assert stable is False

    # 2フレーム目（同位置）で安定になる
    stable, metrics = physics.update(kps, 0.033)
    assert stable is True
    assert metrics.tilt_deg == 0.0
