from __future__ import annotations

import unittest

from tbrgs.travel_time import (
    CAPACITY_FLOW_PER_HOUR,
    MIN_SPEED_KMH,
    quarter_hour_to_hourly_flow,
    speed_from_hourly_flow,
    travel_time_minutes,
)


class TravelTimeTests(unittest.TestCase):
    def test_quarter_hour_to_hourly_flow(self) -> None:
        self.assertEqual(quarter_hour_to_hourly_flow(20), 80.0)

    def test_quarter_hour_negative_is_clipped(self) -> None:
        self.assertEqual(quarter_hour_to_hourly_flow(-5), 0.0)

    def test_speed_free_flow_uses_speed_limit(self) -> None:
        self.assertEqual(speed_from_hourly_flow(200, speed_limit_kmh=60.0), 60.0)

    def test_speed_drops_below_limit_after_threshold(self) -> None:
        speed = speed_from_hourly_flow(900, speed_limit_kmh=60.0)
        self.assertLess(speed, 60.0)

    def test_speed_is_never_below_minimum(self) -> None:
        speed = speed_from_hourly_flow(CAPACITY_FLOW_PER_HOUR * 10, speed_limit_kmh=60.0)
        self.assertGreaterEqual(speed, MIN_SPEED_KMH)

    def test_congested_branch_is_slower(self) -> None:
        uncongested = speed_from_hourly_flow(1200, use_congested_branch=False)
        congested = speed_from_hourly_flow(1200, use_congested_branch=True)
        self.assertLess(congested, uncongested)

    def test_travel_time_is_positive(self) -> None:
        minutes = travel_time_minutes(distance_km=1.5, hourly_flow=800)
        self.assertGreater(minutes, 0.0)

    def test_intersection_delay_increases_time(self) -> None:
        no_delay = travel_time_minutes(distance_km=2.0, hourly_flow=600, intersection_delay_seconds=0)
        delayed = travel_time_minutes(distance_km=2.0, hourly_flow=600, intersection_delay_seconds=30)
        self.assertGreater(delayed, no_delay)


if __name__ == "__main__":
    unittest.main()
