from __future__ import annotations

import math


FLOW_SPEED_A = 1.4648375
FLOW_SPEED_B = 93.75
CAPACITY_FLOW_PER_HOUR = 1500.0
CAPACITY_SPEED_KMH = 32.0
FREE_FLOW_THRESHOLD = 351.0
MIN_SPEED_KMH = 5.0


def quarter_hour_to_hourly_flow(quarter_hour_flow: float) -> float:
    return max(0.0, float(quarter_hour_flow) * 4.0)


def speed_from_hourly_flow(
    hourly_flow: float,
    speed_limit_kmh: float = 60.0,
    use_congested_branch: bool = False,
) -> float:
    flow = max(0.0, min(float(hourly_flow), CAPACITY_FLOW_PER_HOUR))
    if flow <= FREE_FLOW_THRESHOLD:
        return float(speed_limit_kmh)

    discriminant = (FLOW_SPEED_B ** 2) - (4.0 * FLOW_SPEED_A * flow)
    if discriminant < 0:
        return CAPACITY_SPEED_KMH

    sqrt_disc = math.sqrt(discriminant)
    low_root = (FLOW_SPEED_B - sqrt_disc) / (2.0 * FLOW_SPEED_A)
    high_root = (FLOW_SPEED_B + sqrt_disc) / (2.0 * FLOW_SPEED_A)

    chosen = low_root if use_congested_branch else high_root
    chosen = max(MIN_SPEED_KMH, min(float(speed_limit_kmh), chosen))
    return chosen


def travel_time_minutes(
    distance_km: float,
    hourly_flow: float,
    speed_limit_kmh: float = 60.0,
    use_congested_branch: bool = False,
    intersection_delay_seconds: float = 30.0,
) -> float:
    speed_kmh = speed_from_hourly_flow(
        hourly_flow=hourly_flow,
        speed_limit_kmh=speed_limit_kmh,
        use_congested_branch=use_congested_branch,
    )
    moving_minutes = (float(distance_km) / speed_kmh) * 60.0
    delay_minutes = float(intersection_delay_seconds) / 60.0
    return moving_minutes + delay_minutes
