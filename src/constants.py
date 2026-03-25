#!/usr/bin/env python3
"""Hằng số dùng chung trong toàn bộ project — single source of truth (từ config)."""

from config import get_config
_config = get_config()

UAV_RANGE               = float(getattr(_config, 'uav_range', 150.0))
MBS_RANGE               = float(getattr(_config, 'mbs_range', 250.0))
UAV_ALTITUDE            = float(getattr(_config, 'H', 100.0))
UAV_TRAJECTORY_SPEED_MS = float(getattr(_config, 'uav_speed', 20.0))
UAV_TRAJECTORY_RADIUS   = float(getattr(_config, 'uav_radius', 200.0))
