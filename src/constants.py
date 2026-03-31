#!/usr/bin/env python3
"""
Hằng số dùng chung trong toàn bộ project.

Lưu ý:
- Tránh "đóng băng" config tại import-time (khó debug khi chạy nhiều cấu hình).
- Vẫn giữ API cũ: import UAV_RANGE/MBS_RANGE/... sẽ hoạt động.
"""

from config import get_config

def _cfg():
    # get_config() là cheap; gọi mỗi lần giúp đồng bộ với config mặc định hiện tại.
    return get_config()


def __getattr__(name: str):
    """
    Module attribute hook (PEP 562).
    Cho phép `from constants import UAV_RANGE` vẫn dùng được và giá trị luôn lấy từ config.
    """
    c = _cfg()
    if name == "UAV_RANGE":
        return float(getattr(c, "uav_range", 150.0))
    if name == "MBS_RANGE":
        return float(getattr(c, "mbs_range", 250.0))
    if name == "UAV_ALTITUDE":
        return float(getattr(c, "H", 100.0))
    if name == "UAV_TRAJECTORY_SPEED_MS":
        return float(getattr(c, "uav_speed", 20.0))
    if name == "UAV_TRAJECTORY_RADIUS":
        return float(getattr(c, "uav_radius", 200.0))
    raise AttributeError(name)


# Backward-compatible eager values (không bắt buộc, nhưng hữu ích khi introspect).
_c0 = _cfg()
UAV_RANGE               = float(getattr(_c0, 'uav_range', 150.0))
MBS_RANGE               = float(getattr(_c0, 'mbs_range', 250.0))
UAV_ALTITUDE            = float(getattr(_c0, 'H', 100.0))
UAV_TRAJECTORY_SPEED_MS = float(getattr(_c0, 'uav_speed', 20.0))
UAV_TRAJECTORY_RADIUS   = float(getattr(_c0, 'uav_radius', 200.0))
