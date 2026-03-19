#!/usr/bin/env python3
"""Shared helper functions for position, distance, and runtime load estimation."""

import math
import re


def get_node_xy(node):
    """Lấy (x, y) từ Mininet node hoặc SimpleNamespace."""
    pos = getattr(node, 'position', None) or getattr(node, 'pos', None)
    if pos is not None:
        if isinstance(pos, str):
            parts = [p.strip() for p in pos.split(',')]
            return float(parts[0]), float(parts[1])
        return float(pos[0]), float(pos[1])
    
    if hasattr(node, 'params') and 'position' in node.params:
        p = node.params['position']
        if isinstance(p, str):
            parts = [p_part.strip() for p_part in p.split(',')]
            return float(parts[0]), float(parts[1])
        return float(p[0]), float(p[1])
    return 0.0, 0.0


def dist_2d(n1, n2):
    """Khoảng cách 2D giữa 2 node."""
    x1, y1 = get_node_xy(n1)
    x2, y2 = get_node_xy(n2)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def dist_3d(uav_node, user_node, altitude):
    """d_{l,k} = sqrt(d_2d^2 + H^2)"""
    d2 = dist_2d(uav_node, user_node)
    return math.sqrt(d2 ** 2 + altitude ** 2)


def get_associated_ap_name(car):
    """
    Read AP name currently associated with a car (if Mininet exposes metadata).
    Returns None when metadata is unavailable.
    """
    assoc = None
    params = getattr(car, "params", None)
    if isinstance(params, dict):
        assoc = params.get("associatedTo", None)
    if assoc is None:
        assoc = getattr(car, "associatedTo", None)

    if isinstance(assoc, (list, tuple)):
        assoc = assoc[0] if assoc else None
    if assoc is None:
        return None
    if isinstance(assoc, str):
        return assoc
    return getattr(assoc, "name", None)


def _has_association_metadata(car):
    """True nếu car có metadata association (kể cả giá trị None)."""
    params = getattr(car, "params", None)
    if isinstance(params, dict) and ("associatedTo" in params):
        return True
    # Tránh false-positive khi object có sẵn attribute nhưng giá trị luôn None.
    return getattr(car, "associatedTo", None) is not None


def _normalize_ap_name(name):
    """Normalize AP names like 'uav1-wlan1' -> 'uav1' for robust matching."""
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    s = re.sub(r"-wlan\d+$", "", s)
    return s


def estimate_runtime_users_for_uav(cars, uavs, uav_idx, uav_range):
    """
    Estimate runtime user count for selected UAV.
    Hybrid strategy per car:
      - Nếu car có association metadata -> dùng metadata.
      - Nếu car thiếu metadata -> fallback coverage-distance cho chính car đó.
    """
    idx = int(uav_idx)
    if not (0 <= idx < len(uavs)):
        return 0

    target = uavs[idx]
    target_name = _normalize_ap_name(getattr(target, "name", None))
    c = 0
    for car in cars:
        has_assoc_meta = _has_association_metadata(car)
        assoc_name = _normalize_ap_name(get_associated_ap_name(car))
        if has_assoc_meta:
            if target_name and assoc_name == target_name:
                c += 1
            continue

        # Chỉ fallback khi metadata association thực sự không có.
        if dist_2d(car, target) <= float(uav_range):
            c += 1
    return c


def estimate_runtime_cpu_load(users_on_uav, max_users_per_uav, cap=0.95):
    """[DEPRECATED] Kept for backward-compat; delay model Xie không dùng CPU load."""
    m_cfg = max(int(max_users_per_uav), 1)
    cap_f = min(max(float(cap), 0.1), 0.99)
    return min(max(float(users_on_uav) / float(m_cfg), 0.0), cap_f)
