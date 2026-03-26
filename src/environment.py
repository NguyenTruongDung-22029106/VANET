#!/usr/bin/env python3
"""
environment.py — VanetEnvironment: Môi trường học tăng cường theo mô hình Xie et al.

State  : động theo số UAV, gồm xe requesting + UAV features + MBS context + popularity
Action : (#UAV × #bitrate × cache_decision) — chỉ UAV, không có MBS tier
         Encoding: a = uav_idx + L*(z_cached + Z*cache_dec)
Reward : -log(1+delay), raw delay trong info['raw_delay']
"""
import math
import random
import numpy as np
from types import SimpleNamespace
from models import calculate_total_cost, calculate_mbs_only_delay
from helpers import (
    get_node_xy, dist_2d,
)
from constants import UAV_RANGE, MBS_RANGE


# ============================================================
# Helper: tính phân phối Zipf
# ============================================================

def _compute_zipf_probs(num_videos: int, gamma: float) -> np.ndarray:
    ranks = np.arange(1, num_videos + 1, dtype=np.float64)
    raw   = ranks ** (-gamma)
    return raw / raw.sum()

def _compute_zipf_joint_probs(num_videos: int, num_bitrates: int, gamma: float) -> np.ndarray:
    """Zipf trên toàn bộ cặp (f,z), shape (F, Z)."""
    F = max(int(num_videos), 1)
    Z = max(int(num_bitrates), 1)
    ranks = np.arange(1, F * Z + 1, dtype=np.float64)
    raw = ranks ** (-gamma)
    return (raw / raw.sum()).reshape(F, Z)


def _chunk_size_bits(z_idx: int, config) -> float:
    """s_{f,z}: chunk size theo bitrate z (tuyến tính theo z+1)."""
    s0_bits = float(getattr(config, 'chunk_size_MB', 8.0)) * 8e6
    return s0_bits * (int(z_idx) + 1)


# ============================================================
# Helper: tạo dummy nodes từ config
# ============================================================

def _make_dummy_nodes(config):
    """
    Tạo SimpleNamespace nodes từ config — dùng nội bộ cho from_config().
    Vị trí tam giác đều khớp với main_thesis.py.
    """
    plot_max = getattr(config, 'plot_max', 400)
    cx, cy   = plot_max / 2.0, plot_max / 2.0

    num_cars = getattr(config, 'cars', 10)
    num_uavs = getattr(config, 'uavs', 3)
    num_rsus = getattr(config, 'rsus', 1)

    r_poly   = plot_max / 4.0
    uav_verts = []
    for i in range(num_uavs):
        angle = 2 * math.pi * i / max(num_uavs, 1) + math.pi / 2
        uav_verts.append((cx + r_poly * math.cos(angle), cy + r_poly * math.sin(angle)))

    cars = [
        SimpleNamespace(
            name=f'car{i}',
            params={'position': (cx + (i - num_cars // 2) * 30, cy - 50)}
        )
        for i in range(1, num_cars + 1)
    ]
    rsus = [
        SimpleNamespace(
            name=f'rsu{i}',
            params={'position': (50 + (i - 1) * 150, 50)}
        )
        for i in range(1, num_rsus + 1)
    ]
    uavs = [
        SimpleNamespace(
            name=f'uav{i}',
            params={'position': uav_verts[i - 1]}
        )
        for i in range(1, num_uavs + 1)
    ]
    stations = cars + uavs
    return stations, rsus, uavs


def _nearest_in_range_rsu(car_node, rsus):
    """Return nearest RSU/MBS within MBS_RANGE, else None."""
    best_node = None
    best_d = None
    for rsu in rsus:
        d = dist_2d(car_node, rsu)
        if d <= float(MBS_RANGE) and (best_d is None or d < best_d):
            best_d = d
            best_node = rsu
    return best_node


def _nearest_rsu(car_node, rsus):
    """Return nearest RSU/MBS regardless of coverage radius, else None."""
    best_node = None
    best_d = None
    for rsu in rsus:
        d = dist_2d(car_node, rsu)
        if best_d is None or d < best_d:
            best_d = d
            best_node = rsu
    return best_node


def _count_cars_in_uav_range(cars, uav_node):
    """Estimate runtime users served by a UAV by coverage count."""
    if uav_node is None:
        return 0
    c = 0
    for car in cars:
        if dist_2d(car, uav_node) <= float(UAV_RANGE):
            c += 1
    return c


def _count_cars_in_mbs_range(cars, mbs_node):
    """Estimate runtime users served by an MBS/RSU by coverage count."""
    if mbs_node is None:
        return 0
    c = 0
    for car in cars:
        if dist_2d(car, mbs_node) <= float(MBS_RANGE):
            c += 1
    return c


def _effective_mbs_capacity(config) -> int:
    """Return serving capacity used to normalize MBS load."""
    return max(int(getattr(config, 'M_bs', getattr(config, 'M', 30))), 1)


def _disconnect_reward(config) -> float:
    """Reward used when request is dropped due to no reachable serving tier."""
    return float(getattr(config, 'disconnect_reward', -2.0))


# ============================================================
# VanetEnvironment
# ============================================================

class VanetEnvironment:
    """
    Môi trường VANET-UAV-SDN cho D3QN.

    Action encoding (3-chiều):
        a = uav_idx + L * (z_cached + Z * cache_dec)
        MBS tier: a = L * Z * 2

    Ví dụ với L=5 UAV, Z=4 bitrates:
        action_size = 5*4*2 + 1 = 41

    Khởi tạo:
        VanetEnvironment(config, stations, aps, uavs_list)  ← dùng trong main_thesis.py
        VanetEnvironment.from_config(config)                ← dùng trong ryu_app.py
    """
    def __init__(self, config, stations, aps=None, uavs_list=None):
        self.config = config

        self.rsus = list(aps)       if aps        else []
        self.uavs = list(uavs_list) if uavs_list  else []

        rsu_names = {n.name for n in self.rsus}
        uav_names = {n.name for n in self.uavs}
        self.cars = [
            n for n in stations
            if getattr(n, 'name', '') not in rsu_names
            and getattr(n, 'name', '') not in uav_names
        ]

        self.num_bitrates        = int(getattr(config, 'num_bitrates', 4))
        self.num_cache_actions   = int(getattr(config, 'num_cache_acts', 2))
        self.num_offload_targets = len(self.uavs)
        self.Z                   = self.num_bitrates

        # action_size = L * Z * 2 (chỉ UAV, không có MBS — theo đúng Paper Xie et al.)
        self._uav_action_size = self.num_offload_targets * self.num_bitrates * self.num_cache_actions
        self.action_size = max(1, self._uav_action_size)

        # State: car pos + UAV features + nearest-MBS context + popularity + z_req (one-hot)
        self.state_size = (
            2 +                      # requesting user position
            len(self.uavs) * 2 +      # UAV positions
            len(self.uavs) +          # distance user→each UAV
            len(self.uavs) +          # cache fullness per UAV
            1 +                       # distance user→nearest MBS/RSU
            1 +                       # normalized load around nearest MBS/RSU
            1 +                       # popularity of requested chunk
            self.Z                    # z_req one-hot encoded
        )

        # Paper caching: y_{l,f,z} per UAV, per chunk, per bitrate
        self.cache_uav_MB = float(getattr(config, 'cache_uav_MB', 300))
        self.C_cache_bits = self.cache_uav_MB * 8e6
        self.chunk_sizes = np.array(
            [_chunk_size_bits(z, config) for z in range(self.Z)],
            dtype=np.float64,
        )
        self.Y = np.zeros((len(self.uavs), int(getattr(config, 'num_videos', 100)), self.Z), dtype=np.int8)

        self.num_videos    = int(getattr(config, 'num_videos',    100))
        self.zipf_exponent = float(getattr(config, 'zipf_exponent', 0.7))
        self._zipf_probs_fz = _compute_zipf_joint_probs(
            self.num_videos, self.Z, self.zipf_exponent
        )
        self._zipf_probs = self._zipf_probs_fz.sum(axis=1)
        self.f_req         = 0
        self.z_req         = 0

        self.requesting_car = self.cars[0] if self.cars else None

        self._norm_scale = float(getattr(config, 'plot_max', 400))
        self._last_actual_uav_idx = None
        self._last_served_request = None

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config):
        """
        Tạo VanetEnvironment trực tiếp từ config — không cần truyền nodes ngoài.
        Dùng cho ryu_app.py.
        """
        stations, rsus, uavs = _make_dummy_nodes(config)
        return cls(config, stations, aps=rsus, uavs_list=uavs)

    # ------------------------------------------------------------------
    @staticmethod
    def get_pos_from_node(node):
        return get_node_xy(node)

    def _cache_usage_bits(self, uav_idx: int) -> float:
        return float(np.sum(self.Y[uav_idx] * self.chunk_sizes[np.newaxis, :]))

    def get_state(self):
        sv = []
        ns = self._norm_scale
        if self.requesting_car is not None:
            car = self.requesting_car
        elif self.cars:
            car = self.cars[0]
        else:
            car = None

        if car is None:
            cx, cy = 0.0, 0.0
        else:
            cx, cy = self.get_pos_from_node(car)
        sv.extend([cx / ns, cy / ns])
        for n in self.uavs:
            x, y = self.get_pos_from_node(n)
            sv.extend([x / ns, y / ns])
        for u in self.uavs:
            ux, uy = self.get_pos_from_node(u)
            d = math.sqrt((cx - ux)**2 + (cy - uy)**2 + self.config.H**2) / ns
            sv.append(d)
        for l in range(len(self.uavs)):
            usage_bits = self._cache_usage_bits(l)
            sv.append(min(usage_bits / max(self.C_cache_bits, 1.0), 1.0))

        nearest_mbs = _nearest_in_range_rsu(car, self.rsus) if car is not None else None
        if car is None or nearest_mbs is None:
            sv.extend([1.0, 0.0])
        else:
            d_mbs = min(dist_2d(car, nearest_mbs) / ns, 1.0)
            mbs_users = _count_cars_in_mbs_range(self.cars, nearest_mbs)
            mbs_load = min(float(mbs_users) / float(_effective_mbs_capacity(self.config)), 1.0)
            sv.extend([d_mbs, mbs_load])

        p_fz = float(self._zipf_probs_fz[self.f_req, self.z_req])
        sv.append(p_fz)

        # z_req one-hot encoded — agent biết dứt khoát xe đang yêu cầu mức bitrate nào
        z_req_onehot = [0.0] * self.Z
        if 0 <= int(self.z_req) < self.Z:
            z_req_onehot[int(self.z_req)] = 1.0
        sv.extend(z_req_onehot)

        return np.array(sv, dtype=np.float32)

    # ------------------------------------------------------------------
    def _decode_action(self, action_idx: int):
        """
        Decode action index thành (uav_idx, z_cached, cache_dec).

        Encoding (UAV-only, theo Paper Xie et al.):
            a = uav_idx + L * (z_cached + Z * cache_dec)

        Returns:
            (uav_idx, z_cached, cache_dec)
        """
        a = int(action_idx) % max(self._uav_action_size, 1)  # clamp an toàn
        L = max(int(self.num_offload_targets), 1)
        Z = max(int(self.num_bitrates), 1)

        uav_idx   = a % L
        t         = a // L
        z_cached  = int(t % Z)
        cache_dec = int(t // Z)
        return int(uav_idx), int(z_cached), int(cache_dec)

    def encode_action(self, uav_idx: int, z_cached: int, cache: int) -> int:
        """
        Encode UAV-tier action (3-chiều) thành action index.
        FIX: thêm chiều z_cached so với phiên bản cũ 2-chiều.

        Args:
            uav_idx  : chỉ số UAV [0..L-1]
            z_cached : mức bitrate cần cache [0..Z-1]
            cache    : 0=no_cache, 1=cache

        Returns:
            action index tương ứng với encoding:
            a = uav_idx + L * (z_cached + Z * cache)
        """
        L = max(int(self.num_offload_targets), 1)
        Z = max(int(self.num_bitrates), 1)
        return int(uav_idx) + L * (int(z_cached) + Z * int(cache))

    # ------------------------------------------------------------------
    def _new_request(self):
        """Chọn request theo joint popularity p_{f,z} — gọi TRƯỚC get_state()."""
        if not self.cars:
            self.requesting_car = None
        else:
            # Theo Paper: mọi xe đều phục vụ qua UAV, chọn xe bất kỳ
            valid_cars = []
            for car in self.cars:
                covered_by_uav = any(dist_2d(car, uav) <= float(UAV_RANGE) for uav in self.uavs)
                if covered_by_uav:
                    valid_cars.append(car)
            self.requesting_car = random.choice(valid_cars) if valid_cars else random.choice(self.cars)
        joint_flat = self._zipf_probs_fz.reshape(-1)
        req_idx = int(np.random.choice(joint_flat.size, p=joint_flat))
        self.f_req = int(req_idx // self.Z)
        self.z_req = int(req_idx % self.Z)

    def step(self, action_idx: int):
        requesting_car = self.requesting_car
        f = int(self.f_req)
        z_req = int(self.z_req)

        uav_idx, z_cached_action, cache_dec = self._decode_action(action_idx)

        # ------------------------------
        # UAV tier only (theo Paper Xie et al.)
        # ------------------------------
        uav_idx     = int(max(0, min(int(uav_idx), len(self.uavs) - 1)))
        target_node = self.uavs[uav_idx]

        out_of_range = False
        if requesting_car is not None:
            if dist_2d(requesting_car, target_node) > UAV_RANGE:
                out_of_range = True

        # ── Cache placement (FIX: clamp z_cached_action vào [0, Z-1]) ──────────
        z_cached_action = int(max(0, min(int(z_cached_action), self.Z - 1)))

        if cache_dec == 1 and not out_of_range:
            self.Y[uav_idx, f, z_cached_action] = 1
            # Capacity enforcement: random eviction
            while self._cache_usage_bits(uav_idx) > self.C_cache_bits:
                ones = np.argwhere(self.Y[uav_idx] == 1)
                if ones.size == 0:
                    break
                j = random.randrange(len(ones))
                ff, zz = int(ones[j, 0]), int(ones[j, 1])
                self.Y[uav_idx, ff, zz] = 0

        # ── Xác định cache_mode sau khi đã cập nhật Y ───────────────────────
        if int(self.Y[uav_idx, f, z_req]) == 1:
            cache_mode = 1          # Scenario 1: direct hit (Eq.10)
            z_cached   = z_req
        else:
            z_plus = None
            for z2 in range(z_req + 1, self.Z):
                if int(self.Y[uav_idx, f, z2]) == 1:
                    z_plus = z2
                    break
            if z_plus is not None:
                cache_mode = 2      # Scenario 2: transcoding hit (Eq.11)
                z_cached   = z_plus
            else:
                cache_mode = 0      # Scenario 3: cache miss — MBS→UAV→Car (Eq.12)
                z_cached   = z_req

        # ── Out-of-range: phạt nặng, không fallback (theo Paper) ─────────────
        if out_of_range:
            oor_penalty = 0.0
            if cache_dec == 1 and z_cached_action != z_req:
                oor_penalty = -10.0
            cost = float(getattr(self.config, 'no_uav_penalty', 1000.0))
            reward = -math.log1p(cost) + oor_penalty
            served_decision = {
                'tier': 'uav', 'uav_idx': int(uav_idx),
                'offload_name': getattr(target_node, 'name', f'uav{uav_idx + 1}'),
                'cache': 0, 'f_req': f, 'z_req': z_req, 'z_cached': int(z_cached_action),
                'popularity': float(self._zipf_probs_fz[f, z_req]),
            }
            self._last_actual_uav_idx = uav_idx
            self._last_served_request = dict(served_decision)
            self._new_request()
            return self.get_state(), reward, False, {
                'raw_delay': cost, 'actual_uav_idx': uav_idx,
                'out_of_range': True, 'fallback': False,
                'disconnected': False, 'decision': served_decision,
            }

        # ── In-range: tính delay theo 3 kịch bản Eq(10-12) ───────────────────
        num_users_on_uav = _count_cars_in_uav_range(self.cars, target_node)
        cost = calculate_total_cost(
            requesting_car, target_node, self.config,
            cache_mode=cache_mode,
            all_uavs=self.uavs,
            z_req=z_req,
            z_cached=z_cached,
            num_uavs=len(self.uavs),
            rsus=self.rsus,
            num_users_per_uav=num_users_on_uav,
        )

        # ── Reward Shaping ────────────────────────────────────────────────────
        action_bonus = 0.0
        if cache_dec == 1:
            if z_cached_action != z_req:
                action_bonus = -10.0  # Phạt cache sai bitrate

        base_reward = -math.log1p(cost)
        reward = base_reward + action_bonus
        served_decision = {
            'tier': 'uav',
            'uav_idx': int(uav_idx),
            'offload_name': getattr(target_node, 'name', f'uav{uav_idx + 1}'),
            'cache': int(cache_dec),
            'f_req': f,
            'z_req': z_req,
            'z_cached': int(z_cached_action),
            'popularity': float(self._zipf_probs_fz[f, z_req]),
        }
        self._last_actual_uav_idx = uav_idx
        self._last_served_request = dict(served_decision)
        self._new_request()
        return self.get_state(), reward, False, {
            'raw_delay': cost,
            'actual_uav_idx': uav_idx,
            'out_of_range': False,
            'fallback': False,
            'disconnected': False,
            'decision': served_decision,
        }

    # ------------------------------------------------------------------
    def reset(self):
        self.Y[:] = 0
        self.f_req          = 0
        self.z_req          = 0
        self.requesting_car = self.cars[0] if self.cars else None
        self._last_actual_uav_idx = None
        self._last_served_request = None
        self._new_request()
        return self.get_state()

    # ------------------------------------------------------------------
    def get_action_components(self, action_idx: int):
        served = getattr(self, '_last_served_request', None)
        if served is not None:
            return {
                'tier':         str(served.get('tier', 'uav')),
                'uav_idx':      int(served.get('uav_idx', -1)),
                'offload_name': str(served.get('offload_name', 'none')),
                'cache':        int(served.get('cache', 0)),
                'f_req':        int(served.get('f_req', self.f_req)),
                'z_req':        int(served.get('z_req', self.z_req)),
                'z_cached':     int(served.get('z_cached', -1)),
                'popularity':   float(
                    served.get('popularity', self._zipf_probs_fz[self.f_req, self.z_req])
                ),
            }

        uav_idx, z_cached, cache = self._decode_action(action_idx)
        if 0 <= uav_idx < len(self.uavs):
            offload_name = getattr(self.uavs[int(uav_idx)], 'name', f'uav{int(uav_idx)+1}')
        else:
            offload_name = 'none'

        return {
            'tier':         'uav',
            'uav_idx':      int(uav_idx),
            'offload_name': offload_name,
            'cache':        int(cache),
            'z_cached':     int(z_cached),
            'f_req':        int(self.f_req),
            'z_req':        int(self.z_req),
            'popularity':   float(self._zipf_probs_fz[self.f_req, self.z_req]),
        }
