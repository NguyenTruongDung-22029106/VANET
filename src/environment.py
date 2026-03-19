#!/usr/bin/env python3
"""
environment.py — VanetEnvironment: Môi trường học tăng cường theo mô hình Xie et al.

State  : động theo số UAV, gồm xe requesting + UAV features + MBS context + popularity
Action : (#UAV × cache_decision) + 1 action MBS tier
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


def _count_cars_in_mbs_range(cars, mbs_node):
    """Estimate runtime users served by an MBS/RSU by coverage count."""
    if mbs_node is None:
        return 0
    c = 0
    for car in cars:
        if dist_2d(car, mbs_node) <= float(MBS_RANGE):
            c += 1
    return c


# ============================================================
# VanetEnvironment
# ============================================================

class VanetEnvironment:
    """
    Môi trường VANET-UAV-SDN cho D3QN.

    Khởi tạo:
        VanetEnvironment(config, stations, aps, uavs_list)  ← dùng trong main_thesis.py
        VanetEnvironment.from_config(config)                ← dùng trong ryu_app.py
    """

    NUM_BITRATES   = 4
    NUM_CACHE_ACTS = 2

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

        # Hướng 1 (tier decision):
        #   - UAV tier: chọn UAV l và cache/no-cache cho request hiện tại
        #   - MBS/RSU tier: phục vụ trực tiếp, bỏ qua cache_dec
        self.num_bitrates      = self.NUM_BITRATES
        self.num_cache_actions = self.NUM_CACHE_ACTS
        self.num_offload_targets = len(self.uavs)  # chỉ UAVs
        self._uav_action_size = self.num_offload_targets * self.num_cache_actions
        # +1 extra action for MBS/RSU tier
        self.action_size = max(1, self._uav_action_size + 1)

        # State: car pos + UAV features + nearest-MBS context + popularity
        self.state_size = (
            2 +                      # requesting user position
            len(self.uavs) * 2 +      # UAV positions
            len(self.uavs) +          # distance user→each UAV
            len(self.uavs) +          # cache fullness per UAV
            1 +                       # distance user→nearest MBS/RSU
            1 +                       # normalized load around nearest MBS/RSU
            1                         # popularity of requested chunk
        )

        # Paper caching: y_{l,f,z} per UAV, per chunk, per bitrate
        self.cache_uav_MB = float(getattr(config, 'cache_uav_MB', 300))
        self.C_cache_bits = self.cache_uav_MB * 8e6
        self.Z = self.NUM_BITRATES
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
        Dùng cho ryu_app.py: không cần _create_stub_nodes nữa.

        Ví dụ:
            env = VanetEnvironment.from_config(get_config())
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
        # Cache fullness per UAV (0..1)
        for l in range(len(self.uavs)):
            usage_bits = self._cache_usage_bits(l)
            sv.append(min(usage_bits / max(self.C_cache_bits, 1.0), 1.0))

        # Nearest-MBS context so agent can learn balanced UAV/MBS tier decisions.
        nearest_mbs = _nearest_in_range_rsu(car, self.rsus) if car is not None else None
        if car is None or nearest_mbs is None:
            sv.extend([1.0, 0.0])
        else:
            d_mbs = min(dist_2d(car, nearest_mbs) / ns, 1.0)
            mbs_users = _count_cars_in_mbs_range(self.cars, nearest_mbs)
            mbs_load = min(float(mbs_users) / max(float(getattr(self.config, 'M', 30)), 1.0), 1.0)
            sv.extend([d_mbs, mbs_load])

        p_fz = float(self._zipf_probs_fz[self.f_req, self.z_req])
        sv.append(p_fz)
        return np.array(sv, dtype=np.float32)

    # ------------------------------------------------------------------
    def _decode_action(self, action_idx: int):
        a = int(action_idx)
        L = max(int(self.num_offload_targets), 1)
        mbs_action_idx = int(self._uav_action_size)
        if a == mbs_action_idx:
            return 'mbs', -1, 0
        # UAV tier
        uav_idx = a % L
        cache_dec = a // L
        return 'uav', int(uav_idx), int(cache_dec)

    def encode_action(self, uav_idx: int, cache: int) -> int:
        """Encode a UAV-tier action (not used for MBS tier)."""
        L = max(int(self.num_offload_targets), 1)
        return int(uav_idx) + L * int(cache)

    # ------------------------------------------------------------------
    def _new_request(self):
        """Chọn request theo joint popularity p_{f,z} — gọi TRƯỚC get_state()."""
        # Practical sampling for mixed UAV/MBS deployment:
        # choose cars covered by at least one serving tier (UAV or MBS).
        if not self.cars:
            self.requesting_car = None
        else:
            valid_cars = []
            for car in self.cars:
                covered_by_uav = any(dist_2d(car, uav) <= float(UAV_RANGE) for uav in self.uavs)
                covered_by_mbs = _nearest_in_range_rsu(car, self.rsus) is not None
                if covered_by_uav or covered_by_mbs:
                    valid_cars.append(car)
            # Nếu hiếm khi không có xe hợp lệ thì fallback về toàn bộ xe để không crash.
            self.requesting_car = random.choice(valid_cars) if valid_cars else random.choice(self.cars)
        joint_flat = self._zipf_probs_fz.reshape(-1)
        req_idx = int(np.random.choice(joint_flat.size, p=joint_flat))
        self.f_req = int(req_idx // self.Z)
        self.z_req = int(req_idx % self.Z)

    def step(self, action_idx: int):
        requesting_car = self.requesting_car
        f = int(self.f_req)
        z_req = int(self.z_req)

        tier, uav_idx, cache_dec = self._decode_action(action_idx)

        # ------------------------------
        # MBS/RSU tier
        # ------------------------------
        if tier == 'mbs' or not self.uavs:
            mbs_node = None
            if requesting_car is not None:
                mbs_node = _nearest_in_range_rsu(requesting_car, self.rsus)

            if requesting_car is not None and mbs_node is not None:
                # Practical extension using V2B delay path (Chen multi-path final tier).
                users_bs = _count_cars_in_mbs_range(self.cars, mbs_node)
                cost = calculate_mbs_only_delay(
                    requesting_car,
                    mbs_node,
                    self.config,
                    z_req=z_req,
                    num_users_bs=users_bs,
                )
                out_of_range = False
                offload_name = getattr(mbs_node, 'name', 'mbs')
            else:
                # No reachable MBS/RSU: penalize invalid serving action.
                cost = float(getattr(self.config, 'no_uav_penalty', 1000.0))
                out_of_range = True
                offload_name = 'none'

            reward = -math.log1p(cost)
            served_decision = {
                'tier': 'mbs',
                'uav_idx': -1,
                'offload_name': offload_name,
                'cache': 0,
                'f_req': f,
                'z_req': z_req,
                'popularity': float(self._zipf_probs_fz[f, z_req]),
            }
            self._last_actual_uav_idx = -1
            self._last_served_request = dict(served_decision)
            self._new_request()
            return self.get_state(), reward, False, {
                'raw_delay': cost,
                'actual_uav_idx': -1,
                'out_of_range': out_of_range,
                'decision': served_decision,
            }

        # ------------------------------
        # UAV tier
        # ------------------------------
        uav_idx = int(max(0, min(int(uav_idx), len(self.uavs) - 1)))
        target_node = self.uavs[uav_idx]

        # Coverage-range check used for SDN flow gating + MBS fallback.
        out_of_range = False
        if requesting_car is not None:
            if dist_2d(requesting_car, target_node) > UAV_RANGE:
                out_of_range = True

        # Determine cache mode per paper (Eq10-12)
        if int(self.Y[uav_idx, f, z_req]) == 1:
            cache_mode = 1
            z_cached = z_req
        else:
            z_plus = None
            for z2 in range(z_req + 1, self.Z):
                if int(self.Y[uav_idx, f, z2]) == 1:
                    z_plus = z2
                    break
            if z_plus is not None:
                cache_mode = 2
                z_cached = z_plus
            else:
                cache_mode = 0
                z_cached = z_req

        # Apply cache decision for this request (online).
        # If UAV is out-of-coverage, it won't actually serve this request.
        if cache_dec == 1 and not out_of_range:
            self.Y[uav_idx, f, z_req] = 1
            # Capacity enforcement: random eviction (Algorithm 2 spirit)
            while self._cache_usage_bits(uav_idx) > self.C_cache_bits:
                ones = np.argwhere(self.Y[uav_idx] == 1)
                if ones.size == 0:
                    break
                j = random.randrange(len(ones))
                ff, zz = int(ones[j, 0]), int(ones[j, 1])
                self.Y[uav_idx, ff, zz] = 0

        if out_of_range:
            # Follow Xie 2022: user is not served by UAV if out-of-coverage;
            # do not fallback to MBS. Penalize instead.
            cost = float(getattr(self.config, 'no_uav_penalty', 1000.0))
        else:
            cost = calculate_total_cost(
                requesting_car, target_node, self.config,
                cache_mode=cache_mode,
                all_uavs=self.uavs,
                z_req=z_req,
                z_cached=z_cached,
                num_uavs=len(self.uavs),
                rsus=self.rsus,
                num_users_per_uav=None,
            )

        reward = -math.log1p(cost)
        served_decision = {
            'tier': 'uav',
            'uav_idx': int(uav_idx),
            'offload_name': getattr(target_node, 'name', f'uav{uav_idx + 1}'),
            'cache': int(cache_dec),
            'f_req': f,
            'z_req': z_req,
            'popularity': float(self._zipf_probs_fz[f, z_req]),
        }
        self._last_actual_uav_idx = uav_idx
        self._last_served_request = dict(served_decision)
        self._new_request()
        return self.get_state(), reward, False, {
            'raw_delay': cost,
            'actual_uav_idx': uav_idx,
            'out_of_range': out_of_range,
            'decision': served_decision,
        }

    # ------------------------------------------------------------------
    def reset(self):
        self.Y[:] = 0
        self.f_req         = 0
        self.z_req         = 0
        self.requesting_car = self.cars[0] if self.cars else None
        self._last_actual_uav_idx = None
        self._last_served_request = None
        self._new_request()
        return self.get_state()

    # ------------------------------------------------------------------
    def get_action_components(self, action_idx: int):
        # Ưu tiên trả snapshot của request vừa phục vụ (đúng với state/action tại step trước)
        served = getattr(self, '_last_served_request', None)
        if served is not None:
            return {
                'tier': str(served.get('tier', 'uav')),
                'uav_idx': int(served.get('uav_idx', -1)),
                'offload_name': str(served.get('offload_name', 'none')),
                'cache': int(served.get('cache', 0)),
                'f_req': int(served.get('f_req', self.f_req)),
                'z_req': int(served.get('z_req', self.z_req)),
                'popularity': float(
                    served.get('popularity', self._zipf_probs_fz[self.f_req, self.z_req])
                ),
            }

        tier, uav_idx, cache = self._decode_action(action_idx)
        if tier == 'mbs':
            offload_name = 'mbs'
        else:
            if 0 <= uav_idx < len(self.uavs):
                offload_name = getattr(self.uavs[int(uav_idx)], 'name', f'uav{int(uav_idx)+1}')
            else:
                offload_name = 'none'

        return {
            'tier': str(tier),
            'uav_idx': int(uav_idx),
            'offload_name': offload_name,
            'cache': int(cache),
            'f_req': int(self.f_req),
            'z_req': int(self.z_req),
            'popularity': float(self._zipf_probs_fz[self.f_req, self.z_req]),
        }
