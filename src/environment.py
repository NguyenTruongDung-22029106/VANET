#!/usr/bin/env python3
"""
environment.py — VanetEnvironment: Môi trường học tăng cường cho VANET-UAV-SDN.

State  : 37 chiều — vị trí xe/UAV/RSU, CPU load, cache mode, video popularity
Action : 20 chiều — offload(5) × bitrate(2) × cache(2)
Reward : -delay (giây), delay tính từ models.calculate_total_cost()

FIXES TRONG FILE NÀY:
  - Bug 1 Fix: requesting_car = random.choice(self.cars) thay vì cars[0]
  - Bug 2 Fix: popularity = Zipf p_f động thay vì cứng 0.7
  - Bug 2b Fix: lưu self.requesting_car sau mỗi step()
  - Bug 5 Fix: get_action_components() thêm offload_name + bitrate_label
  - from_config Fix: thêm classmethod from_config() — tạo dummy nodes từ config
    → ryu_app.py không cần _create_stub_nodes nữa
"""
import math
import random
import numpy as np
from types import SimpleNamespace
from models import calculate_total_cost


# ============================================================
# Helper: tính phân phối Zipf
# ============================================================

def _compute_zipf_probs(num_videos: int, gamma: float) -> np.ndarray:
    ranks = np.arange(1, num_videos + 1, dtype=np.float64)
    raw   = ranks ** (-gamma)
    return raw / raw.sum()


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
    r_tri    = plot_max / 4.0
    cos30    = math.cos(math.pi / 6)
    sin30    = math.sin(math.pi / 6)
    uav_verts = [
        (cx,                   cy + r_tri),
        (cx - r_tri * cos30,   cy - r_tri * sin30),
        (cx + r_tri * cos30,   cy - r_tri * sin30),
    ]

    num_cars = getattr(config, 'cars', 10)
    num_uavs = getattr(config, 'uavs', 3)
    num_rsus = getattr(config, 'rsus', 1)

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
            params={'position': uav_verts[(i - 1) % 3]}
        )
        for i in range(1, num_uavs + 1)
    ]
    stations = cars + uavs
    return stations, rsus, uavs


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

    NUM_BITRATES   = 2
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

        self.OFFLOAD_LOCAL         = 0
        self.OFFLOAD_UAV_START_IDX = 1
        self.OFFLOAD_RSU_START_IDX = 1 + len(self.uavs)
        self.num_offload_targets   = 1 + len(self.uavs) + len(self.rsus)
        self.num_bitrates          = self.NUM_BITRATES
        self.num_cache_actions     = self.NUM_CACHE_ACTS

        self.action_size = (
            self.num_offload_targets
            * self.num_bitrates
            * self.num_cache_actions
        )

        self.state_size = (
            len(self.cars)  * 2 +
            len(self.uavs)  * 2 +
            len(self.rsus)  * 2 +
            len(self.uavs) + len(self.rsus) +   # CPU load
            len(self.uavs) + len(self.rsus) +   # cache status
            1                                    # video popularity
        )

        self.cache_state   = {n.name: 0   for n in (self.uavs + self.rsus)}
        self.cache_bitrate = {n.name: 0   for n in (self.uavs + self.rsus)}
        self.cpu_load      = {n.name: 0.0 for n in (self.uavs + self.rsus)}
        self._cpu_decay    = 0.05
        self._cpu_per_req  = 0.1

        self.num_videos    = int(getattr(config, 'num_videos',    100))
        self.zipf_exponent = float(getattr(config, 'zipf_exponent', 0.7))
        self._zipf_probs   = _compute_zipf_probs(self.num_videos, self.zipf_exponent)
        self.f_req         = 0

        self.requesting_car = self.cars[0] if self.cars else None

        # Fix 5: calibrate M theo actual users per UAV
        if self.uavs:
            actual_M = max(1, len(self.cars) // len(self.uavs))
            self.config.M = actual_M

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
        if hasattr(node, 'params') and 'position' in node.params:
            p = node.params['position']
            return float(p[0]), float(p[1])
        return 0.0, 0.0

    def get_state(self):
        sv = []
        for n in self.cars + self.uavs + self.rsus:
            sv.extend(self.get_pos_from_node(n))
        for n in self.uavs + self.rsus:
            sv.append(float(self.cpu_load.get(n.name, 0.0)))
        for n in self.uavs + self.rsus:
            sv.append(float(self.cache_state.get(n.name, 0)))
        p_f = float(self._zipf_probs[self.f_req])
        sv.append(p_f)
        return np.array(sv, dtype=np.float32)

    # ------------------------------------------------------------------
    def _decode_action(self, action_idx: int):
        a         = int(action_idx)
        offload   = a % self.num_offload_targets
        remainder = a // self.num_offload_targets
        z_req     = remainder % self.num_bitrates
        cache_dec = remainder // self.num_bitrates
        return offload, z_req, int(cache_dec)

    def encode_action(self, offload: int, z_req: int, cache: int) -> int:
        return offload + self.num_offload_targets * (z_req + self.num_bitrates * cache)

    # ------------------------------------------------------------------
    def step(self, action_idx: int):
        requesting_car      = random.choice(self.cars)
        self.requesting_car = requesting_car

        self.f_req = int(np.random.choice(self.num_videos, p=self._zipf_probs))

        offload_choice, z_req, cache_dec = self._decode_action(action_idx)

        if offload_choice == self.OFFLOAD_LOCAL:
            target_node = requesting_car
        elif self.OFFLOAD_UAV_START_IDX <= offload_choice < self.OFFLOAD_RSU_START_IDX:
            target_node = self.uavs[offload_choice - self.OFFLOAD_UAV_START_IDX]
        else:
            target_node = self.rsus[offload_choice - self.OFFLOAD_RSU_START_IDX]

        node_name         = getattr(target_node, 'name', '')
        cache_mode_before = int(self.cache_state.get(node_name, 0))
        z_cached_before   = int(self.cache_bitrate.get(node_name, 0))

        if target_node is not requesting_car and node_name in self.cache_state:
            if cache_dec == 1:
                self.cache_bitrate[node_name] = z_req
                self.cache_state[node_name]   = 1 if z_req == 0 else 2
            else:
                self.cache_state[node_name]   = 0
                self.cache_bitrate[node_name] = 0

        for n in self.uavs + self.rsus:
            self.cpu_load[n.name] = max(0.0, self.cpu_load[n.name] - self._cpu_decay)
        if target_node is not requesting_car and node_name in self.cpu_load:
            self.cpu_load[node_name] = min(1.0, self.cpu_load[node_name] + self._cpu_per_req)

        cpu  = self.cpu_load.get(node_name, 0.0)
        cost = calculate_total_cost(
            requesting_car, target_node, self.config,
            cache_mode = cache_mode_before,
            all_uavs   = self.uavs,
            z_req      = z_req,
            z_cached   = z_cached_before if cache_mode_before == 2 else z_req,
            num_uavs   = len(self.uavs),
            cpu_load   = cpu,
        )
        return self.get_state(), -cost, False, {}

    # ------------------------------------------------------------------
    def reset(self):
        self.cache_state   = {n.name: 0   for n in (self.uavs + self.rsus)}
        self.cache_bitrate = {n.name: 0   for n in (self.uavs + self.rsus)}
        self.cpu_load      = {n.name: 0.0 for n in (self.uavs + self.rsus)}
        self.f_req         = 0
        self.requesting_car = self.cars[0] if self.cars else None
        return self.get_state()

    # ------------------------------------------------------------------
    def get_action_components(self, action_idx: int):
        offload, z_req, cache = self._decode_action(action_idx)

        if offload == self.OFFLOAD_LOCAL:
            offload_name = 'Local'
        elif self.OFFLOAD_UAV_START_IDX <= offload < self.OFFLOAD_RSU_START_IDX:
            uav_idx = offload - self.OFFLOAD_UAV_START_IDX
            offload_name = getattr(
                self.uavs[uav_idx] if uav_idx < len(self.uavs) else None,
                'name', f'uav{uav_idx+1}'
            )
        else:
            rsu_idx = offload - self.OFFLOAD_RSU_START_IDX
            offload_name = getattr(
                self.rsus[rsu_idx] if rsu_idx < len(self.rsus) else None,
                'name', f'rsu{rsu_idx+1}'
            )

        _labels      = ['low(480p)', 'high(1080p)']
        bitrate_label = _labels[z_req] if z_req < len(_labels) else f'z{z_req}'

        return {
            'offload_idx':   offload,
            'offload_name':  offload_name,
            'bitrate':       z_req,
            'bitrate_label': bitrate_label,
            'cache':         cache,
            'f_req':         self.f_req,
            'popularity':    float(self._zipf_probs[self.f_req]),
        }