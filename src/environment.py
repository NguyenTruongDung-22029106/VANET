#!/usr/bin/env python3
"""
environment.py — VanetEnvironment: Môi trường học tăng cường theo mô hình Xie et al.

State  : động theo số UAV, gồm xe requesting + UAV features + popularity
Action : (#UAV × #bitrate × cache_decision) — chỉ UAV, không có MBS tier
         Encoding: a = uav_idx + L*(z_cached + Z*cache_dec)
Reward : QoE ABR (utility − rebuffer − switch); D từ Eq.(10–12) Xie et al.
         Buffer: B' = B + T − D với T = abr_segment_duration_s (thời lượng segment, cố định);
         s_{f,z} = R_z·T trong models (cùng nguồn với delay).
"""
import math
import random
import numpy as np
from models import calculate_total_cost, _chunk_size_bits, abr_bitrate_kbps_list
from helpers import (
    get_node_xy, dist_2d,
)
from constants import UAV_RANGE


def _compute_zipf_joint_probs(num_videos: int, num_bitrates: int, gamma: float) -> np.ndarray:
    """Zipf trên toàn bộ cặp (f,z), shape (F, Z)."""
    F = max(int(num_videos), 1)
    Z = max(int(num_bitrates), 1)
    ranks = np.arange(1, F * Z + 1, dtype=np.float64)
    raw = ranks ** (-gamma)
    return (raw / raw.sum()).reshape(F, Z)


def _count_cars_in_uav_range(cars, uav_node):
    """Estimate runtime users served by a UAV by coverage count."""
    if uav_node is None:
        return 0
    c = 0
    for car in cars:
        if dist_2d(car, uav_node) <= float(UAV_RANGE):
            c += 1
    return c


def _get_abr_labels(config, Z: int):
    labels = getattr(config, 'abr_bitrate_labels', None)
    if labels is None:
        labels = tuple(f"z{z}" for z in range(Z))
    try:
        labels = list(labels)
    except Exception:
        labels = [f"z{z}" for z in range(Z)]
    if len(labels) < Z:
        while len(labels) < Z:
            labels.append(str(labels[-1]) if labels else f"z{len(labels)}")
    return [str(x) for x in labels[:Z]]


def _abr_utility(bitrate_kbps: float, mode: str = 'log') -> float:
    b = max(float(bitrate_kbps), 1.0)
    if str(mode).lower().strip() == 'linear':
        return b / 1000.0
    # log utility (Pensieve-like)
    return math.log(b / 1000.0)


# ============================================================
# VanetEnvironment
# ============================================================

class VanetEnvironment:
    """
    Môi trường VANET-UAV-SDN cho D3QN.

    Action encoding (3-chiều):
        a = uav_idx + L * (z_cached + Z * cache_dec)

    Ví dụ với L=5 UAV, Z=4 bitrates:
        action_size = 5*4*2 = 40

    Khởi tạo:
        VanetEnvironment(config, stations, aps, uavs_list)
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

        # Kích thước state = 2 + 7L + 2Z + 3 (L = số UAV, Z = num_bitrates)
        self.state_size = (
            2 +                      # requesting user position
            len(self.uavs) * 2 +      # UAV positions
            len(self.uavs) +          # distance user→each UAV
            len(self.uavs) +          # cache fullness per UAV
            len(self.uavs) * 3 +      # Cache hit, UAV load, transcoding avail per UAV
            1 +                       # popularity p_{f,z}
            self.Z +                  # z_req one-hot
            1 +                       # buffer_norm
            self.Z +                  # last_bitrate one-hot
            1                         # throughput EWMA norm
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
        # Slot bookkeeping (paper-like: tất cả cars cùng request trong 1 slot/round)
        self.K = len(self.cars)
        self._slot_id = 0
        self._slot_car_idx = 0
        self._slot_f_req = np.zeros(self.K, dtype=np.int64)
        self._slot_z_req = np.zeros(self.K, dtype=np.int64)
        self._slot_delay_i = np.zeros(self.K, dtype=np.float64)
        self._slot_reward_i = np.zeros(self.K, dtype=np.float64)

        # Current request for `requesting_car` (set bởi `_new_request()` / theo từng substep)
        self.f_req = 0
        self.z_req = 0

        self.requesting_car = self.cars[0] if self.cars else None

        self._norm_scale = float(getattr(config, 'plot_max', 400))
        self._last_actual_uav_idx = None
        self._last_served_request = None

        # ── ABR/QoE runtime state (per requesting stream) ─────────────────
        self._abr_segment_s = float(getattr(config, 'abr_segment_duration_s', 2.0))
        self._abr_max_buf_s = float(getattr(config, 'abr_max_buffer_s', 30.0))
        self._abr_init_buf_s = float(getattr(config, 'abr_init_buffer_s', 2.0))
        # Per-car ABR state: mỗi xe có buffer/switch history riêng
        self._buffer_s = np.full(self.K, float(self._abr_init_buf_s), dtype=np.float64)
        self._last_bitrate_idx = np.zeros(self.K, dtype=np.int64)
        self._tp_ewma_kbps = np.zeros(self.K, dtype=np.float64)
        self._abr_tp_alpha = float(getattr(config, 'abr_tp_ewma_alpha', 0.9))
        self._abr_bitrates_kbps = abr_bitrate_kbps_list(config, self.Z)
        self._abr_labels = _get_abr_labels(config, self.Z)

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
        
        # --- NEW STATE FEATURES ---
        for l in range(len(self.uavs)):
            # 1. Cache hit indicator (0.0 or 1.0)
            sv.append(float(self.Y[l, self.f_req, self.z_req]))
            
            # 2. UAV Load (normalized 0 to 1)
            uav_node = self.uavs[l]
            uav_users = _count_cars_in_uav_range(self.cars, uav_node)
            uav_capacity = float(getattr(self.config, 'M', 30.0))
            sv.append(min(uav_users / max(uav_capacity, 1.0), 1.0))
            
            # 3. Transcoding availability
            can_transcode = 0.0
            for z2 in range(self.z_req + 1, self.Z):
                if int(self.Y[l, self.f_req, z2]) == 1:
                    can_transcode = 1.0
                    break
            sv.append(can_transcode)
        # --------------------------


        # --- END NEW STATE FEATURES ---

        p_fz = float(self._zipf_probs_fz[self.f_req, self.z_req])
        sv.append(p_fz)

        # z_req one-hot encoded — agent biết dứt khoát xe đang yêu cầu mức bitrate nào
        z_req_onehot = [0.0] * self.Z
        if 0 <= int(self.z_req) < self.Z:
            z_req_onehot[int(self.z_req)] = 1.0
        sv.extend(z_req_onehot)

        # --- ABR/QoE state features (per-car) ---
        cur_i = int(getattr(self, '_slot_car_idx', 0))
        cur_i = max(0, min(cur_i, max(self.K - 1, 0)))

        buf_norm = float(self._buffer_s[cur_i]) / max(float(self._abr_max_buf_s), 1e-6)
        sv.append(min(max(buf_norm, 0.0), 1.0))

        last_onehot = [0.0] * self.Z
        li = int(self._last_bitrate_idx[cur_i])
        if 0 <= li < self.Z:
            last_onehot[li] = 1.0
        sv.extend(last_onehot)

        max_br = float(max(self._abr_bitrates_kbps) if self._abr_bitrates_kbps else 1.0)
        tp_norm = float(self._tp_ewma_kbps[cur_i]) / max(max_br, 1.0)
        sv.append(min(max(tp_norm, 0.0), 1.0))
        # ------------------------------

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
        """
        Start a new slot (paper-like):
          - pre-generate request (f,z) cho TOÀN BỘ cars trong slot
          - reset current substep về car idx = 0
        """
        self._slot_id = int(self._slot_id) + 1
        self._slot_car_idx = 0

        if not self.cars or self.K <= 0:
            self.requesting_car = None
            self.f_req = 0
            self.z_req = 0
            return

        joint_flat = self._zipf_probs_fz.reshape(-1)
        req_idx = np.random.choice(joint_flat.size, size=self.K, p=joint_flat).astype(np.int64)
        self._slot_f_req = (req_idx // self.Z).astype(np.int64)
        self._slot_z_req = (req_idx % self.Z).astype(np.int64)

        # Reset accumulators for aggregation at end-of-slot
        self._slot_delay_i[:] = 0.0
        self._slot_reward_i[:] = 0.0

        self.requesting_car = self.cars[0]
        self.f_req = int(self._slot_f_req[0])
        self.z_req = int(self._slot_z_req[0])

    def step(self, action_idx: int):
        car_i = int(getattr(self, '_slot_car_idx', 0))
        slot_id = int(getattr(self, '_slot_id', 0))

        requesting_car = self.requesting_car
        f = int(self.f_req)
        z_req = int(self.z_req)

        uav_idx, z_cached_action, cache_dec = self._decode_action(action_idx)

        # ------------------------------
        # UAV tier only (theo Paper Xie et al.)
        # ------------------------------
        uav_idx     = int(max(0, min(int(uav_idx), len(self.uavs) - 1)))
        target_node = self.uavs[uav_idx]

        # ── out_of_range: logging ─────────────
        distance_2d = 0.0
        if requesting_car is not None:
            distance_2d = float(dist_2d(requesting_car, target_node))
        # NOTE: out_of_range flag is not used for penalty anymore

        # ── Cache placement (FIX: clamp z_cached_action vào [0, Z-1]) ──────────
        z_cached_action = int(max(0, min(int(z_cached_action), self.Z - 1)))

        if cache_dec == 1:
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

        # ── Physics-based delay: mọi UAV đều tính delay thật ─────────────────
        # UAV xa → SINR thấp → rate thấp → delay tự động tăng cao
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

        # ── ABR QoE reward (per car) ─────────────────────────────────────
        bitrate_kbps = float(self._abr_bitrates_kbps[z_req] if 0 <= z_req < len(self._abr_bitrates_kbps) else 1000.0)
        last_bitrate_i = int(self._last_bitrate_idx[car_i]) if self.K > 0 else 0
        last_kbps = float(self._abr_bitrates_kbps[last_bitrate_i] if 0 <= last_bitrate_i < len(self._abr_bitrates_kbps) else bitrate_kbps)
        bitrate_label = str(self._abr_labels[z_req] if 0 <= z_req < len(self._abr_labels) else str(z_req))

        download_time_s = float(cost)
        s_bits = float(self.chunk_sizes[z_req]) if 0 <= z_req < len(self.chunk_sizes) else _chunk_size_bits(z_req, self.config)
        playback_s = float(self._abr_segment_s)
        # Buffer: B' = B + T − D (T = thời lượng segment; s_{f,z} = R_z·T trong Xie ABR)
        buf_next = float(self._buffer_s[car_i]) + playback_s - download_time_s
        rebuffer_s = 0.0
        if buf_next < 0.0:
            rebuffer_s = -buf_next
            buf_next = 0.0
        buf_next = min(buf_next, float(self._abr_max_buf_s))

        # Effective throughput (kbps): s_{f,z} [kbit] / D — khớp cùng s, D với Eq.(10–12)
        tp_kbps = 0.0
        if download_time_s > 1e-6:
            tp_kbps = (s_bits / 1000.0) / download_time_s
        self._tp_ewma_kbps[car_i] = (
            self._abr_tp_alpha * float(self._tp_ewma_kbps[car_i])
            + (1.0 - self._abr_tp_alpha) * float(tp_kbps)
        )

        # QoE terms
        util = _abr_utility(bitrate_kbps, mode=getattr(self.config, 'abr_utility', 'log'))
        rebuf_pen = float(getattr(self.config, 'abr_rebuffer_penalty', 4.3))
        sw_pen = float(getattr(self.config, 'abr_switch_penalty', 1.0))
        switch_mag = abs((bitrate_kbps - last_kbps) / 1000.0)  # Mbps delta
        abr_reward = util - rebuf_pen * float(rebuffer_s) - sw_pen * float(switch_mag)
        reward = abr_reward

        served_decision = {
            'tier': 'uav',
            'uav_idx': int(uav_idx),
            'offload_name': getattr(target_node, 'name', f'uav{uav_idx + 1}'),
            'cache': int(cache_dec),
            'cache_mode': int(cache_mode),
            'f_req': f,
            'z_req': z_req,
            'z_cached': int(z_cached_action),
            'bitrate': float(bitrate_kbps),
            'bitrate_label': bitrate_label,
            'popularity': float(self._zipf_probs_fz[f, z_req]),
        }
        self._last_actual_uav_idx = uav_idx
        self._last_served_request = dict(served_decision)
        # Persist ABR state for this car
        self._buffer_s[car_i] = float(buf_next)
        self._last_bitrate_idx[car_i] = int(z_req)

        # Slot accumulators for aggregation/logging
        if self.K > 0:
            self._slot_delay_i[car_i] = float(cost)
            self._slot_reward_i[car_i] = float(reward)

        # Advance to next substep (next car) or next slot
        self._slot_car_idx = car_i + 1
        done_slot = self._slot_car_idx >= self.K

        info = {
            'raw_delay': float(cost),
            'playback_chunk_s': float(playback_s),
            'distance_2d': float(distance_2d),
            'actual_uav_idx': int(uav_idx),
            'buffer_s': float(self._buffer_s[car_i]),
            'rebuffer_s': float(rebuffer_s),
            'bitrate_kbps': float(bitrate_kbps),
            'bitrate_label': bitrate_label,
            'bitrate_switch_mbps': float(switch_mag),
            'qoe_reward': float(abr_reward),
            'slot_id': slot_id,
            'substep_idx': int(car_i),
            'fallback': False,
            'disconnected': False,
            'decision': served_decision,
        }

        if not done_slot:
            # Update current request for the next car
            self.requesting_car = self.cars[self._slot_car_idx]
            self.f_req = int(self._slot_f_req[self._slot_car_idx])
            self.z_req = int(self._slot_z_req[self._slot_car_idx])
            next_state = self.get_state()
            return next_state, reward, False, info

        # End of slot: aggregate across cars (xấp xỉ D_tot xu hướng)
        delay_slot_mean = float(np.mean(self._slot_delay_i)) if self.K > 0 else 0.0
        reward_slot_mean = float(np.mean(self._slot_reward_i)) if self.K > 0 else 0.0
        info['delay_slot_mean'] = delay_slot_mean
        info['reward_slot_mean'] = reward_slot_mean

        # Start the next slot: generates requests for all cars
        self._new_request()
        next_state = self.get_state()
        return next_state, reward, False, info

    # ------------------------------------------------------------------
    def reset(self):
        self.Y[:] = 0
        self._last_actual_uav_idx = None
        self._last_served_request = None
        self._abr_segment_s = float(getattr(self.config, 'abr_segment_duration_s', 2.0))
        self._slot_id = 0
        self._slot_car_idx = 0
        self.f_req = 0
        self.z_req = 0

        # Reset per-car ABR state
        if self.K > 0:
            self._buffer_s[:] = float(self._abr_init_buf_s)
            self._last_bitrate_idx[:] = 0
            self._tp_ewma_kbps[:] = 0.0
            self.requesting_car = self.cars[0]
        else:
            self.requesting_car = None

        # Start first slot (generate requests for all cars)
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
                'cache_mode':   int(served.get('cache_mode', -1)),
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
