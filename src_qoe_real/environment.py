#!/usr/bin/env python3
"""
environment.py — VanetEnvironment for UAV-only serving with real-QoE reward.

State  : dynamic by number of UAVs + request/context + QoE history
Action : (#UAV × #bitrate × cache_decision), encoding:
         a = uav_idx + L * (z_cached + Z * cache_dec)
Reward : real-QoE-first (P.1203-like from player telemetry) + OOR penalty
"""

import math
import random
import numpy as np
from models import calculate_total_cost
from helpers import get_node_xy, dist_2d
from constants import UAV_RANGE


def _compute_zipf_joint_probs(num_videos: int, num_bitrates: int, gamma: float) -> np.ndarray:
    """Zipf over (f,z) pairs, shape (F, Z)."""
    F = max(int(num_videos), 1)
    Z = max(int(num_bitrates), 1)
    ranks = np.arange(1, F * Z + 1, dtype=np.float64)
    raw = ranks ** (-gamma)
    return (raw / raw.sum()).reshape(F, Z)


def _chunk_size_bits(z_idx: int, config) -> float:
    """s_{f,z}: chunk size by bitrate z (linear by z+1)."""
    s0_bits = float(getattr(config, 'chunk_size_MB', 8.0)) * 8e6
    return s0_bits * (int(z_idx) + 1)


def _count_cars_in_uav_range(cars, uav_node):
    """Estimate users served by a UAV by coverage count."""
    if uav_node is None:
        return 0
    c = 0
    for car in cars:
        if dist_2d(car, uav_node) <= float(UAV_RANGE):
            c += 1
    return c


class VanetEnvironment:
    """VANET-UAV environment used by D3QN through REST endpoints."""

    def __init__(self, config, stations, aps=None, uavs_list=None):
        self.config = config

        self.rsus = list(aps) if aps else []
        self.uavs = list(uavs_list) if uavs_list else []

        rsu_names = {n.name for n in self.rsus}
        uav_names = {n.name for n in self.uavs}
        self.cars = [
            n for n in stations
            if getattr(n, 'name', '') not in rsu_names
            and getattr(n, 'name', '') not in uav_names
        ]

        self.num_bitrates = int(getattr(config, 'num_bitrates', 4))
        self.num_cache_actions = int(getattr(config, 'num_cache_acts', 2))
        self.num_offload_targets = len(self.uavs)
        self.Z = self.num_bitrates

        self._uav_action_size = self.num_offload_targets * self.num_bitrates * self.num_cache_actions
        self.action_size = max(1, self._uav_action_size)

        # car pos + uav pos + car->uav dist + cache fill + cache-hit/load/transcode +
        # prev_delay/stall + last mos/rebuffer/switch + popularity + z_req one-hot
        self.state_size = (
            2 +
            len(self.uavs) * 2 +
            len(self.uavs) +
            len(self.uavs) +
            len(self.uavs) * 3 +
            5 +
            1 +
            self.Z
        )

        self.cache_uav_MB = float(getattr(config, 'cache_uav_MB', 300))
        self.C_cache_bits = self.cache_uav_MB * 8e6
        self.chunk_sizes = np.array([_chunk_size_bits(z, config) for z in range(self.Z)], dtype=np.float64)
        self.Y = np.zeros((len(self.uavs), int(getattr(config, 'num_videos', 100)), self.Z), dtype=np.int8)

        self.num_videos = int(getattr(config, 'num_videos', 100))
        self.zipf_exponent = float(getattr(config, 'zipf_exponent', 0.7))
        self._zipf_probs_fz = _compute_zipf_joint_probs(self.num_videos, self.Z, self.zipf_exponent)

        self.f_req = 0
        self.z_req = 0
        self.requesting_car = self.cars[0] if self.cars else None

        self._norm_scale = float(getattr(config, 'plot_max', 400))
        self.oor_penalty_alpha = float(getattr(config, 'oor_penalty_alpha', 2.0))
        self.oor_penalty_beta = float(getattr(config, 'oor_penalty_beta', 2.0))
        self.oor_penalty_cap = float(getattr(config, 'oor_penalty_cap', 5.0))

        self.objective_profile = str(getattr(config, 'objective_profile', 'qoe_real'))
        self.chunk_playback_sec = max(float(getattr(config, 'chunk_playback_sec', 2.0)), 1e-6)
        self.reward_clip_min = float(getattr(config, 'reward_clip_min', -25.0))
        self.reward_clip_max = float(getattr(config, 'reward_clip_max', 0.0))

        # QoE fallback/proxy weights when real segment event is missing
        self.qoe_w_delay = float(getattr(config, 'qoe_w_delay', 1.0))
        self.qoe_w_stall = float(getattr(config, 'qoe_w_stall', 1.5))
        self.qoe_w_smooth = float(getattr(config, 'qoe_w_smooth', 0.5))
        self.qoe_event_timeout_s = float(getattr(config, 'qoe_event_timeout_s', 0.75))
        self.qoe_strict_event = bool(getattr(config, 'qoe_strict_event', False))
        self.qoe_missing_penalty = float(getattr(config, 'qoe_missing_event_penalty', 3.0))

        self.qoe_session_id = str(getattr(config, 'qoe_default_session_id', 'default'))
        self.qoe_focus_car_name = str(getattr(config, 'qoe_focus_car_name', '')).strip()
        self._qoe_store = None

        self._prev_delay_by_car = {}
        self._segment_idx_by_car = {}
        self._last_mos_by_car = {}
        self._last_rebuffer_by_car = {}
        self._last_switch_by_car = {}

        self._last_actual_uav_idx = None
        self._last_served_request = None

    @staticmethod
    def get_pos_from_node(node):
        return get_node_xy(node)

    @staticmethod
    def _car_key(car):
        if car is None:
            return '__none__'
        return str(getattr(car, 'name', f'car_{id(car)}'))

    def attach_qoe_store(self, qoe_store):
        self._qoe_store = qoe_store

    def set_qoe_session(self, session_id):
        if session_id is None:
            return
        sid = str(session_id).strip()
        if sid:
            self.qoe_session_id = sid

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
            d = math.sqrt((cx - ux) ** 2 + (cy - uy) ** 2 + self.config.H ** 2) / ns
            sv.append(d)

        for l in range(len(self.uavs)):
            usage_bits = self._cache_usage_bits(l)
            sv.append(min(usage_bits / max(self.C_cache_bits, 1.0), 1.0))

        for l in range(len(self.uavs)):
            sv.append(float(self.Y[l, self.f_req, self.z_req]))
            uav_node = self.uavs[l]
            uav_users = _count_cars_in_uav_range(self.cars, uav_node)
            uav_capacity = float(getattr(self.config, 'M', 30))
            sv.append(min(uav_users / max(uav_capacity, 1.0), 1.0))

            can_transcode = 0.0
            for z2 in range(self.z_req + 1, self.Z):
                if int(self.Y[l, self.f_req, z2]) == 1:
                    can_transcode = 1.0
                    break
            sv.append(can_transcode)

        play_t = self.chunk_playback_sec
        car_key = self._car_key(car)
        prev_delay = float(self._prev_delay_by_car.get(car_key, 0.0))
        prev_delay_norm = min(max(prev_delay / play_t, 0.0), 1.0)
        prev_stall_norm = min(max((prev_delay - play_t) / play_t, 0.0), 1.0)

        last_mos = float(self._last_mos_by_car.get(car_key, 3.0))
        last_rebuffer = float(self._last_rebuffer_by_car.get(car_key, 0.0))
        last_switch = float(self._last_switch_by_car.get(car_key, 0.0))
        mos_norm = min(max((last_mos - 1.0) / 4.0, 0.0), 1.0)
        rebuf_norm = min(max(last_rebuffer / play_t, 0.0), 1.0)
        switch_norm = min(max(last_switch / max(self.Z - 1, 1), 0.0), 1.0)
        sv.extend([prev_delay_norm, prev_stall_norm, mos_norm, rebuf_norm, switch_norm])

        p_fz = float(self._zipf_probs_fz[self.f_req, self.z_req])
        sv.append(p_fz)

        z_req_onehot = [0.0] * self.Z
        if 0 <= int(self.z_req) < self.Z:
            z_req_onehot[int(self.z_req)] = 1.0
        sv.extend(z_req_onehot)

        return np.array(sv, dtype=np.float32)

    def _decode_action(self, action_idx: int):
        a = int(action_idx) % max(self._uav_action_size, 1)
        L = max(int(self.num_offload_targets), 1)
        Z = max(int(self.num_bitrates), 1)

        uav_idx = a % L
        t = a // L
        z_cached = int(t % Z)
        cache_dec = int(t // Z)
        return int(uav_idx), int(z_cached), int(cache_dec)

    def encode_action(self, uav_idx: int, z_cached: int, cache: int) -> int:
        L = max(int(self.num_offload_targets), 1)
        Z = max(int(self.num_bitrates), 1)
        return int(uav_idx) + L * (int(z_cached) + Z * int(cache))

    def _new_request(self):
        if not self.cars:
            self.requesting_car = None
        else:
            # Real-QoE demo mode: keep requesting car aligned with player telemetry stream.
            focus = self.qoe_focus_car_name
            if focus:
                match = next((c for c in self.cars if str(getattr(c, 'name', '')) == focus), None)
                if match is not None:
                    self.requesting_car = match
                else:
                    valid_cars = []
                    for car in self.cars:
                        covered_by_uav = any(dist_2d(car, uav) <= float(UAV_RANGE) for uav in self.uavs)
                        if covered_by_uav:
                            valid_cars.append(car)
                    self.requesting_car = random.choice(valid_cars) if valid_cars else random.choice(self.cars)
            else:
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

    def _qoe_from_telemetry_or_proxy(self, car_key, segment_idx, delay):
        metric = None
        source = 'proxy'
        missing = False

        if self._qoe_store is not None:
            metric = self._qoe_store.wait_and_consume(
                session_id=self.qoe_session_id,
                car_name=car_key,
                segment_idx=segment_idx,
                timeout_s=self.qoe_event_timeout_s,
            )

        if metric is None and self._qoe_store is not None:
            metric = self._qoe_store.proxy_from_delay(
                delay=float(delay),
                prev_delay=float(self._prev_delay_by_car.get(car_key, float(delay))),
                playback_sec=self.chunk_playback_sec,
                w_delay=self.qoe_w_delay,
                w_stall=self.qoe_w_stall,
                w_smooth=self.qoe_w_smooth,
            )
            source = 'proxy'
            missing = True
        elif metric is not None:
            source = 'player'
        else:
            # Safety path when no store is attached.
            d = float(delay)
            delay_term = math.log1p(max(d, 0.0))
            stall_norm = max(0.0, (d - self.chunk_playback_sec) / self.chunk_playback_sec)
            smooth_norm = abs(d - float(self._prev_delay_by_car.get(car_key, d))) / self.chunk_playback_sec
            qoe_cost = self.qoe_w_delay * delay_term + self.qoe_w_stall * stall_norm + self.qoe_w_smooth * smooth_norm
            qoe_mos = max(1.0, min(5.0, 5.0 - qoe_cost))
            metric = {
                'startup_sec': 0.0,
                'rebuffer_sec': max(0.0, d - self.chunk_playback_sec),
                'rebuffer_count': 1.0 if d > self.chunk_playback_sec else 0.0,
                'quality_index': float(self.z_req),
                'switch_magnitude': 0.0,
                'segment_download_sec': d,
                'buffer_sec': 0.0,
                'qoe_mos': qoe_mos,
                'qoe_cost': max(0.0, 5.0 - qoe_mos),
                'delay_term': delay_term,
                'stall_norm': stall_norm,
                'smooth_norm': smooth_norm,
            }
            missing = True

        if missing and self.qoe_strict_event:
            metric = dict(metric)
            metric['qoe_cost'] = float(metric.get('qoe_cost', 0.0)) + self.qoe_missing_penalty
            metric['qoe_mos'] = max(1.0, float(metric.get('qoe_mos', 5.0)) - self.qoe_missing_penalty)

        return metric, source, missing

    def step(self, action_idx: int):
        requesting_car = self.requesting_car
        f = int(self.f_req)
        z_req = int(self.z_req)

        uav_idx, z_cached_action, cache_dec = self._decode_action(action_idx)

        uav_idx = int(max(0, min(int(uav_idx), len(self.uavs) - 1)))
        target_node = self.uavs[uav_idx]

        distance_2d = 0.0
        out_of_range = False
        if requesting_car is not None:
            distance_2d = float(dist_2d(requesting_car, target_node))
            out_of_range = distance_2d > float(UAV_RANGE)

        z_cached_action = int(max(0, min(int(z_cached_action), self.Z - 1)))

        if cache_dec == 1:
            self.Y[uav_idx, f, z_cached_action] = 1
            while self._cache_usage_bits(uav_idx) > self.C_cache_bits:
                ones = np.argwhere(self.Y[uav_idx] == 1)
                if ones.size == 0:
                    break
                j = random.randrange(len(ones))
                ff, zz = int(ones[j, 0]), int(ones[j, 1])
                self.Y[uav_idx, ff, zz] = 0

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

        car_key = self._car_key(requesting_car)
        prev_delay = float(self._prev_delay_by_car.get(car_key, float(cost)))
        seg_idx = int(self._segment_idx_by_car.get(car_key, 0))
        qoe_metric, qoe_source, missing_event = self._qoe_from_telemetry_or_proxy(car_key, seg_idx, cost)

        qoe_cost = max(float(qoe_metric.get('qoe_cost', 0.0)), 0.0)
        qoe_mos = float(qoe_metric.get('qoe_mos', max(1.0, 5.0 - qoe_cost)))
        base_reward = -qoe_cost

        overshoot = max(0.0, (distance_2d / float(UAV_RANGE)) - 1.0)
        oor_penalty = min(self.oor_penalty_alpha * (overshoot ** self.oor_penalty_beta), self.oor_penalty_cap)

        reward_unclipped = base_reward - oor_penalty
        reward = float(np.clip(reward_unclipped, self.reward_clip_min, self.reward_clip_max))

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

        self._prev_delay_by_car[car_key] = float(cost)
        self._segment_idx_by_car[car_key] = seg_idx + 1
        self._last_mos_by_car[car_key] = qoe_mos
        self._last_rebuffer_by_car[car_key] = float(qoe_metric.get('rebuffer_sec', 0.0))
        self._last_switch_by_car[car_key] = float(qoe_metric.get('switch_magnitude', 0.0))

        self._last_actual_uav_idx = uav_idx
        self._last_served_request = dict(served_decision)

        self._new_request()
        info = {
            'raw_delay': float(cost),
            'actual_uav_idx': int(uav_idx),
            'distance_2d': float(distance_2d),
            'overshoot': float(overshoot),
            'oor_penalty': float(oor_penalty),
            'base_reward': float(base_reward),
            'reward_unclipped': float(reward_unclipped),
            'reward_final': float(reward),
            'out_of_range': bool(out_of_range),
            'fallback': bool(missing_event),
            'disconnected': False,
            'decision': served_decision,
            'objective_profile': self.objective_profile,
            'qoe_source': qoe_source,
            'qoe_event_missing': bool(missing_event),
            'qoe_session_id': self.qoe_session_id,
            'segment_idx': int(seg_idx),
            'qoe_mos': float(qoe_mos),
            'qoe_cost': float(qoe_cost),
            'startup_sec': float(qoe_metric.get('startup_sec', 0.0)),
            'rebuffer_sec': float(qoe_metric.get('rebuffer_sec', 0.0)),
            'rebuffer_count': float(qoe_metric.get('rebuffer_count', 0.0)),
            'quality_index': float(qoe_metric.get('quality_index', z_req)),
            'switch_magnitude': float(qoe_metric.get('switch_magnitude', 0.0)),
            'segment_download_sec': float(qoe_metric.get('segment_download_sec', cost)),
            'buffer_sec': float(qoe_metric.get('buffer_sec', 0.0)),
            'delay_term': float(qoe_metric.get('delay_term', math.log1p(max(float(cost), 0.0)))),
            'stall_norm': float(qoe_metric.get('stall_norm', max(0.0, (float(cost) - self.chunk_playback_sec) / self.chunk_playback_sec))),
            'smooth_norm': float(qoe_metric.get('smooth_norm', abs(float(cost) - prev_delay) / self.chunk_playback_sec)),
        }
        return self.get_state(), reward, False, info

    def reset(self):
        self.Y[:] = 0
        self.f_req = 0
        self.z_req = 0
        self.requesting_car = self.cars[0] if self.cars else None

        self._prev_delay_by_car.clear()
        self._segment_idx_by_car.clear()
        self._last_mos_by_car.clear()
        self._last_rebuffer_by_car.clear()
        self._last_switch_by_car.clear()

        self._last_actual_uav_idx = None
        self._last_served_request = None

        self._new_request()
        return self.get_state()

    def get_action_components(self, action_idx: int):
        served = getattr(self, '_last_served_request', None)
        if served is not None:
            return {
                'tier': str(served.get('tier', 'uav')),
                'uav_idx': int(served.get('uav_idx', -1)),
                'offload_name': str(served.get('offload_name', 'none')),
                'cache': int(served.get('cache', 0)),
                'f_req': int(served.get('f_req', self.f_req)),
                'z_req': int(served.get('z_req', self.z_req)),
                'z_cached': int(served.get('z_cached', -1)),
                'popularity': float(served.get('popularity', self._zipf_probs_fz[self.f_req, self.z_req])),
            }

        uav_idx, z_cached, cache = self._decode_action(action_idx)
        if 0 <= uav_idx < len(self.uavs):
            offload_name = getattr(self.uavs[int(uav_idx)], 'name', f'uav{int(uav_idx)+1}')
        else:
            offload_name = 'none'

        return {
            'tier': 'uav',
            'uav_idx': int(uav_idx),
            'offload_name': offload_name,
            'cache': int(cache),
            'z_cached': int(z_cached),
            'f_req': int(self.f_req),
            'z_req': int(self.z_req),
            'popularity': float(self._zipf_probs_fz[self.f_req, self.z_req]),
        }
