#!/usr/bin/env python3
"""Runtime QoE telemetry store and P.1203-like scoring helpers."""

import csv
import math
import os
import threading
import time
import uuid


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _to_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _to_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return int(default)


class QoESessionStore:
    """Collects per-segment player events and exposes synchronized consumption."""

    def __init__(self, config=None):
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._pending = {}
        self._history = []
        self._sessions = {}

        self.max_history = int(getattr(config, 'qoe_store_max_history', 50000)) if config else 50000
        self.segment_playback_sec = max(float(getattr(config, 'chunk_playback_sec', 2.0)), 1e-6) if config else 2.0
        self.max_quality_index = max(int(getattr(config, 'qoe_max_quality_index', 3)), 1) if config else 3

        # P.1203-like coefficients (practical composite approximation)
        self.alpha_startup = float(getattr(config, 'qoe_alpha_startup', 0.18)) if config else 0.18
        self.alpha_rebuffer_time = float(getattr(config, 'qoe_alpha_rebuffer_time', 0.35)) if config else 0.35
        self.alpha_rebuffer_count = float(getattr(config, 'qoe_alpha_rebuffer_count', 0.20)) if config else 0.20
        self.beta_quality = float(getattr(config, 'qoe_beta_quality', 0.8)) if config else 0.8
        self.beta_switch = float(getattr(config, 'qoe_beta_switch', 0.25)) if config else 0.25

        log_dir = 'results'
        if config is not None:
            raw_log_dir = getattr(config, 'log_dir', 'results')
            if os.path.isabs(raw_log_dir):
                log_dir = raw_log_dir
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                log_dir = os.path.abspath(os.path.join(base_dir, raw_log_dir))
        self.log_csv_path = os.path.join(log_dir, 'qoe_runtime_events.csv')
        self._init_csv()

    def _init_csv(self):
        try:
            os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)
            with open(self.log_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'ts', 'session_id', 'car_name', 'segment_idx',
                    'startup_sec', 'rebuffer_sec', 'rebuffer_count',
                    'quality_index', 'switch_magnitude', 'segment_download_sec', 'buffer_sec',
                    'qoe_mos', 'qoe_cost', 'delay_term', 'stall_norm', 'smooth_norm',
                ])
        except Exception:
            pass

    def _append_csv(self, metric):
        try:
            with open(self.log_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{time.time():.6f}",
                    metric.get('session_id', ''),
                    metric.get('car_name', ''),
                    metric.get('segment_idx', 0),
                    f"{float(metric.get('startup_sec', 0.0)):.6f}",
                    f"{float(metric.get('rebuffer_sec', 0.0)):.6f}",
                    f"{float(metric.get('rebuffer_count', 0.0)):.6f}",
                    f"{float(metric.get('quality_index', 0.0)):.6f}",
                    f"{float(metric.get('switch_magnitude', 0.0)):.6f}",
                    f"{float(metric.get('segment_download_sec', 0.0)):.6f}",
                    f"{float(metric.get('buffer_sec', 0.0)):.6f}",
                    f"{float(metric.get('qoe_mos', 0.0)):.6f}",
                    f"{float(metric.get('qoe_cost', 0.0)):.6f}",
                    f"{float(metric.get('delay_term', 0.0)):.6f}",
                    f"{float(metric.get('stall_norm', 0.0)):.6f}",
                    f"{float(metric.get('smooth_norm', 0.0)):.6f}",
                ])
        except Exception:
            pass

    def start_session(self, session_id=None, car_name=None, meta=None):
        sid = str(session_id).strip() if session_id is not None else ''
        if not sid:
            sid = str(uuid.uuid4())
        with self._lock:
            self._sessions[sid] = {
                'created_at': time.time(),
                'car_name': str(car_name or ''),
                'meta': dict(meta or {}),
            }
        return sid

    def reset(self):
        with self._lock:
            self._pending.clear()
            self._history.clear()
            self._sessions.clear()

    def p1203_like_score(
        self,
        startup_sec,
        rebuffer_sec,
        rebuffer_count,
        quality_index,
        switch_magnitude,
    ):
        q_norm = _clip(_to_float(quality_index) / float(self.max_quality_index), 0.0, 1.0)
        switch_norm = _clip(abs(_to_float(switch_magnitude)) / float(self.max_quality_index), 0.0, 1.0)

        startup_pen = self.alpha_startup * _clip(_to_float(startup_sec), 0.0, 15.0)
        rebuffer_pen = (
            self.alpha_rebuffer_time * _clip(_to_float(rebuffer_sec), 0.0, 30.0)
            + self.alpha_rebuffer_count * _clip(_to_float(rebuffer_count), 0.0, 10.0)
        )
        quality_gain = self.beta_quality * q_norm
        switch_pen = self.beta_switch * switch_norm

        mos = 5.0 - startup_pen - rebuffer_pen - switch_pen + quality_gain
        mos = _clip(mos, 1.0, 5.0)
        qoe_cost = _clip(5.0 - mos, 0.0, 25.0)
        return float(mos), float(qoe_cost)

    def ingest_segment_event(self, event):
        sid = str(event.get('session_id', 'default')).strip() or 'default'
        car_name = str(event.get('car_name') or event.get('car') or '').strip()
        seg_idx = _to_int(event.get('segment_idx', 0), 0)

        startup_sec = _to_float(event.get('startup_sec', 0.0), 0.0)
        rebuffer_sec = _to_float(event.get('rebuffer_sec', 0.0), 0.0)
        rebuffer_count = _to_float(event.get('rebuffer_count', 0.0), 0.0)
        quality_index = _to_float(event.get('quality_index', 0.0), 0.0)
        switch_magnitude = _to_float(event.get('switch_magnitude', 0.0), 0.0)
        segment_download_sec = _to_float(event.get('segment_download_sec', 0.0), 0.0)
        buffer_sec = _to_float(event.get('buffer_sec', 0.0), 0.0)

        delay_term = math.log1p(max(segment_download_sec, 0.0))
        stall_norm = max(0.0, (segment_download_sec - self.segment_playback_sec) / self.segment_playback_sec)
        smooth_norm = _clip(abs(switch_magnitude) / max(float(self.max_quality_index), 1.0), 0.0, 1.0)

        qoe_mos, qoe_cost = self.p1203_like_score(
            startup_sec=startup_sec,
            rebuffer_sec=rebuffer_sec,
            rebuffer_count=rebuffer_count,
            quality_index=quality_index,
            switch_magnitude=switch_magnitude,
        )

        metric = {
            'session_id': sid,
            'car_name': car_name,
            'segment_idx': int(seg_idx),
            'startup_sec': float(startup_sec),
            'rebuffer_sec': float(rebuffer_sec),
            'rebuffer_count': float(rebuffer_count),
            'quality_index': float(quality_index),
            'switch_magnitude': float(switch_magnitude),
            'segment_download_sec': float(segment_download_sec),
            'buffer_sec': float(buffer_sec),
            'qoe_mos': float(qoe_mos),
            'qoe_cost': float(qoe_cost),
            'delay_term': float(delay_term),
            'stall_norm': float(stall_norm),
            'smooth_norm': float(smooth_norm),
        }

        key = (sid, car_name, int(seg_idx))
        with self._cond:
            self._pending[key] = dict(metric)
            self._history.append(dict(metric))
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
            self._cond.notify_all()

        self._append_csv(metric)
        return metric

    def wait_and_consume(self, session_id, car_name, segment_idx, timeout_s=0.75):
        sid = str(session_id or 'default').strip() or 'default'
        ckey = str(car_name or '').strip()
        seg = int(segment_idx)
        key = (sid, ckey, seg)

        timeout = max(float(timeout_s), 0.0)
        t_end = time.time() + timeout

        with self._cond:
            while key not in self._pending:
                remain = t_end - time.time()
                if remain <= 0.0:
                    return None
                self._cond.wait(timeout=remain)
            return self._pending.pop(key, None)

    def proxy_from_delay(self, delay, prev_delay, playback_sec, w_delay, w_stall, w_smooth):
        d = max(_to_float(delay), 0.0)
        prev = max(_to_float(prev_delay, d), 0.0)
        play = max(_to_float(playback_sec, 2.0), 1e-6)

        delay_term = math.log1p(d)
        stall_norm = max(0.0, (d - play) / play)
        smooth_norm = abs(d - prev) / play
        qoe_cost = max(
            _to_float(w_delay, 1.0) * delay_term
            + _to_float(w_stall, 1.5) * stall_norm
            + _to_float(w_smooth, 0.5) * smooth_norm,
            0.0,
        )
        qoe_mos = _clip(5.0 - qoe_cost, 1.0, 5.0)

        return {
            'startup_sec': 0.0,
            'rebuffer_sec': max(0.0, d - play),
            'rebuffer_count': 1.0 if d > play else 0.0,
            'quality_index': 0.0,
            'switch_magnitude': 0.0,
            'segment_download_sec': d,
            'buffer_sec': 0.0,
            'qoe_mos': qoe_mos,
            'qoe_cost': max(0.0, 5.0 - qoe_mos),
            'delay_term': delay_term,
            'stall_norm': stall_norm,
            'smooth_norm': smooth_norm,
        }
