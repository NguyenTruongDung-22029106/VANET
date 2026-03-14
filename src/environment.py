#!/usr/bin/env python3
"""
environment.py — VanetEnvironment: Môi trường học tăng cường cho VANET-UAV-SDN.

State  : 37 chiều — vị trí xe/UAV/RSU, CPU load, cache mode, video popularity
Action : 20 chiều — offload(5) × bitrate(2) × cache(2)
Reward : -delay (giây), delay tính từ models.calculate_total_cost()

FIXES TRONG FILE NÀY:
  - Bug 1 Fix (cũ): requesting_car = random.choice(self.cars) thay vì cars[0]
  - Bug 2 Fix (cũ): popularity = Zipf p_f động thay vì cứng 0.7
  - Bug 2b Fix (mới): lưu self.requesting_car sau mỗi step()
    -> control_layer.get_forced_ap_name() đọc được xe đang request
  - Bug 5 Fix (mới): get_action_components() thêm offload_name + bitrate_label
    -> ryu_app.py dùng decision['offload_name'] và decision['bitrate_label']
    không còn KeyError
"""
import random
import numpy as np
from models import calculate_total_cost


# ============================================================
# Helper: tính phân phối Zipf
# ============================================================

def _compute_zipf_probs(num_videos: int, gamma: float) -> np.ndarray:
    """
    Tính xác suất Zipf cho F video.
    p_f = f^{-gamma} / Sum_{j=1}^{F} j^{-gamma}   (Eq.1 Chen et al.)
    """
    ranks = np.arange(1, num_videos + 1, dtype=np.float64)
    raw   = ranks ** (-gamma)
    return raw / raw.sum()


# ============================================================
# VanetEnvironment
# ============================================================

class VanetEnvironment:
    """
    Moi truong VANET-UAV-SDN cho D3QN.

    Tham so khoi tao:
        config    : SimpleNamespace tu get_config()
        stations  : list tat ca node (cars + uavs + rsus)
        aps       : list RSU/MBS nodes
        uavs_list : list UAV nodes
    """

    # Constants
    NUM_BITRATES   = 2   # 0=480p, 1=1080p
    NUM_CACHE_ACTS = 2   # 0=no cache, 1=cache

    def __init__(self, config, stations, aps=None, uavs_list=None):
        self.config = config

        # Phan loai nodes
        self.rsus = list(aps)       if aps        else []
        self.uavs = list(uavs_list) if uavs_list  else []

        rsu_names = {n.name for n in self.rsus}
        uav_names = {n.name for n in self.uavs}
        self.cars = [
            n for n in stations
            if getattr(n, 'name', '') not in rsu_names
            and getattr(n, 'name', '') not in uav_names
        ]

        # --- Action space constants ---
        self.OFFLOAD_LOCAL         = 0
        self.OFFLOAD_UAV_START_IDX = 1
        self.OFFLOAD_RSU_START_IDX = 1 + len(self.uavs)

        self.num_offload_targets = 1 + len(self.uavs) + len(self.rsus)
        self.num_bitrates        = self.NUM_BITRATES
        self.num_cache_actions   = self.NUM_CACHE_ACTS

        # action_idx = offload + num_offload * (bitrate + num_bitrates * cache)
        self.action_size = (
            self.num_offload_targets
            * self.num_bitrates
            * self.num_cache_actions
        )

        # --- State space ---
        self.state_size = (
            len(self.cars)  * 2 +               # positions x,y
            len(self.uavs)  * 2 +
            len(self.rsus)  * 2 +
            len(self.uavs) + len(self.rsus) +   # CPU load
            len(self.uavs) + len(self.rsus) +   # cache status (mode 0/1/2)
            1                                    # video popularity (Zipf p_f)
        )

        # Cache state: mode per node (0=miss, 1=hit, 2=transcoding)
        self.cache_state   = {n.name: 0   for n in (self.uavs + self.rsus)}
        self.cache_bitrate = {n.name: 0   for n in (self.uavs + self.rsus)}

        # CPU load dong (Fix 3)
        self.cpu_load     = {n.name: 0.0 for n in (self.uavs + self.rsus)}
        self._cpu_decay   = 0.05   # giam moi step
        self._cpu_per_req = 0.1    # tang khi co request

        # -------------------------------------------------------
        # Bug 2 Fix: Zipf popularity dong
        # -------------------------------------------------------
        self.num_videos    = int(getattr(config, 'num_videos',    100))
        self.zipf_exponent = float(getattr(config, 'zipf_exponent', 0.7))
        self._zipf_probs   = _compute_zipf_probs(self.num_videos, self.zipf_exponent)
        self.f_req         = 0

        # -------------------------------------------------------
        # Bug 2b Fix: luu requesting_car de control_layer doc duoc
        # -------------------------------------------------------
        self.requesting_car = self.cars[0] if self.cars else None

        # -------------------------------------------------------
        # Fix 5: Calibrate config.M theo actual users per UAV
        #
        # Vấn đề: Paper dùng M=30 là MAXIMUM capacity của UAV.
        # Nhưng _rate_uav_user() tính r_lk = (B/M)*log2(1+SINR),
        # khiến bandwidth/user bị chia cho 30 dù chỉ có ~3 xe/UAV.
        # Kết quả: r_lk thấp hơn thực tế 10 lần, UAV delay ~10s thay vì ~1s,
        # agent luôn chọn Local vì Local chỉ có 4.63s.
        #
        # Fix: dùng số xe thực tế / số UAV làm M hiệu quả.
        # cars=10, uavs=3 → M_actual = 3 → r_lk tăng 10x → D1 ≈ 1s << Local ≈ 13s
        # -------------------------------------------------------
        if self.uavs:
            actual_M = max(1, len(self.cars) // len(self.uavs))
            self.config.M = actual_M  # override Paper's M=30

    # ------------------------------------------------------------------
    @staticmethod
    def get_pos_from_node(node):
        if hasattr(node, 'params') and 'position' in node.params:
            p = node.params['position']
            return float(p[0]), float(p[1])
        return 0.0, 0.0

    def get_state(self):
        """
        State vector day du: pos + cpu_load + cache_mode + popularity.
        Bug 2 Fix: popularity = Zipf p_{f_req} tinh dong.
        """
        sv = []
        for n in self.cars + self.uavs + self.rsus:
            sv.extend(self.get_pos_from_node(n))
        for n in self.uavs + self.rsus:
            sv.append(float(self.cpu_load.get(n.name, 0.0)))
        for n in self.uavs + self.rsus:
            sv.append(float(self.cache_state.get(n.name, 0)))

        # Bug 2 Fix: Zipf p_f dong thay vi 0.7 cung
        p_f = float(self._zipf_probs[self.f_req])
        sv.append(p_f)

        return np.array(sv, dtype=np.float32)

    # ------------------------------------------------------------------
    def _decode_action(self, action_idx: int):
        """
        Giai ma action_idx thanh (offload_target, z_req, cache_decision).
        Encoding:
          action_idx = offload + num_offload * (z + num_bitrates * cache)
        """
        a         = int(action_idx)
        offload   = a % self.num_offload_targets
        remainder = a // self.num_offload_targets
        z_req     = remainder % self.num_bitrates
        cache_dec = remainder // self.num_bitrates
        return offload, z_req, int(cache_dec)

    def encode_action(self, offload: int, z_req: int, cache: int) -> int:
        """Encode nguoc lai (dung cho test/debug)."""
        return offload + self.num_offload_targets * (z_req + self.num_bitrates * cache)

    # ------------------------------------------------------------------
    def step(self, action_idx: int):
        """
        One step:
          1. Bug 1 Fix: chon requesting_car ngau nhien (khong co dinh cars[0])
          2. Bug 2 Fix: chon f_req ngau nhien theo phan phoi Zipf
          2b. Bug 2b Fix: luu self.requesting_car de control_layer doc duoc
          3. Decode action -> (offload, z_req, cache)
          4. Update cache state theo bitrate agent chon
          5. Update CPU load
          6. Tinh delay (Eq 10-12 + queuing penalty)
          7. reward = -delay
        """
        # Bug 1 Fix: random requesting_car
        requesting_car = random.choice(self.cars)

        # Bug 2b Fix: luu lai de control_layer co the doc
        self.requesting_car = requesting_car

        # Bug 2 Fix: random f_req theo Zipf
        self.f_req = int(np.random.choice(
            self.num_videos,
            p=self._zipf_probs
        ))

        offload_choice, z_req, cache_dec = self._decode_action(action_idx)

        # Xac dinh target node
        if offload_choice == self.OFFLOAD_LOCAL:
            target_node = requesting_car
        elif (self.OFFLOAD_UAV_START_IDX
              <= offload_choice < self.OFFLOAD_RSU_START_IDX):
            target_node = self.uavs[
                offload_choice - self.OFFLOAD_UAV_START_IDX
            ]
        else:
            rsu_idx = offload_choice - self.OFFLOAD_RSU_START_IDX
            target_node = self.rsus[rsu_idx]

        node_name = getattr(target_node, 'name', '')

        # --- Lay cache_mode TRUOC khi update (dung cho delay) ---
        cache_mode_before = int(self.cache_state.get(node_name, 0))
        z_cached_before   = int(self.cache_bitrate.get(node_name, 0))

        # --- Update cache state ---
        if target_node is not requesting_car and node_name in self.cache_state:
            if cache_dec == 1:
                self.cache_bitrate[node_name] = z_req
                if z_req == 0:
                    self.cache_state[node_name] = 1  # direct hit
                else:
                    self.cache_state[node_name] = 2  # transcoding
            else:
                self.cache_state[node_name]   = 0
                self.cache_bitrate[node_name] = 0

        # --- Update CPU load ---
        for n in self.uavs + self.rsus:
            self.cpu_load[n.name] = max(
                0.0, self.cpu_load[n.name] - self._cpu_decay
            )
        if target_node is not requesting_car and node_name in self.cpu_load:
            self.cpu_load[node_name] = min(
                1.0, self.cpu_load[node_name] + self._cpu_per_req
            )

        # --- Tinh delay ---
        cpu = self.cpu_load.get(node_name, 0.0)
        cost = calculate_total_cost(
            requesting_car,
            target_node,
            self.config,
            cache_mode  = cache_mode_before,
            all_uavs    = self.uavs,
            z_req       = z_req,
            z_cached    = z_cached_before if cache_mode_before == 2 else z_req,
            num_uavs    = len(self.uavs),
            cpu_load    = cpu,
        )

        reward = -cost

        return self.get_state(), reward, False, {}

    # ------------------------------------------------------------------
    def reset(self):
        """Reset toan bo trang thai moi truong moi episode."""
        self.cache_state   = {n.name: 0   for n in (self.uavs + self.rsus)}
        self.cache_bitrate = {n.name: 0   for n in (self.uavs + self.rsus)}
        self.cpu_load      = {n.name: 0.0 for n in (self.uavs + self.rsus)}
        self.f_req = 0
        # Bug 2b Fix: reset requesting_car ve mac dinh
        self.requesting_car = self.cars[0] if self.cars else None
        return self.get_state()

    # ------------------------------------------------------------------
    def get_action_components(self, action_idx: int):
        """
        Public helper: tra ve dict mo ta action (dung trong control_layer va ryu_app).

        BUG 5 FIX: them offload_name va bitrate_label vao dict.
        Truoc day dict thieu 2 key nay:
          - ryu_app.py log decision['offload_name'] -> KeyError
          - ryu_app.py log decision['bitrate_label'] -> KeyError
        """
        offload, z_req, cache = self._decode_action(action_idx)

        # offload_name
        if offload == self.OFFLOAD_LOCAL:
            offload_name = 'Local'
        elif self.OFFLOAD_UAV_START_IDX <= offload < self.OFFLOAD_RSU_START_IDX:
            uav_idx = offload - self.OFFLOAD_UAV_START_IDX
            if uav_idx < len(self.uavs):
                offload_name = getattr(self.uavs[uav_idx], 'name', f'uav{uav_idx+1}')
            else:
                offload_name = f'uav{uav_idx+1}'
        else:
            rsu_idx = offload - self.OFFLOAD_RSU_START_IDX
            if rsu_idx < len(self.rsus):
                offload_name = getattr(self.rsus[rsu_idx], 'name', f'rsu{rsu_idx+1}')
            else:
                offload_name = f'rsu{rsu_idx+1}'

        # bitrate_label
        _labels = ['low(480p)', 'high(1080p)']
        bitrate_label = _labels[z_req] if z_req < len(_labels) else f'z{z_req}'

        return {
            'offload_idx':   offload,
            'offload_name':  offload_name,      # Bug 5 Fix
            'bitrate':       z_req,
            'bitrate_label': bitrate_label,     # Bug 5 Fix
            'cache':         cache,
            'f_req':         self.f_req,
            'popularity':    float(self._zipf_probs[self.f_req]),
        }
