#!/usr/bin/env python3
"""
VanetEnvironment — đã sửa:
  Bug 1: requesting_car random mỗi step (không còn cố định cars[0])
  Bug 2: popularity Zipf p_f tính động theo f_req thực sự (không còn 0.7)

Công thức Zipf (Chen et al. Eq.1 / Xie et al. Section III-B):
  p_f = f^{-γ} / Σ_{j=1}^{F} j^{-γ}

Cache 3-mode (đồng bộ models.py):
  0 = không cache                     → D^3 cache miss
  1 = cache đúng bitrate yêu cầu      → D^1 direct hit
  2 = cache bản bitrate cao hơn        → D^2 transcoding

CPU load (Fix 3):
  Được truyền vào calculate_total_cost() → ảnh hưởng delay thực sự.
  Agent "nhìn" cpu_load trong state VÀ môi trường phạt khi CPU cao.
"""
import random
import numpy as np
from models import calculate_total_cost

# Số mức bitrate (z): 0=low, 1=high
NUM_BITRATES = 2

# ============================================================
# Zipf helper  (Chen et al. Eq.1)
# ============================================================

def _compute_zipf_probs(num_videos: int, gamma: float) -> np.ndarray:
    """
    p_f = f^{-γ} / Σ_{j=1}^{F} j^{-γ}   (f = 1..F, 1-indexed rank)
    Trả về array shape (num_videos,), index 0 = video phổ biến nhất.
    """
    ranks = np.arange(1, num_videos + 1, dtype=np.float64)
    raw   = ranks ** (-gamma)
    return raw / raw.sum()


class VanetEnvironment:
    def __init__(self, config, stations, aps=None, uavs_list=None):
        self.config = config
        self.cars = [n for n in stations if 'car' in n.name]

        if uavs_list is not None and len(uavs_list) > 0:
            self.uavs = list(uavs_list)
        else:
            self.uavs = [n for n in stations if 'uav' in getattr(n, 'name', '')]
        self.rsus = list(aps) if aps is not None else [
            n for n in stations if 'rsu' in getattr(n, 'name', '')
        ]

        # --- Action space (Fix 2) ---
        self.OFFLOAD_LOCAL         = 0
        self.OFFLOAD_UAV_START_IDX = 1
        self.OFFLOAD_RSU_START_IDX = self.OFFLOAD_UAV_START_IDX + len(self.uavs)
        self.num_offload_targets   = 1 + len(self.uavs) + len(self.rsus)
        self.num_bitrates          = NUM_BITRATES   # z ∈ {0, 1}
        self.num_cache_actions     = 2              # 0=no_cache, 1=cache
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

        # Cache state: mode per node (0=miss,1=hit,2=transcoding)
        self.cache_state     = {n.name: 0 for n in (self.uavs + self.rsus)}
        # Bitrate cached per node (z đang cache)
        self.cache_bitrate   = {n.name: 0 for n in (self.uavs + self.rsus)}

        # CPU load động (Fix 3)
        self.cpu_load        = {n.name: 0.0 for n in (self.uavs + self.rsus)}
        self._cpu_decay      = 0.05   # giảm mỗi step
        self._cpu_per_req    = 0.1    # tăng khi có request

        # -------------------------------------------------------
        # BUG FIX 2: Zipf popularity động
        # -------------------------------------------------------
        # Lấy tham số từ config (có fallback về giá trị mặc định)
        self.num_videos    = int(getattr(config, 'num_videos',    100))
        self.zipf_exponent = float(getattr(config, 'zipf_exponent', 0.7))
        # Precompute xác suất Zipf cho tất cả F video — shape (num_videos,)
        self._zipf_probs   = _compute_zipf_probs(self.num_videos, self.zipf_exponent)
        # Video đang được request trong step hiện tại (0-indexed)
        self.f_req         = 0

    # ------------------------------------------------------------------
    @staticmethod
    def get_pos_from_node(node):
        if hasattr(node, 'params') and 'position' in node.params:
            p = node.params['position']
            return float(p[0]), float(p[1])
        return 0.0, 0.0

    def get_state(self):
        """
        State vector đầy đủ: pos + cpu_load + cache_mode + popularity.

        BUG FIX 2: popularity = Zipf p_{f_req} tính động,
                   KHÔNG còn là giá trị cứng 0.7.
        """
        sv = []
        for n in self.cars + self.uavs + self.rsus:
            sv.extend(self.get_pos_from_node(n))
        for n in self.uavs + self.rsus:
            sv.append(float(self.cpu_load.get(n.name, 0.0)))
        for n in self.uavs + self.rsus:
            sv.append(float(self.cache_state.get(n.name, 0)))

        # --- BUG FIX 2: Zipf p_f động thay vì 0.7 cứng ---
        p_f = float(self._zipf_probs[self.f_req])
        sv.append(p_f)

        return np.array(sv, dtype=np.float32)

    # ------------------------------------------------------------------
    def _decode_action(self, action_idx: int):
        """
        Giải mã action_idx thành (offload_target, z_req, cache_decision).

        Encoding:
          action_idx = offload + num_offload * (z + num_bitrates * cache)
        """
        a            = int(action_idx)
        offload      = a % self.num_offload_targets
        remainder    = a // self.num_offload_targets
        z_req        = remainder % self.num_bitrates
        cache_dec    = remainder // self.num_bitrates
        return offload, z_req, int(cache_dec)

    def encode_action(self, offload: int, z_req: int, cache: int) -> int:
        """Encode ngược lại (dùng cho test/debug)."""
        return offload + self.num_offload_targets * (z_req + self.num_bitrates * cache)

    # ------------------------------------------------------------------
    def step(self, action_idx: int):
        """
        One step:
          1. BUG FIX 1: chọn requesting_car ngẫu nhiên (không cố định cars[0])
          2. BUG FIX 2: chọn f_req ngẫu nhiên theo phân phối Zipf
          3. Decode action → (offload, z_req, cache)
          4. Update cache state theo bitrate agent chọn
          5. Update CPU load
          6. Tính delay (Eq 10-12 + queuing penalty)
          7. reward = -delay  (Fix 1)
        """
        # -------------------------------------------------------
        # BUG FIX 1: random requesting_car — không còn cứng cars[0]
        # -------------------------------------------------------
        requesting_car = random.choice(self.cars)

        # -------------------------------------------------------
        # BUG FIX 2: random f_req theo Zipf (video phổ biến hơn
        #            được chọn nhiều hơn, đúng với mô hình paper)
        # -------------------------------------------------------
        self.f_req = int(np.random.choice(
            self.num_videos,
            p=self._zipf_probs
        ))

        offload_choice, z_req, cache_dec = self._decode_action(action_idx)

        # Xác định target node
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

        # --- Lấy cache_mode TRƯỚC khi update (dùng cho delay) ---
        cache_mode_before = int(self.cache_state.get(node_name, 0))
        z_cached_before   = int(self.cache_bitrate.get(node_name, 0))

        # --- Update cache state (Fix 2: agent chọn bitrate z) ---
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

        # --- Update CPU load (Fix 3) ---
        for n in self.uavs + self.rsus:
            self.cpu_load[n.name] = max(
                0.0, self.cpu_load[n.name] - self._cpu_decay
            )
        if target_node is not requesting_car and node_name in self.cpu_load:
            self.cpu_load[node_name] = min(
                1.0, self.cpu_load[node_name] + self._cpu_per_req
            )

        # --- Tính delay (Fix 1+3) ---
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

        # Fix 1: reward = -cost (delay, seconds)
        reward = -cost

        return self.get_state(), reward, False, {}

    # ------------------------------------------------------------------
    def reset(self):
        """Reset toàn bộ trạng thái môi trường mỗi episode."""
        self.cache_state   = {n.name: 0   for n in (self.uavs + self.rsus)}
        self.cache_bitrate = {n.name: 0   for n in (self.uavs + self.rsus)}
        self.cpu_load      = {n.name: 0.0 for n in (self.uavs + self.rsus)}
        # Reset f_req về video phổ biến nhất
        self.f_req = 0
        return self.get_state()

    # ------------------------------------------------------------------
    def get_action_components(self, action_idx: int):
        """Public helper: trả về dict mô tả action (dùng trong control_layer)."""
        offload, z_req, cache = self._decode_action(action_idx)
        return {
            'offload_idx':  offload,
            'bitrate':      z_req,
            'cache':        cache,
            'f_req':        self.f_req,
            'popularity':   float(self._zipf_probs[self.f_req]),
        }