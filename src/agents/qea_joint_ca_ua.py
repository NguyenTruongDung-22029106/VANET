#!/usr/bin/env python3
"""
QEA-based Joint Caching and User Association (CA+UA) baseline.

Cài đặt đúng theo paper:
  "Joint Caching and User Association Optimization for Adaptive Bitrate
   Video Streaming in UAV-Assisted Cellular Networks"
  Xie et al., IEEE Access 2022. DOI: 10.1109/ACCESS.2022.3211940

Công thức tham chiếu:
  Eq(1-4)  : LoS/NLoS path loss + SINR (UAV → user)
  Eq(5)    : downlink rate r_{l,k}
  Eq(6-9)  : backhaul path loss + rate r_{BS,l}
  Eq(10)   : D^1 direct hit delay
  Eq(11)   : D^2 transcoding hit delay
  Eq(12)   : D^3 cache miss delay
  Eq(13)   : D_{l,k} = D^1 + D^2 + D^3
  Eq(14)   : D_tot = Σ_l Σ_k x_{l,k} * D_{l,k}
  Eq(21-22): Observation (Q-bit → binary)
  Eq(23-24): Rotation gate
  Table 1  : Lookup table Δθ
  Algo 1   : Main QEA loop (9 steps)
  Algo 2   : Binary solution repair (4 ràng buộc)
"""

import math
from typing import List, Tuple

import numpy as np

# Fix 4: dùng chung cost model + kích thước chunk với D3QN — so sánh công bằng
from models import calculate_total_cost as _models_cost, _chunk_size_bits


# ============================================================
# Tham số mặc định (Table 2 trong paper)
# ============================================================
_DEFAULT = dict(
    M            = 30,           # max users per UAV  (constraint 15c)
    cache_uav_MB = 300,          # UAV cache capacity (MB) → constraint 15a
    zipf_alpha   = 0.7,          # Zipf popularity skew
    # Tham số communication model đầy đủ nằm trong config.py / models.py
)


def _cfg(config, key):
    """Lấy giá trị từ config (SimpleNamespace/dict) hoặc fallback về default."""
    if key == 'zipf_alpha':
        aliases = ('zipf_alpha', 'zipf_exponent')
    else:
        aliases = (key,)

    if config is None:
        return _DEFAULT[key]
    if isinstance(config, dict):
        for k in aliases:
            if k in config:
                return config[k]
        return _DEFAULT[key]

    for k in aliases:
        if hasattr(config, k):
            return getattr(config, k)
    return _DEFAULT[key]


# ============================================================
# Zipf popularity  p_{k,f,z}
# ============================================================

def _zipf_popularity(F: int, Z: int, alpha: float) -> np.ndarray:
    """p_{f,z} theo Zipf-like trên F×Z pairs [paper Section III-B]. Shape (F,Z)."""
    ranks = np.arange(1, F * Z + 1, dtype=np.float64)
    raw   = 1.0 / ranks ** alpha
    return (raw / raw.sum()).reshape(F, Z)


# ============================================================
# QEA Optimizer
# ============================================================

class QEAJointCAUA:
    """
    QEA-based Joint Caching and User Association Optimizer.
    Cài đặt đầy đủ theo Algorithm 1 + Algorithm 2 + Table 1 của paper.

    Biến tối ưu:
      X ∈ {0,1}^{L×K}   — x_{l,k}: user k gán cho UAV l
      Y ∈ {0,1}^{L×F×Z} — y_{l,f,z}: UAV l cache chunk f bitrate z

    Objective: D_tot = Σ_l Σ_k x_{l,k}·D_{l,k}  Eq(14)
    """

    # Table 1: Δθ theo (x_cur, b_best, f(Xi) >= f(B))
    _LOOKUP = {
        (0, 0, False): 0.0,
        (0, 0, True):  0.0,
        (0, 1, False): +0.01 * math.pi,
        (0, 1, True):  0.0,
        (1, 0, False): -0.01 * math.pi,
        (1, 0, True):  0.0,
        (1, 1, False): 0.0,
        (1, 1, True):  0.0,
    }

    def __init__(
        self,
        cars:     List,
        uavs:     List,
        rsus:     List,          # dùng cho backhaul delay khi cache miss
        config         = None,
        F:        int  = 10,
        Z:        int  = 4,
        pop_size: int  = 20,
        t_max:    int  = 100,
        seed:     int  = 0,
    ):
        self.cars   = list(cars)
        self.uavs   = list(uavs)
        self.rsus   = list(rsus)
        self.config = config

        self.K = len(self.cars)
        self.uav_count = len(self.uavs)
        self.L = self.uav_count
        self.F = int(F)
        self.Z = int(Z)

        self.pop_size = int(pop_size)
        self.t_max    = int(t_max)
        self.rng      = np.random.RandomState(seed)

        # Cache capacity (bits) per UAV — constraint 15a
        cache_MB     = _cfg(config, 'cache_uav_MB')
        self.C_cache = float(cache_MB) * 8e6

        # Kích thước chunk theo bitrate z (bits)
        self.chunk_sizes = np.array(
            [_chunk_size_bits(z, config) for z in range(self.Z)],
            dtype=np.float64,
        )

        # Popularity p_{f,z}  (F × Z)
        zipf_a    = _cfg(config, 'zipf_alpha')
        self.p_fz = _zipf_popularity(self.F, self.Z, zipf_a)

        # Q-populations
        self.QX_alpha = None;  self.QX_beta = None
        self.QY_alpha = None;  self.QY_beta = None

        self.X_best = None   # (L, K)
        self.Y_best = None   # (L, F, Z)
        self.f_best = float("inf")
        self.convergence: List[float] = []   # f_best mỗi thế hệ

        self._init_q_population()

    # ------------------------------------------------------------------
    # Step 2: Init Q-population
    # ------------------------------------------------------------------
    def _init_q_population(self):
        v = 1.0 / math.sqrt(2.0)
        self.QX_alpha = np.full((self.pop_size, self.L, self.K),          v, dtype=np.float64)
        self.QX_beta  = np.full((self.pop_size, self.L, self.K),          v, dtype=np.float64)
        self.QY_alpha = np.full((self.pop_size, self.L, self.F, self.Z),  v, dtype=np.float64)
        self.QY_beta  = np.full((self.pop_size, self.L, self.F, self.Z),  v, dtype=np.float64)

    # ------------------------------------------------------------------
    # Step 3: Observation  Eq(21-22)
    # ------------------------------------------------------------------
    def _observe_population(self) -> Tuple[np.ndarray, np.ndarray]:
        X = (self.rng.rand(self.pop_size, self.L, self.K)
             < self.QX_beta ** 2).astype(np.int8)
        Y = (self.rng.rand(self.pop_size, self.L, self.F, self.Z)
             < self.QY_beta ** 2).astype(np.int8)
        return X, Y

    # ------------------------------------------------------------------
    # Step 4: Repair  Algorithm 2
    # ------------------------------------------------------------------
    def _repair_X(self, X: np.ndarray) -> np.ndarray:
        """
        Algorithm 2 lines 11-28:
          15b: Σ_l x_{l,k} = 1  (mỗi user chọn đúng 1 UAV)
          15c: Σ_k x_{l,k} ≤ M  (mỗi UAV không quá M users)
        """
        M = int(_cfg(self.config, 'M'))

        # Lines 11-16: đảm bảo mỗi user k có đúng 1 UAV
        for k in range(self.K):
            col = X[:, k]
            s   = int(col.sum())
            if s == 0:
                l = self.rng.randint(0, self.L)
                col[:] = 0;  col[l] = 1
            elif s > 1:
                keep = int(self.rng.choice(np.where(col == 1)[0]))
                col[:] = 0;  col[keep] = 1
            X[:, k] = col

        # Lines 17-22: đảm bảo mỗi UAV l không quá M users
        for l in range(self.L):
            while int(X[l, :].sum()) > M:
                ones = np.where(X[l, :] == 1)[0]
                drop = int(self.rng.choice(ones))
                X[l, drop] = 0

        # Lines 23-28: gán lại user bị bỏ rơi sau bước trên
        for k in range(self.K):
            if int(X[:, k].sum()) == 0:
                candidates = [l for l in range(self.L) if int(X[l, :].sum()) < M]
                if not candidates:
                    candidates = list(range(self.L))
                l = int(self.rng.choice(candidates))
                X[l, k] = 1

        return X

    def _repair_Y(self, Y: np.ndarray) -> np.ndarray:
        """
        Algorithm 2 lines 1-10:
          15a: Σ_{f,z} y_{l,f,z}·s_{f,z} ≤ C^cache_l
          - Xóa ngẫu nhiên nếu vượt capacity.
          - Thêm ngẫu nhiên nếu còn chỗ (giữ đúng spirit của paper).
        """
        for l in range(self.L):
            usage = float(np.sum(Y[l] * self.chunk_sizes[np.newaxis, :]))

            # Lines 2-5: xóa bớt khi vượt capacity
            while usage > self.C_cache:
                ones = list(zip(*np.where(Y[l] == 1)))
                if not ones:
                    break
                idx  = self.rng.randint(0, len(ones))
                f, z = ones[idx]
                Y[l, f, z]  = 0
                usage -= self.chunk_sizes[z]

            # Lines 6-9: lấp đầy khi còn chỗ
            zeros = list(zip(*np.where(Y[l] == 0)))
            self.rng.shuffle(zeros) if zeros else None
            for f, z in zeros:
                if usage + self.chunk_sizes[z] <= self.C_cache:
                    Y[l, f, z]  = 1
                    usage += self.chunk_sizes[z]

        return Y

    # ------------------------------------------------------------------
    # Step 5: Evaluate  Eq(14)
    # ------------------------------------------------------------------
    def _evaluate_individual(self, X_i: np.ndarray, Y_i: np.ndarray) -> float:
        """
        D_tot = Σ_l Σ_k x_{l,k}·D_{l,k}  Eq(14)

        Fix 4: dùng calculate_total_cost() từ models.py — cùng hàm mục tiêu
        với D3QN, đảm bảo so sánh công bằng.

        Mapping Y_i → cache_mode + z_req:
          y_{l,f,z} = 1 tại z_req=0 → mode=1 (direct hit)
          y_{l,f,z} = 1 tại z_req=1 → mode=2 (transcoding, có bản cao hơn)
          không có y nào = 1         → mode=0 (cache miss)
        """
        # Eq(14) + Eq(10-13) trong paper:
        #   D_tot = Σ_l Σ_k x_{l,k} · D_{l,k}
        #   D_{l,k} = D^1_{l,k} + D^2_{l,k} + D^3_{l,k}
        # với D^1, D^2, D^3 tính theo caching state Y_i[l,f,z].
        total = 0.0
        if self.F <= 0 or self.Z <= 0:
            return total

        p_fz = self.p_fz  # shape (F, Z)
        for l in range(self.L):
            uav = self.uavs[l]
            Y_l = Y_i[l]  # (F, Z)
            users_on_l = max(int(np.sum(X_i[l, :])), 1)

            for k in range(self.K):
                if X_i[l, k] == 0:
                    continue
                car = self.cars[k]

                # Precompute miss + direct delays theo z (không phụ thuộc f)
                d_direct_by_z = {}
                d_miss_by_z   = {}
                for z_req in range(self.Z):
                    d_direct_by_z[z_req] = _models_cost(
                        source_node=car, target_node=uav, config=self.config,
                        cache_mode=1, all_uavs=self.uavs,
                        z_req=z_req, z_cached=z_req,
                        num_uavs=self.uav_count, rsus=self.rsus,
                        num_users_per_uav=users_on_l,
                    )
                    d_miss_by_z[z_req] = _models_cost(
                        source_node=car, target_node=uav, config=self.config,
                        cache_mode=0, all_uavs=self.uavs,
                        z_req=z_req, z_cached=z_req,
                        num_uavs=self.uav_count, rsus=self.rsus,
                        num_users_per_uav=users_on_l,
                    )

                # Loop đúng theo Eq(10-12): từng (f,z)
                for f in range(self.F):
                    for z_req in range(self.Z):
                        p = float(p_fz[f, z_req])
                        if p <= 0.0:
                            continue

                        if int(Y_l[f, z_req]) == 1:
                            # Direct hit: y_{l,f,z}=1
                            total += p * d_direct_by_z[z_req]
                            continue

                        # hl,f,z = min( Σ_{z'>z} y_{l,f,z'}, 1 )
                        z_plus = None
                        for z2 in range(z_req + 1, self.Z):
                            if int(Y_l[f, z2]) == 1:
                                z_plus = z2
                                break

                        if z_plus is not None:
                            # Transcoding hit: y_{l,f,z}=0 và h_{l,f,z}=1 với z^+ = min cached > z
                            d_trans = _models_cost(
                                source_node=car, target_node=uav, config=self.config,
                                cache_mode=2, all_uavs=self.uavs,
                                z_req=z_req, z_cached=z_plus,
                                num_uavs=self.uav_count, rsus=self.rsus,
                                num_users_per_uav=users_on_l,
                            )
                            total += p * d_trans
                        else:
                            # Cache miss: không cache requested và không có higher bitrate
                            total += p * d_miss_by_z[z_req]

        return total

    def _evaluate_population(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        fitness = np.zeros(self.pop_size, dtype=np.float64)
        for i in range(self.pop_size):
            fitness[i] = self._evaluate_individual(X[i], Y[i])
        return fitness

    # ------------------------------------------------------------------
    # Step 7: Rotation gate  Eq(23-24) + Table 1
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate(alpha, beta, dtheta):
        ct, st = math.cos(dtheta), math.sin(dtheta)
        a = alpha * ct - beta * st
        b = alpha * st + beta * ct
        n = math.sqrt(a * a + b * b)
        return (a / n, b / n) if n > 0 else (a, b)

    def _rotation_update(self, X, Y, fitness, X_ref, Y_ref, f_ref):
        """Eq(23-24) với Δθ tra Table 1."""
        for i in range(self.pop_size):
            worse = float(fitness[i]) >= f_ref

            # ---- Update QX  Eq(23) ----
            for l in range(self.L):
                for k in range(self.K):
                    dtheta = self._LOOKUP[(int(X[i, l, k]), int(X_ref[l, k]), worse)]
                    if dtheta == 0.0:
                        continue
                    a, b = self._rotate(
                        self.QX_alpha[i, l, k],
                        self.QX_beta[i, l, k],
                        dtheta,
                    )
                    self.QX_alpha[i, l, k] = a
                    self.QX_beta[i, l, k]  = b

            # ---- Update QY  Eq(24) — vectorized ----
            # Theo Table 1, khi worse=True tất cả Δθ=0 → skip hoàn toàn
            if worse:
                continue

            cur  = Y[i]      # (L, F, Z)
            best = Y_ref

            for x_c, b_b, dtheta in [(0, 1, +0.01 * math.pi),
                                      (1, 0, -0.01 * math.pi)]:
                mask = (cur == x_c) & (best == b_b)
                if not mask.any():
                    continue
                ct, st = math.cos(dtheta), math.sin(dtheta)
                a = self.QY_alpha[i];  b = self.QY_beta[i]
                an = np.where(mask, a * ct - b * st, a)
                bn = np.where(mask, a * st + b * ct, b)
                nm = np.sqrt(an ** 2 + bn ** 2)
                nm = np.where(nm > 0, nm, 1.0)
                self.QY_alpha[i] = an / nm
                self.QY_beta[i]  = bn / nm

    # ------------------------------------------------------------------
    # Algorithm 1: Main loop
    # ------------------------------------------------------------------
    def optimize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Chạy QEA t_max thế hệ. Trả về (X_best, Y_best)."""
        BX = None;  BY = None

        for t in range(self.t_max):
            # Step 3
            X_pop, Y_pop = self._observe_population()

            # Step 4
            for i in range(self.pop_size):
                X_pop[i] = self._repair_X(X_pop[i])
                Y_pop[i] = self._repair_Y(Y_pop[i])

            # Step 5
            fitness  = self._evaluate_population(X_pop, Y_pop)

            # Step 6: store best
            idx = int(np.argmin(fitness))
            if t == 0 or fitness[idx] < self.f_best:
                self.f_best = float(fitness[idx])
                BX = X_pop[idx].copy()
                BY = Y_pop[idx].copy()

            self.convergence.append(self.f_best)

            # Step 7
            self._rotation_update(X_pop, Y_pop, fitness, BX, BY, self.f_best)

        self.X_best = BX
        self.Y_best = BY
        return self.X_best, self.Y_best

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def build_offload_policy(self) -> List[int]:
        """
        List uav_idx cho mọi user k (0..L-1), khớp convention hiện tại.
        """
        if self.X_best is None:
            raise RuntimeError("Chưa gọi optimize().")
        policy: List[int] = []
        for k in range(self.K):
            row = int(np.argmax(self.X_best[:, k]))
            policy.append(row)
        return policy

    def get_offload_for_car(self, car_idx: int) -> int:
        if self.X_best is None:
            raise RuntimeError("Chưa gọi optimize().")
        k = int(car_idx)
        if not (0 <= k < self.K):
            raise IndexError("car_idx out of range")
        return int(np.argmax(self.X_best[:, k]))
