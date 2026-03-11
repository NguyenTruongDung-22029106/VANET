#!/usr/bin/env python3
"""
Delay model cho UAV-VANET ABR Video Streaming.

Cài đặt đúng theo công thức trong paper:
  Xie et al. "Joint Caching and User Association Optimization for
  Adaptive Bitrate Video Streaming in UAV-Assisted Cellular Networks"
  IEEE Access 2022.

Công thức tham chiếu:
  Eq(1-2) : Path loss LoS / NLoS
  Eq(3)   : Xác suất LoS  P^LoS_{l,k}
  Eq(4)   : SINR_{l,k}
  Eq(5)   : Downlink rate r_{l,k} = (B/M)·log2(1+SINR)
  Eq(6-8) : Backhaul path loss + SINR
  Eq(9)   : Backhaul rate r_{BS,l} = (B_h/L)·log2(1+SINR_{BS,l})
  Eq(10)  : D^1 — direct hit delay (cache đúng bitrate)
  Eq(11)  : D^2 — transcoding hit delay (cache bitrate cao hơn, cần transcode)
  Eq(12)  : D^3 — cache miss delay (kéo từ backhaul)
  Eq(13)  : D_{l,k} = D^1 + D^2 + D^3

Reward trong environment chỉ dùng delay (bỏ qua energy theo thiết kế dự án).

Cache mode (đồng bộ environment.py):
  0 = miss         → dùng D^3
  1 = direct hit   → dùng D^1
  2 = transcoding  → dùng D^2
"""

import math
from types import SimpleNamespace


# ============================================================
# Tham số mặc định (Table II Chen et al. + Table 2 Xie et al.)
# ============================================================
_DEFAULT = dict(
    # Radio access
    B            = 160e6,       # Bandwidth V2U (Hz) — 160 MHz (Table II Chen)
    Bh           = 60e6,        # Bandwidth V2B backhaul (Hz) — 60 MHz (Table II)
    M            = 30,          # Max users per UAV
    PUAV_dBm     = 30,          # UAV tx power (dBm) — Table II
    PBS_dBm      = 35,          # MBS tx power (dBm) — Table II
    sigma2_dBm   = -95,         # Noise power (dBm) — Table II

    # Path loss (Xie et al.)
    H            = 100.0,       # UAV altitude (m)
    fc           = 5e9,         # Carrier frequency V2U (Hz) — 5 GHz Table II
    d0           = 1.0,         # Reference distance (m)
    nLoS         = 2.0,         # Path loss exponent LoS
    nNLoS        = 2.4,         # Path loss exponent NLoS
    sLoS         = 5.3,         # Extra loss LoS (dB)
    sNLoS        = 5.27,        # Extra loss NLoS (dB)
    kappa        = 11.9,        # LoS probability param κ
    zeta         = 0.13,        # LoS probability param ζ (deg⁻¹)

    # Backhaul (Xie et al.)
    gamma_bs     = 3.5,         # Backhaul path loss exponent — khớp config.py
    eta_bs       = 100.0,       # Backhaul NLoS excess loss — khớp config.py

    # Computing (Xie et al.)
    w0           = 1.0,         # Cycles per bit for transcoding
    C_comp       = 3.4e9,       # UAV computing capacity (cycles/s)

    # Content
    chunk_size_MB = 8.0,        # s_{f,z=0}: khớp config.py content_size_MB=8MB
    # z=0: bitrate thấp (e.g. 480p), z=1: bitrate cao (e.g. 1080p)
    # s_{f,z} = chunk_size_MB * (z+1) — tỉ lệ tuyến tính với bitrate
)


def _cfg(config, key):
    """Lấy tham số từ config object hoặc fallback về default."""
    if config is None:
        return _DEFAULT[key]
    if isinstance(config, dict):
        return config.get(key, _DEFAULT[key])
    return getattr(config, key, _DEFAULT[key])


# ============================================================
# Node position helpers
# ============================================================

def _get_xy(node):
    """Lấy (x, y) từ Mininet node hoặc SimpleNamespace."""
    if hasattr(node, 'params') and 'position' in node.params:
        p = node.params['position']
        return float(p[0]), float(p[1])
    pos = getattr(node, 'position', None) or getattr(node, 'pos', None)
    if pos is not None:
        return float(pos[0]), float(pos[1])
    return 0.0, 0.0


def _dist_2d(n1, n2):
    x1, y1 = _get_xy(n1)
    x2, y2 = _get_xy(n2)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _dist_3d(uav_node, user_node, H):
    """d_{l,k} = sqrt(d_2d² + H²)  (UAV bay ở độ cao H)."""
    d2 = _dist_2d(uav_node, user_node)
    return math.sqrt(d2 ** 2 + H ** 2)


# ============================================================
# LoS probability  Eq(3)
# ============================================================

def _plos(uav_node, user_node, config):
    """
    P^LoS_{l,k} = 1 / (1 + κ·exp(-ζ·(θ_deg - κ)))  Eq(3)
    θ: elevation angle (degrees).
    """
    H     = _cfg(config, 'H')
    kappa = _cfg(config, 'kappa')
    zeta  = _cfg(config, 'zeta')

    d3 = max(_dist_3d(uav_node, user_node, H), 1e-3)
    theta_deg = math.degrees(math.asin(min(H / d3, 1.0)))
    return 1.0 / (1.0 + kappa * math.exp(-zeta * (theta_deg - kappa)))


# ============================================================
# Path loss  Eq(1-2)
# ============================================================

def _path_loss_dB(uav_node, user_node, config):
    """
    g_{l,k} = P^LoS·g^LoS + (1-P^LoS)·g^NLoS  (dB)
    g^LoS  = 20log(4πf_c·d_0/c) + 10·n_LoS·log(d) + s_LoS   Eq(1)
    g^NLoS = 20log(4πf_c·d_0/c) + 10·n_NLoS·log(d) + s_NLoS  Eq(2)
    """
    H     = _cfg(config, 'H')
    fc    = _cfg(config, 'fc')
    d0    = _cfg(config, 'd0')
    nL    = _cfg(config, 'nLoS');   nNL = _cfg(config, 'nNLoS')
    sL    = _cfg(config, 'sLoS');   sNL = _cfg(config, 'sNLoS')

    d   = max(_dist_3d(uav_node, user_node, H), 1e-3)
    c   = 3e8
    fsp = 20 * math.log10(4 * math.pi * fc * d0 / c)   # free-space reference
    gL  = fsp + 10 * nL  * math.log10(d) + sL
    gNL = fsp + 10 * nNL * math.log10(d) + sNL

    p = _plos(uav_node, user_node, config)
    return p * gL + (1 - p) * gNL


# ============================================================
# SINR và downlink rate  Eq(4-5)
# ============================================================

def _sinr_uav(uav_node, user_node, all_uavs, config):
    """
    SINR_{l,k} = P_UAV·|g_{l,k}|² / (Σ_{j≠l} P_UAV·|g_{j,k}|² + σ²)  Eq(4)
    Nhiễu từ các UAV khác trong cùng băng tần.
    """
    PUAV   = 10 ** ((_cfg(config, 'PUAV_dBm') - 30) / 10)   # W
    sigma2 = 10 ** ((_cfg(config, 'sigma2_dBm') - 30) / 10)

    g_signal = _path_loss_dB(uav_node, user_node, config)
    signal   = PUAV * 10 ** (-g_signal / 10)

    interf = 0.0
    for other in all_uavs:
        if getattr(other, 'name', None) != getattr(uav_node, 'name', None):
            g_other = _path_loss_dB(other, user_node, config)
            interf += PUAV * 10 ** (-g_other / 10)

    return signal / max(interf + sigma2, 1e-30)


def _rate_uav_user(uav_node, user_node, all_uavs, config):
    """
    r_{l,k} = (B/M)·log2(1 + SINR_{l,k})  Eq(5)  [bits/s]
    """
    B = _cfg(config, 'B')
    M = max(_cfg(config, 'M'), 1)
    sinr = _sinr_uav(uav_node, user_node, all_uavs, config)
    return (B / M) * math.log2(1 + max(sinr, 1e-10))


# ============================================================
# Backhaul rate  Eq(6-9)
# ============================================================

def _rate_backhaul(uav_node, config, num_uavs=1):
    """
    r_{BS,l} = (B_h/L)·log2(1 + SINR_{BS,l})  Eq(9)
    Path loss backhaul dùng free-space + LoS/NLoS  Eq(6-8).
    Vị trí MBS: (0, 0) nếu không truyền vào.
    """
    Bh     = _cfg(config, 'Bh')
    L      = max(num_uavs, 1)
    PBS    = 10 ** ((_cfg(config, 'PBS_dBm') - 30) / 10)
    sigma2 = 10 ** ((_cfg(config, 'sigma2_dBm') - 30) / 10)
    gamma  = _cfg(config, 'gamma_bs')
    eta    = _cfg(config, 'eta_bs')
    H      = _cfg(config, 'H')
    kappa  = _cfg(config, 'kappa')
    zeta   = _cfg(config, 'zeta')

    # MBS ở mặt đất (0,0)
    mbs = SimpleNamespace(params={'position': (0, 0)})
    d2  = max(_dist_2d(mbs, uav_node), 1e-3)

    # LoS prob backhaul (Eq(3) áp dụng cho link BS→UAV)
    d3    = math.sqrt(d2 ** 2 + H ** 2)
    theta = math.degrees(math.asin(min(H / d3, 1.0)))
    p     = 1.0 / (1.0 + kappa * math.exp(-zeta * (theta - kappa)))

    # Path loss backhaul Eq(6-8)
    g = p * (d2 ** (-gamma)) + (1 - p) * eta * (d2 ** (-gamma))
    sinr = PBS * g / max(sigma2, 1e-30)
    return (Bh / L) * math.log2(1 + max(sinr, 1e-10))


# ============================================================
# Chunk size helper
# ============================================================

def _chunk_size_bits(z_idx, config):
    """
    s_{f,z}: kích thước chunk ở bitrate z (bits).
    z=0 → bitrate thấp (s0), z=1 → bitrate cao (2·s0).
    Tỉ lệ tuyến tính với bitrate theo Xie et al.
    """
    s0 = _cfg(config, 'chunk_size_MB') * 8e6   # bits
    return s0 * (z_idx + 1)


# ============================================================
# 3 kịch bản delay  Eq(10-12)
# ============================================================

def _queuing_penalty(cpu_load: float, base_delay: float) -> float:
    """
    Mô hình queuing đơn giản: khi CPU load cao, delay tăng.
    Dùng M/M/1 approximation: penalty = base_delay * cpu_load / (1 - cpu_load)
    Clamp cpu_load vào [0, 0.95] để tránh singularity.

    Fix 3: đưa cpu_load vào cost thực sự để agent có lý do tránh UAV quá tải.
    """
    rho = min(max(float(cpu_load), 0.0), 0.95)
    if rho < 1e-6:
        return 0.0
    return base_delay * rho / (1.0 - rho)


def _delay_direct_hit(uav_node, user_node, all_uavs, config, z_req=0, cpu_load=0.0):
    """
    D^1_{l,k}: Cache có đúng bitrate yêu cầu → truyền thẳng.
    D^1 = s_{f,z} / r_{l,k}  +  queuing_penalty(cpu_load)   Eq(10) + Fix3
    """
    r_lk      = max(_rate_uav_user(uav_node, user_node, all_uavs, config), 1.0)
    s         = _chunk_size_bits(z_req, config)
    base      = s / r_lk
    return base + _queuing_penalty(cpu_load, base)


def _delay_transcoding(uav_node, user_node, all_uavs, config,
                       z_req=0, z_cached=1, cpu_load=0.0):
    """
    D^2_{l,k}: Cache có bitrate cao hơn (z_cached > z_req) → transcode rồi gửi.
    D^2 = s_{f,z}/r_{l,k} + w0·(s_{f,z+}-s_{f,z})/c_{l,k}  Eq(11)
    c_{l,k} giảm theo cpu_load (CPU đang bận): c_eff = c_lk * (1 - cpu_load)
    """
    r_lk   = max(_rate_uav_user(uav_node, user_node, all_uavs, config), 1.0)
    C_comp = max(_cfg(config, 'C_comp'), 1.0)
    M      = max(_cfg(config, 'M'), 1)
    w0     = _cfg(config, 'w0')
    # CPU còn trống = C_comp/M * (1 - cpu_load), min 1% để tránh /0
    c_lk_eff = max(C_comp / M * (1.0 - min(float(cpu_load), 0.99)), 1.0)

    s_req    = _chunk_size_bits(z_req,    config)
    s_cached = _chunk_size_bits(z_cached, config)

    tx_delay     = s_req / r_lk
    transc_delay = w0 * (s_cached - s_req) / c_lk_eff
    return tx_delay + transc_delay


def _delay_cache_miss(uav_node, user_node, all_uavs, config,
                      z_req=0, num_uavs=1, cpu_load=0.0):
    """
    D^3_{l,k}: Cache miss → kéo từ backhaul rồi gửi cho user.
    D^3 = s_{f,z}/r_{l,k} + s_{f,z}/r_{BS,l}  +  queuing_penalty   Eq(12) + Fix3
    """
    r_lk   = max(_rate_uav_user(uav_node, user_node, all_uavs, config), 1.0)
    r_bs_l = max(_rate_backhaul(uav_node, config, num_uavs), 1.0)
    s      = _chunk_size_bits(z_req, config)
    base   = s / r_lk + s / r_bs_l
    return base + _queuing_penalty(cpu_load, base)


# ============================================================
# Local processing delay (khi offload về xe chính)
# ============================================================

def _delay_local(config, z_req=0):
    """
    Xe tự xử lý: không có backhaul, không có UAV cache.
    Dùng CPU xe (giả định băng thông V2V rất giới hạn).
    Ước lượng: s_{f,z} / r_local với r_local = B_v2v / M_v2v.
    Theo Table II Chen et al.: V2V BW = 20 MHz, carrier = 5.9 GHz.
    """
    B_v2v  = 20e6    # Hz
    M_v2v  = 5       # max 5 xe lân cận
    SINR0  = 10.0    # SNR điển hình V2V (tuyến tính)
    r_local = (B_v2v / M_v2v) * math.log2(1 + SINR0)
    s       = _chunk_size_bits(z_req, config)
    return s / max(r_local, 1.0)


# ============================================================
# Public API — hàm dùng trong environment.py và qea_joint_ca_ua.py
# ============================================================

def calculate_total_cost(
    source_node,
    target_node,
    config,
    cache_mode: int = 0,
    all_uavs=None,
    z_req: int = 0,
    z_cached: int = 1,
    num_uavs: int = 1,
    cpu_load: float = 0.0,
) -> float:
    """
    Tính tổng delay phục vụ 1 request video.

    Tham số:
        source_node  : xe yêu cầu (Mininet station hoặc SimpleNamespace)
        target_node  : node phục vụ (UAV, RSU, hoặc chính source nếu Local)
        config       : config object (từ get_config())
        cache_mode   : 0=miss, 1=direct_hit, 2=transcoding
        all_uavs     : list tất cả UAV (để tính nhiễu SINR); None → [target_node]
        z_req        : bitrate yêu cầu (0=thấp, 1=cao, ...)
        z_cached     : bitrate đang cache (chỉ dùng khi cache_mode=2)
        num_uavs     : số UAV (để tính backhaul rate chia đều)
        cpu_load     : CPU utilization của target node [0,1] — Fix3: ảnh hưởng delay

    Trả về:
        delay (seconds) — reward = -delay trong environment.py
    """
    # Local offload: xe tự xử lý (cpu_load không áp dụng)
    if target_node is source_node or \
       getattr(target_node, 'name', '') == getattr(source_node, 'name', ''):
        return _delay_local(config, z_req)

    # UAV/RSU offload
    _all_uavs = all_uavs if all_uavs is not None else [target_node]

    if cache_mode == 1:
        # Eq(10) + queuing penalty
        return _delay_direct_hit(
            target_node, source_node, _all_uavs, config,
            z_req=z_req, cpu_load=cpu_load,
        )

    elif cache_mode == 2:
        # Eq(11) + CPU contention
        return _delay_transcoding(
            target_node, source_node, _all_uavs, config,
            z_req=z_req, z_cached=z_cached, cpu_load=cpu_load,
        )

    else:
        # Eq(12) + queuing penalty
        return _delay_cache_miss(
            target_node, source_node, _all_uavs, config,
            z_req=z_req, num_uavs=num_uavs, cpu_load=cpu_load,
        )