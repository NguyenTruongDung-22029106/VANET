#!/usr/bin/env python3
"""
Delay model cho UAV-VANET ABR Video Streaming.

Mô hình được điều chỉnh theo thực nghiệm Mininet-WiFi, dựa trên:
  Xie et al. "Joint Caching and User Association Optimization for
  Adaptive Bitrate Video Streaming in UAV-Assisted Cellular Networks"
  IEEE Access 2022.

Công thức tham chiếu:
  Eq(1-2) : Path loss LoS / NLoS
  Eq(3)   : Xác suất LoS  P^LoS_{l,k}
  Eq(4)   : SINR_{l,k}
  Eq(5)   : Downlink rate (runtime-adjusted) r_{l,k} = (B/M_eff)·log2(1+SINR)
  Eq(6-8) : Backhaul path loss + SINR
  Eq(9)   : Backhaul rate r_{BS,l} = (B_h/L)·log2(1+SINR_{BS,l})
  Eq(10)  : D^1 — direct hit delay (cache đúng bitrate)
  Eq(11)  : D^2 — transcoding hit delay (cache bitrate cao hơn, cần transcode)
  Eq(12)  : D^3 — cache miss delay (kéo từ backhaul)
  Eq(13)  : D_{l,k} = D^1 + D^2 + D^3

Reward trong environment chỉ dùng delay (bỏ qua energy).

Cache mode (đồng bộ environment.py):
  0 = miss         → dùng D^3
  1 = direct hit   → dùng D^1
  2 = transcoding  → dùng D^2
"""

import math
import warnings
from types import SimpleNamespace

from helpers import get_node_xy, dist_2d, dist_3d


# ============================================================
# Tham số mặc định (Table II Chen et al. + Mininet runtime adjustments)
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


# Internal aliases used throughout this module
_get_xy = get_node_xy
_dist_2d = dist_2d
_dist_3d = dist_3d

_WARNED_NO_MBS_NODE = False

def _plos_alt(altitude: float, tx_node, user_node, config):
    """
    LoS probability computed with a provided altitude (không dùng config.H cố định).
    Used for BS->user baseline where BS altitude differs from UAV altitude.
    """
    H = float(altitude)
    kappa = _cfg(config, 'kappa')
    zeta  = _cfg(config, 'zeta')

    # d3d uses the provided altitude
    d3 = max(dist_3d(tx_node, user_node, H), 1e-3)
    theta_deg = math.degrees(math.asin(min(H / d3, 1.0)))
    return 1.0 / (1.0 + kappa * math.exp(-zeta * (theta_deg - kappa)))


def _path_loss_dB_alt(tx_node, user_node, config, altitude: float):
    """
    Path loss (dB) similar to _path_loss_dB but parameterized by altitude.
    """
    fc    = _cfg(config, 'fc')
    d0    = _cfg(config, 'd0')
    nL    = _cfg(config, 'nLoS');   nNL = _cfg(config, 'nNLoS')
    sL    = _cfg(config, 'sLoS');   sNL = _cfg(config, 'sNLoS')

    d   = max(dist_3d(tx_node, user_node, float(altitude)), 1e-3)
    c   = 3e8
    fsp = 20 * math.log10(4 * math.pi * fc * d0 / c)   # free-space reference

    gL  = fsp + 10 * nL  * math.log10(d) + sL
    gNL = fsp + 10 * nNL * math.log10(d) + sNL

    p = _plos_alt(altitude, tx_node, user_node, config)
    return p * gL + (1 - p) * gNL

def _effective_users_per_uav(config, num_users_per_uav=None):
    m_cfg = max(int(_cfg(config, 'M')), 1)
    if num_users_per_uav is None:
        return m_cfg
    return max(int(num_users_per_uav), 1)


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
    SINR_{l,k} = P_UAV·|g_{l,k}|² / (Σ_{j≠l} P_UAV·|g_{j,k}|² + σ²)  Eq(4) Xie et al.
    Tất cả UAV dùng chung băng tần B → có inter-UAV co-channel interference.
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


def _rate_uav_user(uav_node, user_node, all_uavs, config, num_users_per_uav=None):
    """
    r_{l,k} = (B/M_eff)·log2(1 + SINR_{l,k})  [bits/s]
    M_eff ưu tiên theo runtime load, fallback về config.M.
    """
    B = _cfg(config, 'B')
    M = _effective_users_per_uav(config, num_users_per_uav)
    sinr = _sinr_uav(uav_node, user_node, all_uavs, config)
    return (B / M) * math.log2(1 + max(sinr, 1e-10))


def _rate_bs_user(bs_node, user_node, config, num_users_bs=None):
    """
    Macro BS (MBS) -> user downlink rate for out-of-coverage UAV fallback.

    Assumptions (to align with your requested modeling):
      - LoS/NLoS probabilistic pathloss (using same parameters as air-to-ground model)
      - Downlink bandwidth is shared equally among the users served by BS.
        Therefore B_{b,v} = Bh / M_eff
      - If `sigma2_dBm` is interpreted as noise power over full `Bh`, then when the
        bandwidth is reduced to Bh/M_eff, noise power scales proportionally
        (sigma2 -> sigma2 / M_eff), which yields SINR scaling by M_eff.
    """
    Bh = _cfg(config, 'Bh')
    M_eff = _effective_users_per_uav(config, num_users_bs)
    M_eff = max(int(M_eff), 1)

    # Equal bandwidth share per user
    Bv = Bh / float(M_eff)

    # BS tx power
    PBS = 10 ** ((_cfg(config, 'PBS_dBm') - 30) / 10)
    sigma2 = 10 ** ((_cfg(config, 'sigma2_dBm') - 30) / 10)

    H_bs = float(getattr(config, 'H_bs', 1.0))
    ploss_dB = _path_loss_dB_alt(bs_node, user_node, config, altitude=H_bs)
    signal = PBS * 10 ** (-ploss_dB / 10)
    # Noise scaling under bandwidth sharing:
    # if sigma2 is noise power over Bh, then noise over Bv is sigma2 / M_eff.
    sinr = (signal / max(sigma2, 1e-30)) * float(M_eff)
    return Bv * math.log2(1 + max(sinr, 1e-10))


def calculate_mbs_only_delay(user_node, bs_node, config, z_req: int = 0, num_users_bs=None) -> float:
    """
    Delay when the user is served directly by MBS/RSU (no UAV in coverage).
    """
    s = _chunk_size_bits(z_req, config)
    r = max(_rate_bs_user(bs_node, user_node, config, num_users_bs=num_users_bs), 1.0)
    return s / r


# ============================================================
# Backhaul rate  Eq(6-9)
# ============================================================

def _rate_backhaul(uav_node, config, num_uavs=1, mbs_node=None):
    """
    r_{BS,l} = (B_h/L)·log2(1 + SINR_{BS,l})  Eq(9)
    Path loss backhaul dùng free-space + LoS/NLoS  Eq(6-8).
    mbs_node: node RSU/MBS thật từ topology. Nếu None → fallback (0,0).
    """
    Bh     = _cfg(config, 'Bh')
    L      = max(num_uavs, 1)
    PBS    = 10 ** ((_cfg(config, 'PBS_dBm') - 30) / 10)
    sigma2 = 10 ** ((_cfg(config, 'sigma2_dBm') - 30) / 10)
    gamma  = _cfg(config, 'gamma_bs')
    eta    = max(float(_cfg(config, 'eta_bs')), 1.0)
    H      = _cfg(config, 'H')
    kappa  = _cfg(config, 'kappa')
    zeta   = _cfg(config, 'zeta')

    global _WARNED_NO_MBS_NODE
    if mbs_node is None:
        # Backward-safe fallback, but explicit warning to avoid silent model drift.
        if not _WARNED_NO_MBS_NODE:
            warnings.warn(
                "No RSU/MBS node provided to backhaul model; falling back to origin (0,0). "
                "This can bias D^3 delay. Pass rsus=[mbs_node] for accurate results.",
                RuntimeWarning,
            )
            _WARNED_NO_MBS_NODE = True
        mbs_node = SimpleNamespace(params={'position': (0, 0)})
    d2  = max(_dist_2d(mbs_node, uav_node), 1e-3)

    d3    = math.sqrt(d2 ** 2 + H ** 2)
    theta = math.degrees(math.asin(min(H / d3, 1.0)))
    p     = 1.0 / (1.0 + kappa * math.exp(-zeta * (theta - kappa)))

    # eta_bs là NLoS excess loss (>1), nên phải làm giảm gain ở nhánh NLoS.
    g = p * (d2 ** (-gamma)) + (1 - p) * (d2 ** (-gamma)) / max(eta, 1e-9)
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

def _delay_direct_hit(uav_node, user_node, all_uavs, config, z_req=0,
                      num_users_per_uav=None):
    """
    D^1_{l,k}: Cache có đúng bitrate yêu cầu → truyền thẳng.
    D^1 = s_{f,z} / r_{l,k}  Eq(10)
    """
    r_lk      = max(_rate_uav_user(uav_node, user_node, all_uavs, config, num_users_per_uav), 1.0)
    s         = _chunk_size_bits(z_req, config)
    base      = s / r_lk
    return base


def _delay_transcoding(uav_node, user_node, all_uavs, config,
                       z_req=0, z_cached=1, num_users_per_uav=None):
    """
    D^2_{l,k}: Cache có bitrate cao hơn (z_cached > z_req) → transcode rồi gửi.
    D^2 = s_{f,z}/r_{l,k} + w0·(s_{f,z+}-s_{f,z})/c_{l,k}  Eq(11)
    """
    r_lk   = max(_rate_uav_user(uav_node, user_node, all_uavs, config, num_users_per_uav), 1.0)
    C_comp = max(_cfg(config, 'C_comp'), 1.0)
    M_eff  = _effective_users_per_uav(config, num_users_per_uav)
    w0     = _cfg(config, 'w0')
    c_lk_eff = max(C_comp / M_eff, 1.0)

    s_req    = _chunk_size_bits(z_req,    config)
    s_cached = _chunk_size_bits(z_cached, config)

    tx_delay     = s_req / r_lk
    transc_delay = w0 * (s_cached - s_req) / c_lk_eff
    return tx_delay + transc_delay


def _delay_cache_miss(uav_node, user_node, all_uavs, config,
                      z_req=0, num_uavs=1, mbs_node=None,
                      num_users_per_uav=None):
    """
    D^3_{l,k}: Cache miss → kéo từ backhaul rồi gửi cho user.
    D^3 = s_{f,z}/r_{l,k} + s_{f,z}/r_{BS,l}  Eq(12)
    """
    r_lk   = max(_rate_uav_user(uav_node, user_node, all_uavs, config, num_users_per_uav), 1.0)
    r_bs_l = max(_rate_backhaul(uav_node, config, num_uavs, mbs_node=mbs_node), 1.0)
    s      = _chunk_size_bits(z_req, config)
    base   = s / r_lk + s / r_bs_l
    return base


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
    rsus=None,
    num_users_per_uav=None,
) -> float:
    """
    Tính tổng delay phục vụ 1 request video.

    Tham số:
        source_node       : xe yêu cầu (Mininet station hoặc SimpleNamespace)
        target_node       : node phục vụ (UAV, RSU, hoặc chính source nếu Local)
        config            : config object (từ get_config())
        cache_mode        : 0=miss, 1=direct_hit, 2=transcoding
        all_uavs          : list tất cả UAV; None → [target_node]
        z_req             : bitrate yêu cầu (0=thấp, 1=cao, ...)
        z_cached          : bitrate đang cache (chỉ dùng khi cache_mode=2)
        num_uavs          : số UAV (để tính backhaul rate chia đều)
        rsus              : list RSU/MBS node thật (dùng vị trí cho backhaul)
        num_users_per_uav : số user runtime trên UAV phục vụ
    """
    _all_uavs = all_uavs if all_uavs is not None else [target_node]
    mbs_node  = rsus[0] if rsus else None

    if cache_mode == 1:
        return _delay_direct_hit(
            target_node, source_node, _all_uavs, config,
            z_req=z_req,
            num_users_per_uav=num_users_per_uav,
        )

    elif cache_mode == 2:
        return _delay_transcoding(
            target_node, source_node, _all_uavs, config,
            z_req=z_req, z_cached=z_cached,
            num_users_per_uav=num_users_per_uav,
        )

    else:
        return _delay_cache_miss(
            target_node, source_node, _all_uavs, config,
            z_req=z_req, num_uavs=num_uavs,
            mbs_node=mbs_node,
            num_users_per_uav=num_users_per_uav,
        )
