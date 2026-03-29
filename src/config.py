#!/usr/bin/env python3
"""

Muốn đổi tham số, sửa trực tiếp các giá trị trong hàm get_config().
"""
from types import SimpleNamespace


def get_config():
    """Trả về cấu hình mô phỏng (theo TABLE II SIMULATION PARAMETERS)."""
    return SimpleNamespace(
        # Simulation parameters
        epochs=50,
        max_steps_per_epoch=1000,

        # Topology (TABLE II)
        roads=8,
        mobility_time=1,
        plot_max=400,           # Simulation site size 400m×400m
        cars=10,               # K: Number of vehicles (Table 2: 100)
        uavs=5,                 # L: Number of UAVs (Table 2: 5)
        rsus=1,                 # 1 MBS (Mobile Base Station)
        uav_range=150.0,        # metres
        mbs_range=250.0,        # metres
        uav_speed=20.0,         # m/s
        uav_radius=200.0,       # metres

        # TABLE II: content & cache
        cache_uav_MB=750,       # C_u (MB) - 6 Gbits = 750 MBytes
        num_bitrates=4,         # Z: Number of bitrates
        num_cache_acts=2,       # Cache mode decision (0/1)

        # Content popularity (Zipf) — Chen et al. Eq(1), Xie et al. Section III-B
        num_videos     = 100,   # F: tổng số video trong thư viện (Table 2 Xie: F=100)
        zipf_exponent  = 0.8,   # độ lệch Zipf (không ghi trong Table 2, dùng chuẩn 0.8)

        # Communication model — Xie et al. Eq(1-13) + Table II Chen et al.
        B           = 20e6,     # V2U bandwidth (Hz) — 20 MHz
        Bh          = 10e6,     # V2B backhaul bandwidth (Hz) — 10 MHz
        M           = 30,       # Max users per UAV
        PUAV_dBm    = 20,       # UAV tx power (dBm)
        PBS_dBm     = 30,       # MBS tx power (dBm)
        sigma2_dBm  = -95,      # Noise power (dBm)
        H           = 100.0,    # UAV altitude (m)
        fc          = 38e9,     # Carrier frequency V2U (Hz) — 38 GHz
        d0          = 5.0,      # Reference distance (m)
        nLoS        = 2.0,      # Path loss exponent LoS
        nNLoS       = 2.4,      # Path loss exponent NLoS
        sLoS        = 5.3,      # Extra loss LoS (dB)
        sNLoS       = 5.27,     # Extra loss NLoS (dB)
        kappa       = 11.9,     # LoS probability param κ
        zeta        = 0.13,     # LoS probability param ζ (deg⁻¹)
        gamma_bs    = 2.0,      # \gamma trong Table 2: Backhaul path loss exponent
        eta_bs      = 100.0,    # \eta trong Table 2: Backhaul NLoS excess loss
        w0          = 1.0,      # Cycles/bit for transcoding
        C_comp      = 3.4e9,    # UAV computing capacity (cycles/s)
        chunk_size_MB = 8.0,    # s_{f,z=0} — khớp content_size_MB

        # Run mode
        plot=True,
        algo_mode='qea',   # DRL-first default; 'ryu_env' (Eval) | 'ryu_train' (Train) | 'qea' (baseline)
        eval_steps=5000,   # số step eval cho ryu_env (<=0 để chạy không giới hạn)
        oor_penalty_alpha=10.0,    # reward shaping theo overshoot ngoài vùng phủ
        oor_penalty_beta=1.25,     # penalty exponent
        oor_penalty_cap=5.0,       # cap để tránh penalty lấn át toàn bộ reward

        # REST env server (for Ryu training/deploy)
        rest_host='127.0.0.1',
        rest_port=8081,

        # Logging
        log_dir='results',

        # Model path
        model_path='agents/models/d3qn.pth',
    )
