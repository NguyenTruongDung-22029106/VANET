#!/usr/bin/env python3
"""
Configuration module: trả về một object cấu hình tĩnh (không dùng cờ/argparse).

Muốn đổi tham số, sửa trực tiếp các giá trị trong hàm get_config().
"""
from types import SimpleNamespace


def get_config():
    """Trả về cấu hình mô phỏng (theo TABLE II SIMULATION PARAMETERS)."""
    return SimpleNamespace(
        # Simulation parameters
        epochs=100,
        max_steps_per_epoch=1000,

        # Topology (TABLE II)
        roads=8,
        mobility_time=1,
        plot_max=400,           # Simulation site size 400m×400m
        cars=10,                # Number of vehicles
        uavs=3,                 # Number of UAVs
        uav_mode='hover',
        rsus=1,                 # 1 MBS (Mobile Base Station)

        # TABLE II: content & cache
        content_size_MB=8,      # s_f (Table II Chen: avg 8MB)
        cache_uav_MB=300,       # C_u (MB)
        vehicle_speed_kmh=20,
        vehicle_range_m=50,     # R (Vehicle Communication Range)

        # Communication model — Xie et al. Eq(1-13) + Table II Chen et al.
        B           = 160e6,    # V2U bandwidth (Hz) — 160 MHz
        Bh          = 60e6,     # V2B backhaul bandwidth (Hz) — 60 MHz
        M           = 30,       # Max users per UAV
        PUAV_dBm    = 30,       # UAV tx power (dBm)
        PBS_dBm     = 35,       # MBS tx power (dBm)
        sigma2_dBm  = -95,      # Noise power (dBm)
        H           = 100.0,    # UAV altitude (m)
        fc          = 5e9,      # Carrier frequency V2U (Hz) — 5 GHz
        d0          = 1.0,      # Reference distance (m)
        nLoS        = 2.0,      # Path loss exponent LoS
        nNLoS       = 2.4,      # Path loss exponent NLoS
        sLoS        = 5.3,      # Extra loss LoS (dB)
        sNLoS       = 5.27,     # Extra loss NLoS (dB)
        kappa       = 11.9,     # LoS probability param κ
        zeta        = 0.13,     # LoS probability param ζ (deg⁻¹)
        gamma_bs    = 3.5,      # Backhaul path loss exponent
        eta_bs      = 100.0,    # Backhaul NLoS excess loss
        w0          = 1.0,      # Cycles/bit for transcoding
        C_comp      = 3.4e9,    # UAV computing capacity (cycles/s)
        chunk_size_MB = 8.0,    # s_{f,z=0} — khớp content_size_MB

        # Run mode
        plot=True,
        algo_mode='drl',        # 'drl' | 'qea' | 'both' | 'drl_eval'

        # Logging
        log_dir='results',

        # Model path
        model_path='agents/models/d3qn.pth',
    )