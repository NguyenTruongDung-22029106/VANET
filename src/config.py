#!/usr/bin/env python3
"""

Muốn đổi tham số, sửa trực tiếp các giá trị trong hàm get_config().
"""
from types import SimpleNamespace


def get_config():
    """Trả về cấu hình mô phỏng (theo TABLE II SIMULATION PARAMETERS)."""
    return SimpleNamespace(
        # Simulation parameters
        # Total training steps = epochs * max_steps_per_epoch
        epochs=50,
        max_steps_per_epoch=1000,

        # Topology (TABLE II)
        roads=8,
        mobility_time=1,
        plot_max=400,           # Simulation site size 400m×400m
        cars=20,               # K: Number of vehicles (paper uses 100; here fixed to 20)
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
        # Run mode
        plot=True,
        algo_mode='qea',   # DRL-first default; 'ryu_env' (Eval) | 'ryu_train' (Train) | 'qea' (baseline)
        eval_steps=5000,   # số step eval cho ryu_env (<=0 để chạy không giới hạn)
        
        # ── ABR QoE (buffer/rebuffer/switching) — khớp Xie: s_{f,z}=R_z*T, 1 step = 1 segment
        abr_segment_duration_s=2.0,   # T (s): thời lượng một segment (cùng cho mọi z)
        abr_init_buffer_s=2.0,
        abr_max_buffer_s=30.0,
        # QoE coefficients (tham khảo Pensieve: rebuffer penalty thường >> switching)
        abr_rebuffer_penalty=4.3,
        abr_switch_penalty=1.0,
        abr_utility='log',  # 'log' | 'linear'
        # Map z_req (0..Z-1) -> representation (low -> high)
        # 4 mức: 240p, 360p, 480p, 720p
        abr_bitrate_labels=('240p', '360p', '480p', '720p'),
        abr_bitrate_values_kbps=(300, 750, 1200, 1850),
        # EWMA throughput estimate (optional state feature)
        abr_tp_ewma_alpha=0.9,

        # REST env server (for Ryu training/deploy)
        rest_host='127.0.0.1',
        rest_port=8081,

        # ── Network / SDN (Mininet + Ryu phải khớp nhau) ─────────────────────
        controller_host='127.0.0.1',
        controller_port=6653,
        # Địa chỉ private LAN dạng hai octet đầu, ví dụ "192.168" → AP: 192.168.{k}.1/24
        private_lan_prefix='192.168',
        lan_subnet_mask='255.255.255.0',
        ap_ip_prefix_len=24,
        # Ad-hoc fallback khi không có UAV trong tầm (10.10.0.x/16)
        adhoc_ipv4_base='10.10.0',
        adhoc_route_cidr='10.10.0.0/16',
        adhoc_channel_mhz=2412,
        adhoc_ssid='adhoc',
        # Interface naming (Mininet-WiFi)
        car_wifi_iface_primary='wlan0',
        car_wifi_iface_alt='wlan1',
        ap_wifi_iface_primary='wlan0',
        ap_wifi_iface_alt='wlan1',
        # Thời gian / kiểm tra liên kết sau khi gán AP
        assoc_initial_sleep_s=1.5,
        assoc_daemon_interval_s=0.5,
        ping_after_assoc=True,
        ping_count=1,
        ping_wait_s=2,
        # Kênh Wi-Fi
        # - RSU hiện chỉ dùng 1 node → 1 channel là đủ
        rsu_wifi_channel='1',
        # Danh sách kênh cho RSU (xoay vòng theo node index nếu cần)
        rsu_wifi_channels=('1', '6', '11'),
        # - UAV có thể nhiều node → gán theo danh sách (xoay vòng nếu thiếu)
        uav_wifi_channel='5',  # fallback nếu không set uav_wifi_channels
        uav_wifi_channels=('1', '6', '11', '1', '6'),

        # ── D3QN hyperparameters ─────────────────────────────────────────────
        d3qn_memory_size=50_000,
        d3qn_gamma=0.95,
        d3qn_epsilon=1.0,
        d3qn_epsilon_min=0.01,
        # Tune decay to reach epsilon_min around ~50k training updates:
        # decay = (eps_min/eps0)^(1/N) with eps0=1.0, eps_min=0.01, N≈50000
        d3qn_epsilon_decay=0.9999079,
        d3qn_learning_rate=5e-4,
        d3qn_batch_size=256,
        d3qn_hidden_size=256,
        d3qn_grad_clip_norm=5.0,
        d3qn_target_update_interval=1000,

        # ── Ryu REST loop tuning ────────────────────────────────────────────
        ryu_meta_retries=20,
        ryu_meta_sleep_s=0.5,
        ryu_reset_retries=20,
        ryu_reset_sleep_s=0.5,
        ryu_step_retries=3,
        ryu_step_sleep_s=0.1,
        ryu_rest_timeout_s=5.0,
        ryu_loop_sleep_s=0.10,
        ryu_log_every_steps=100,
        ryu_save_every_steps=1000,

        # Logging
        log_dir='results',

        # Model path
        model_path='agents/models/d3qn.pth',

        # Car (Mininet-WiFi mobility)
        vehicle_speed_kmh=20.0,
        vehicle_range_m=50.0,

        # Ignore interfaces when extracting per-node MACs
        loopback_interfaces=('lo', 'lo0'),

        # QEA baseline tuning
        qea_generations=100,
    )
