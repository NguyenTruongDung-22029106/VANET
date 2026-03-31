# Ánh xạ kiến trúc: Tài liệu → Mã nguồn

Bảng đối chiếu khái niệm trong các bài báo SDN–VANET–UAV với file/class/hàm trong project DATN.

> **Phiên bản (3/2026):** khớp code sau khi chuẩn hóa ABR và delay theo Xie.
> - Delay Eq.(10)–(12): một API `models.calculate_total_cost()`; chunk **\(s_{f,z} = R_z \cdot T\)** với `abr_segment_duration_s` và `abr_bitrate_values_kbps` (`_chunk_size_bits`, `abr_bitrate_kbps_list`).
> - **Reward RL:** QoE ABR trong `environment.py` (utility log/linear − rebuffer − đổi bitrate) — **không** dùng \(-\log(1+\text{delay})\) hay reward Chen piecewise.
> - **State:** `2 + 7L + 2Z + 3` (ví dụ L=5, Z=4 → 48); thêm buffer norm, one-hot bitrate trước, throughput EWMA, v.v. (`get_state`).
> - **Action:** UAV-only 3 chiều `(uav_idx, z_cached, cache_dec)`; MBS không nằm trong action (chỉ backhaul khi miss).
> - QEA dùng chung `_chunk_size_bits` / `calculate_total_cost` từ `models.py`.
> - `chunk_size_MB` đã bỏ; RSU/UAV không hardcode vị trí (`plotGraph()`).

---

## Control plane (SDN / DRL)

| Khái niệm (tài liệu) | Mã nguồn |
|----------------------|----------|
| Vòng điều khiển train/eval | `main_thesis.py` (REST server + `VanetEnvironment`) khi `algo_mode` là `ryu_train` / `ryu_env`; `ryu_app.py` vòng lặp REST gọi `/step` và cài flow OpenFlow. |
| D3QN | `src/agents/d3qn_agent.py` — `select_action`, `store_experience`, `train`. |
| State | `VanetEnvironment.get_state()` trong `environment.py`. |
| Bước môi trường | `VanetEnvironment.step(action_idx)` → delay từ `calculate_total_cost`, reward QoE ABR. |
| Policy / flow | Action decode → chọn UAV + cache; Ryu đẩy flow theo bảng offload. |

---

## Data plane

| Khái niệm (tài liệu) | Mã nguồn |
|----------------------|----------|
| Vehicles | `net.cars` (Mininet-WiFi stations), `env.cars` |
| UAV | `net.aps` (`addAccessPoint`, tên chứa `'uav'`), `env.uavs` |
| RSU / MBS | `net.aps` (tên chứa `'rsu'`), `env.rsus` — backhaul trong mô hình delay |
| Switch | `s1` (OVSKernelSwitch) |
| Controller (Mininet) | `c1` (RemoteController) |
| V2I (car–UAV/RSU) | `update_car_ap_association()` trong `main_thesis.py` |
| Vị trí RSU/UAV | Không hardcode — `plotGraph()` GUI |

---

## Mô hình toán (công thức)

| Công thức / Ký hiệu | Mã nguồn |
|---------------------|----------|
| Eq(1–3): LoS/NLoS path loss | `models` (nội bộ) |
| Eq(5): \(r_{l,k}\) | `models._rate_uav_user()` |
| Eq(9): \(r_{\mathrm{BS},l}\) | `models._rate_backhaul()` |
| Eq(10–12): \(D^1,D^2,D^3\) | `models._delay_*` → **`calculate_total_cost(...)`** |
| \(s_{f,z}\) ABR (impl.) | **`models._chunk_size_bits`** = \(R_z \cdot T\) |
| \(p_{f,z}\) Zipf | `environment._compute_zipf_joint_probs`, `qea_joint_ca_ua._zipf_popularity` |

---

## Action space (3 chiều — UAV-only, Xie)

| Chiều | Giá trị | Mô tả |
|-------|---------|-------|
| `uav_idx` | `0..L-1` | UAV phục vụ |
| `z_cached` | `0..Z-1` | Mức bitrate cache tại UAV |
| `cache_decision` | `0, 1` | Có cache hay không |

**Encoding:** `action_idx = uav_idx + L × (z_cached + Z × cache_dec)`  
**Tổng:** `L × Z × 2` (mặc định 5×4×2 = **40**).

> MBS không có trong action; chỉ xuất hiện trong backhaul khi cache miss (Eq.12).

---

## Benchmark

| Chức năng | Mã nguồn |
|-----------|----------|
| So sánh D3QN vs QEA | `benchmark.py` (không cần Mininet) |
| Đồ thị / CSV | `src/results/benchmark_d3qn_vs_qea.png`, `benchmark_summary.csv` |
| Chạy | `cd src && python3 ../benchmark.py` |

---

## Thuật ngữ tài liệu ↔ code

| Tài liệu | Code / Ghi chú |
|----------|----------------|
| \(D_{l,k}\), delay segment | `calculate_total_cost()` → giây; vào buffer/rebuffer QoE |
| \(s_{f,z}\) | `R_z * T` với `abr_segment_duration_s`, `abr_bitrate_values_kbps` |
| QoE / reward (thesis) | `environment.py` — không phải `-log(1+delay)` |
| PC-ID3QN / Chen | Cảm hứng; agent: `D3QNAgent` |
| QEA | `qea_joint_ca_ua.py` — cùng `calculate_total_cost` + `_chunk_size_bits` |
| SD-MEC | UAV/RSU data plane + `VanetEnvironment` / Ryu |

---

*Cập nhật: 3/2026 — ABR \(s=R_z T\), QoE reward, state 48 dims (L=5,Z=4), bỏ `control_layer.py` (không dùng), README + `system_model_formulas.tex` (XeLaTeX).*
