# Ánh xạ kiến trúc: Tài liệu → Mã nguồn

Bảng đối chiếu khái niệm trong các bài báo SDN–VANET–UAV với file/class/hàm trong project DATN.

> **Lưu ý phiên bản:** Bảng này phản ánh code **sau khi cleanup** (tháng 3/2026).
> Các thay đổi chính so với phiên bản cũ:
> - Agent đổi từ `pc_id3qn_agent.py / PC_ID3QN_Agent` → `d3qn_agent.py / D3QNAgent`
> - UAV dùng `addAccessPoint` (không phải `addAircraft`)
> - Mesh link (wlan1) đã xóa — không còn trong `main_thesis.py`
> - `models.py` hợp nhất tất cả hàm delay vào 1 API duy nhất: `calculate_total_cost()`
> - Reward đổi từ `social_welfare − cost` → `reward = -cost` (delay, giây)
> - `w_delay`, `w_cr` đã xóa khỏi `config.py`

---

## Control plane (SDN / DRL)

| Khái niệm (tài liệu) | Mã nguồn |
|----------------------|----------|
| **Control Layer (SDN Controller)** | **`src/control_layer.py`: `class ControlLayer(env, agent)`** — điều phối offloading và caching mỗi bước (`run_simulation_loop` gọi `control_layer.step()`). |
| SDN Controller (logic) | `ControlLayer` giữ env + agent; mỗi step: state → action → env.step(action) → store_experience → train; `get_decision(action_idx)` giải mã offload/bitrate/cache để log. |
| DRL-SDNC (state → action → rule) | `src/agents/d3qn_agent.py`: `D3QNAgent.select_action(state)` → action_idx |
| State gathering (vị trí, queue, channel) | `src/environment.py`: `VanetEnvironment.get_state()` — positions (cars/UAVs/RSUs), CPU load, cache status, video popularity (37 chiều) |
| Policy / Rule installation | Action = (offload_target × bitrate × cache_decision); thực thi qua `env.step(action_idx)` |
| R_total = r_d + r_e + ω·r_cr | Trong paper DRL-SDNC. Trong code: `reward = -cost`, cost = delay (giây) từ `calculate_total_cost()`; energy và social_welfare **đã bỏ khỏi hàm mục tiêu**. |

---

## Data plane

| Khái niệm (tài liệu) | Mã nguồn |
|----------------------|----------|
| Vehicles / Legacy vehicles | `net.cars` (Mininet-WiFi stations), `env.cars` |
| UAV (cloudlet / relay / aerial BS) | `net.aps` (addAccessPoint với `'uav'` trong tên), `env.uavs` |
| RSU / MEC server / GRSU | `net.aps` (addAccessPoint với `'rsu'` trong tên), `env.rsus` |
| Switch | `s1` (OVSKernelSwitch) |
| Controller (Mininet) | `c1` (RemoteController) — chỉ điều khiển switch/AP, không SDN logic |
| V2I (car–RSU/UAV) | `update_car_ap_association()` trong `main_thesis.py`; car wlan0 → AP gần nhất |
| V2V / V2U (mesh) | **Đã xóa** — mesh import và mesh link không còn trong code; liên kết qua AP thay thế |

---

## Mô hình toán (công thức)

| Công thức / Ký hiệu | Mã nguồn |
|---------------------|----------|
| Eq(1-3): LoS/NLoS path loss | `models._plos()`, `models._path_loss_dB()` (internal) |
| Eq(4): SINR_{l,k} UAV→user | `models._sinr_uav()` (internal) |
| Eq(5): r_{l,k} downlink rate | `models._rate_uav_user()` (internal) |
| Eq(6-9): backhaul rate r_{BS,l} | `models._rate_backhaul()` (internal) |
| Eq(10): D^1 direct hit delay | `models._delay_direct_hit()` (internal) |
| Eq(11): D^2 transcoding delay | `models._delay_transcoding()` (internal) |
| Eq(12): D^3 cache miss delay | `models._delay_cache_miss()` (internal) |
| Eq(13): D_{l,k} tổng delay | **`models.calculate_total_cost(source, target, config, cache_mode, ...)`** — API duy nhất ra ngoài |
| p_f (Zipf popularity) | Nằm trong `qea_joint_ca_ua.py: _zipf_popularity(F, Z, alpha)` |
| Cost (delay-only) | `calculate_total_cost()` → delay (giây); dùng trong cả D3QN lẫn QEA |

> **Lưu ý:** Các hàm `calculate_uplink_rate`, `calculate_downlink_rate`, `zipf_popularity`,
> `caching_value`, `calculate_social_welfare` **không còn tồn tại** trong `models.py` sau cleanup.
> Toàn bộ đã được hợp nhất vào `calculate_total_cost()` theo Eq(1–13) của Xie et al. 2022.

---

## Action space (Fix 2 — 3 chiều)

| Chiều | Giá trị | Mô tả |
|-------|---------|-------|
| offload_target | 0 = Local, 1..L = UAV_l, L+1 = RSU | Nơi xử lý request |
| bitrate z | 0 = 480p (low), 1 = 1080p (high) | Mức chất lượng video |
| cache_decision | 0 = no_cache, 1 = cache | Có lưu đệm tại node không |

`action_idx = offload + num_offload × (z + num_bitrates × cache)`

Topology mặc định: 3 UAV + 1 RSU → `action_size = 5 × 2 × 2 = 20`

---

## Thuật ngữ viết tắt (tài liệu ↔ code)

| Tài liệu | Code / Ghi chú |
|----------|----------------|
| ASC (Average System Cost) | `calculate_total_cost()` → delay; `reward = -delay` |
| UVCO (UAV-assisted Vehicular computation Cost Optimization) | Offload options: local=0, UAV_1..3=1..3, RSU=4 |
| CRE (Computational Resource Efficiency) | Trong paper: throughput, latency, energy. **Trong luận văn: chỉ latency (delay).** |
| PC-ID3QN (paper Chen et al.) | Cảm hứng thiết kế; cài đặt dưới tên `D3QNAgent` (Double + Dueling DQN) |
| QEA baseline | `qea_joint_ca_ua.py: QEAJointCAUA` — cùng `calculate_total_cost()` → so sánh công bằng |
| SD-MEC (Software-Defined MEC) | UAV/RSU trong data plane; logic MEC = VanetEnvironment + D3QNAgent |
| V2U, V2I | Car–UAV (qua AP association), Car–RSU (qua AP association) |
| V2V | Không mô phỏng trực tiếp trong code hiện tại |

---

*Cập nhật: tháng 3/2026 — theo code sau cleanup và SDN_VANET_UAV_Architecture_Summary.md.*
