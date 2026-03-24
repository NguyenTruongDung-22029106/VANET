# Ánh xạ kiến trúc: Tài liệu → Mã nguồn

Bảng đối chiếu khái niệm trong các bài báo SDN–VANET–UAV với file/class/hàm trong project DATN.

> **Lưu ý phiên bản:** Bảng này phản ánh code **sau khi cleanup + fix** (tháng 3/2026).
> Các thay đổi chính so với phiên bản cũ:
> - Agent đổi từ `pc_id3qn_agent.py / PC_ID3QN_Agent` → `d3qn_agent.py / D3QNAgent`
> - UAV dùng `addAccessPoint` (không phải `addAircraft`)
> - Mesh link (wlan1) đã xóa — không còn trong `main_thesis.py`
> - `models.py` hợp nhất tất cả hàm delay vào 1 API duy nhất: `calculate_total_cost()`
> - Reward đổi từ `social_welfare − cost` → `reward = -log(1+delay)`
> - `w_delay`, `w_cr` đã xóa khỏi `config.py`
> - **Action space mở rộng từ 2D → 3D** (thêm chiều bitrate `z_cached`)
> - `encode_action()` cập nhật nhận đủ 3 tham số `(uav_idx, z_cached, cache)`
> - `get_action_vector()` trong `D3QNAgent` cập nhật decode đúng 3 chiều
> - RSU và UAV **không còn hardcode vị trí** — đặt thủ công qua `plotGraph()` GUI
> - `benchmark.py` được implement để so sánh D3QN vs QEA không cần Mininet

---

## Control plane (SDN / DRL)

| Khái niệm (tài liệu) | Mã nguồn |
|----------------------|----------|
| **Control Layer (SDN Controller)** | **`src/control_layer.py`: `class ControlLayer(env, agent)`** — điều phối offloading và caching mỗi bước. |
| SDN Controller (logic) | `ControlLayer` giữ env + agent; mỗi step: state → action → env.step(action) → store_experience → train. |
| DRL-SDNC (state → action → rule) | `src/agents/d3qn_agent.py`: `D3QNAgent.select_action(state)` → action_idx |
| State gathering (vị trí, cache, channel) | `src/environment.py`: `VanetEnvironment.get_state()` — positions (cars/UAVs), khoảng cách xe→UAV, mức đầy cache, video popularity |
| Policy / Rule installation | Action = (uav_idx × z_cached × cache_decision) + MBS tier; thực thi qua `env.step(action_idx)` |
| Reward | `reward = -log(1 + delay)`, delay (giây) từ `calculate_total_cost()` |

---

## Data plane

| Khái niệm (tài liệu) | Mã nguồn |
|----------------------|----------|
| Vehicles / Legacy vehicles | `net.cars` (Mininet-WiFi stations), `env.cars` |
| UAV (cloudlet / relay / aerial BS) | `net.aps` (addAccessPoint với `'uav'` trong tên), `env.uavs` |
| RSU / MEC server / GRSU | `net.aps` (addAccessPoint với `'rsu'` trong tên), `env.rsus` |
| Switch | `s1` (OVSKernelSwitch) |
| Controller (Mininet) | `c1` (RemoteController) |
| V2I (car–RSU/UAV) | `update_car_ap_association()` trong `main_thesis.py` |
| V2V / V2U (mesh) | **Đã xóa** — liên kết qua AP thay thế |
| Vị trí RSU/UAV | **Không hardcode** — đặt thủ công qua `plotGraph()` GUI khi chạy |

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
| p_f (Zipf popularity) | `environment._compute_zipf_joint_probs()`, `qea_joint_ca_ua._zipf_popularity()` |
| Cost (delay-only) | `calculate_total_cost()` → delay (giây); dùng trong cả D3QN lẫn QEA |

> **Lưu ý:** Các hàm `calculate_uplink_rate`, `calculate_downlink_rate`, `zipf_popularity`,
> `caching_value`, `calculate_social_welfare` **không còn tồn tại** trong `models.py`.
> Toàn bộ đã được hợp nhất vào `calculate_total_cost()` theo Eq(1–13) của Xie et al. 2022.

---

## Action space (3 chiều — hiện tại)

| Chiều | Giá trị | Mô tả |
|-------|---------|-------|
| `uav_idx` | `0..L-1` | UAV được chọn để phục vụ |
| `z_cached` | `0..Z-1` | Mức bitrate cần cache tại UAV |
| `cache_decision` | `0, 1` | Có lưu đệm nội dung hay không |

**Encoding:** `action_idx = uav_idx + L × (z_cached + Z × cache_dec)`  
**MBS tier:** `action_idx = L × Z × 2`  
**Tổng action_size:** `L × Z × 2 + 1`

**Mặc định hiện tại** (L=5 UAV, Z=4 bitrates): `action_size = 5 × 4 × 2 + 1 = 41`

```python
# Encode (environment.py)
def encode_action(self, uav_idx, z_cached, cache):
    return uav_idx + L * (z_cached + Z * cache)

# Decode (environment.py + d3qn_agent.py — ĐỒNG BỘ)
uav_idx  = a % L
t        = a // L
z_cached = t % Z
cache    = t // Z
```

---

## Benchmark

| Chức năng | Mã nguồn |
|-----------|----------|
| So sánh D3QN vs QEA | `benchmark.py` — chạy không cần Mininet-WiFi |
| Output đồ thị | `src/results/benchmark_d3qn_vs_qea.png` |
| Output CSV | `src/results/benchmark_summary.csv` |
| Chạy | `cd src && python3 ../benchmark.py` |

---

## Thuật ngữ viết tắt (tài liệu ↔ code)

| Tài liệu | Code / Ghi chú |
|----------|----------------|
| ASC (Average System Cost) | `calculate_total_cost()` → delay; `reward = -log(1+delay)` |
| PC-ID3QN (paper Chen et al.) | Cảm hứng thiết kế; cài đặt dưới tên `D3QNAgent` (Double + Dueling DQN) |
| QEA baseline | `qea_joint_ca_ua.py: QEAJointCAUA` — cùng `calculate_total_cost()` → so sánh công bằng |
| SD-MEC (Software-Defined MEC) | UAV/RSU trong data plane; logic MEC = VanetEnvironment + D3QNAgent |
| V2U, V2I | Car–UAV (qua AP association), Car–RSU (qua AP association) |
| V2V | Không mô phỏng trực tiếp trong code hiện tại |

---

*Cập nhật: tháng 3/2026 — sau cleanup, fix action space 3D, implement benchmark.py, xóa hardcode vị trí RSU/UAV.*