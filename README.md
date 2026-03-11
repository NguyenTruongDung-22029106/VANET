# Tối ưu hóa Caching và Liên kết người dùng cho Video Streaming trong mạng SDN-VANET/UAV sử dụng D3QN

**Đồ án tốt nghiệp: Joint Optimization of Caching and User Association for Video Streaming in SDN-VANET/UAV using D3QN.**

![SDN-VANET Architecture](https://img.shields.io/badge/Architecture-SDN%20%7C%20UAV%20%7C%20VANET-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-D3QN-green)
![Simulation](https://img.shields.io/badge/Simulation-Mininet--WiFi-orange)

---

## 1. Giới thiệu (Introduction)

Đồ án giải quyết bài toán **Tối ưu hóa kết hợp (Joint Optimization)** giữa Edge Caching và User Association nhằm nâng cao chất lượng Video Streaming trong mạng VANET có UAV và điều khiển bởi SDN.

### Vấn đề cốt lõi
* **Thách thức:** Các phương pháp truyền thống (như QEA) giải bài toán joint caching + user association nhưng có thời gian hội tụ dài, khó đáp ứng tính thời gian thực trong mạng xe/UAV.
* **Giải pháp:** Thuật toán **D3QN (Dueling Double DQN)** ra quyết định đồng thời về:
    1. **Offload target:** Chọn nút xử lý — Local / UAV₁₋₃ / RSU.
    2. **Bitrate:** Mức chất lượng video — 0 = 480p (thấp) / 1 = 1080p (cao).
    3. **Caching:** Có lưu nội dung video tại node đó hay không.

### Vai trò của UAV trong đồ án
* **UAV không do AI điều khiển di chuyển (không tối ưu quỹ đạo).** Không gian hành động D3QN chỉ gồm Offloading + Bitrate + Caching.
* Trong mô phỏng Mininet-WiFi, UAV hoạt động theo chế độ **Hovering** — đứng yên tại đỉnh tam giác đều (vị trí cố định), đóng vai trò Aerial Base Station.

---

## 2. Các tính năng chính (Key Features)

* **Kiến trúc SDN-VANET:** Controller + Switch + RSU (Access Point) + Cars + UAV (Access Point trên không).
* **Topology 400×400m:** 10 xe, 3 UAV (tam giác đều, độ cao 100m), 1 RSU/MBS.
* **Action space 3 chiều:** `offload × bitrate × cache = 5 × 2 × 2 = 20 actions`.
* **Reward:** `R = -delay` (giây) — thống nhất với mô tả bài toán; năng lượng không được tối ưu.
* **So sánh:** D3QN (online, adaptive) vs QEA baseline (offline, evolutionary) — cùng hàm `calculate_total_cost()`.

---

## 3. Mô hình hệ thống (System Model)

* **Data Layer (Xe):** Di chuyển ngẫu nhiên trong vùng 400×400m, gửi yêu cầu video mỗi bước.
* **Edge Layer (UAV, RSU):** UAV/RSU là nút offload và cache; CPU load động ảnh hưởng delay thực sự (M/M/1 queuing penalty).
* **Control Layer (SDN Controller):** Chạy agent D3QN — lớp `ControlLayer` trong `src/control_layer.py` giữ env + agent; mỗi bước gọi `control_layer.step()` → chọn action (offload + bitrate + cache), tính reward, train.

### 3.1 Kiến trúc SDN–VANET–UAV (tham khảo)

Tóm tắt từ 7 tài liệu tham khảo: **`References/SDN_VANET_UAV_Architecture_Summary.md`**.
Bảng ánh xạ khái niệm tài liệu → mã nguồn: **`References/architecture_mapping.md`**.

### 3.2 Hàm mục tiêu
Tối thiểu hóa tổng độ trễ phân phối nội dung (Xie et al. IEEE Access 2022, Eq.10–13):

$$D_{l,k} = D^1_{l,k} + D^2_{l,k} + D^3_{l,k}$$

* $D^1$: Direct hit — cache đúng bitrate → chỉ truyền xuống.
* $D^2$: Transcoding hit — cache bitrate cao hơn → transcode + truyền.
* $D^3$: Cache miss → kéo từ backhaul MBS + truyền.

Tất cả thành phần bị ảnh hưởng bởi **CPU load** (M/M/1 queuing). **Năng lượng KHÔNG được đưa vào hàm mục tiêu.** Agent tối ưu `R = -D_{l,k}`.

---

## 4. Thuật toán D3QN (Dueling Double DQN)

* **State (37 chiều):** Vị trí xe/UAV/RSU (2D), CPU load mỗi node, cache mode mỗi node, độ phổ biến video (Zipf).
* **Action (rời rạc, 20 actions):** offload_target × bitrate × cache_decision.
  * Encoding: `idx = offload + num_offload × (z + num_bitrates × cache)`
* **Reward:** `R = -delay` (giây), delay tính bằng `calculate_total_cost()` từ `models.py`.
* **Kiến trúc mạng:** DuelingDQN — shared feature layer → value stream V(s) + advantage stream A(s,a).
* **Double DQN:** policy_net chọn action, target_net đánh giá Q-value → tránh overestimate.
* **Training:** Experience replay (10K), batch 64, epsilon decay ×0.995, target sync mỗi 200 bước, Huber loss, gradient clip max_norm=5.0.

---

## 5. Cấu trúc project

```
DATN/
├── README.md
├── Outline/                        # Đề cương, timeline
├── References/
│   ├── SDN_VANET_UAV_Architecture_Summary.md
│   ├── architecture_mapping.md
│   ├── system_model_formulas.tex
│   └── *.pdf                       # 7 bài báo tham khảo
└── src/
    ├── main_thesis.py              # Entry point Mininet-WiFi
    ├── ryu_app.py                  # Entry point Ryu SDN controller
    ├── config.py                   # Tham số tĩnh (SimpleNamespace, không dùng argparse)
    ├── environment.py              # VanetEnvironment: state 37D, action 20D, reward=-delay
    ├── models.py                   # Delay model Eq(1–13): calculate_total_cost()
    ├── control_layer.py            # ControlLayer: cầu nối env ↔ agent
    ├── benchmark.py                # So sánh D3QN vs QEA offline (không cần Mininet)
    └── agents/
        ├── d3qn_agent.py           # D3QNAgent (Dueling Double DQN)
        └── qea_joint_ca_ua.py      # QEAJointCAUA (baseline offline)
```

---

## 6. Cài đặt và chạy mô phỏng

### Yêu cầu
* **OS:** Ubuntu 20.04 LTS (khuyến nghị).
* **Python:** 3.8+.
* **Mininet-WiFi:** [mininet-wifi](https://github.com/intrig-unicamp/mininet-wifi).
* **PyTorch** (cho D3QN agent).

### Cài đặt Mininet-WiFi

```bash
git clone https://github.com/intrig-unicamp/mininet-wifi
cd mininet-wifi
sudo util/install.sh -Wlnfv
```

### Cài đặt dependencies Python

```bash
pip install torch numpy matplotlib
```

### Cấu hình tham số

Tất cả tham số chỉnh sửa trực tiếp trong **`src/config.py`** (không dùng argparse):

```python
# Ví dụ: đổi số epoch và chế độ thuật toán
epochs = 100
algo_mode = 'drl'   # 'drl' | 'qea' | 'both' | 'drl_eval'
uav_mode  = 'hover'
log_dir   = 'results'
```

### Chạy mô phỏng Mininet-WiFi

```bash
cd DATN/src
sudo python3 main_thesis.py
```

### Chạy Ryu SDN controller (thay thế)

```bash
ryu-manager ryu_app.py
```

### Chạy benchmark offline (không cần Mininet)

```bash
cd DATN/src
python3 benchmark.py
# Xuất ra results/benchmark_*.png và results/benchmark_summary.txt
```

### Test ping (trong Mininet-WiFi CLI)

```bash
pingall                         # ping tất cả nodes
car1 ping -c 2 <uav1_ip>        # xe 1 → UAV 1 (qua AP association)
exit
```

---

## 7. Tham số cấu hình chính (config.py)

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `epochs` | 100 | Số epoch huấn luyện D3QN |
| `max_steps_per_epoch` | 1000 | Số bước tối đa mỗi epoch |
| `cars` | 10 | Số xe |
| `uavs` | 3 | Số UAV |
| `rsus` | 1 | Số RSU (MBS) |
| `plot_max` | 400 | Kích thước vùng mô phỏng (m) |
| `uav_mode` | `'hover'` | Chế độ UAV: `'hover'` (đứng yên) |
| `algo_mode` | `'drl'` | `'drl'` \| `'qea'` \| `'both'` \| `'drl_eval'` |
| `log_dir` | `'results'` | Thư mục ghi log và kết quả |
| `model_path` | `'agents/models/d3qn.pth'` | Đường dẫn lưu/load model |
| `H` | 100.0 | Độ cao UAV (m) |
| `B` | 160 MHz | Băng thông V2U |
| `Bh` | 60 MHz | Băng thông backhaul V2B |
| `cache_uav_MB` | 300 | Dung lượng cache mỗi UAV (MB) |

> **Không còn tham số:** `w_delay`, `w_cr`, `no_log`, `--ryu` (đã xóa trong quá trình cleanup).

---

## 8. Kiến trúc mạng và mapping tài liệu → code

| Khái niệm tài liệu | Code |
|--------------------|------|
| SDN Controller (logic) | `ControlLayer` + `D3QNAgent` |
| Vehicles | `net.cars` (Mininet stations) |
| UAV (Aerial BS / cache) | `net.aps` — `addAccessPoint` với tên `uav*` |
| RSU / MBS | `net.aps` — `addAccessPoint` với tên `rsu*` |
| V2I / V2U association | `update_car_ap_association()` |
| Delay model Eq(1–13) | `models.calculate_total_cost()` |
| QEA baseline | `agents/qea_joint_ca_ua.py` |

---

## 9. Metrics và log kết quả

* **benchmark.py** xuất 3 file ảnh vào `results/`:
  * `benchmark_delay_comparison.png` — D3QN avg delay vs đường ngang QEA baseline.
  * `benchmark_d3qn_detail.png` — reward, delay, loss, epsilon qua các epoch.
  * `benchmark_qea_convergence.png` — f_best QEA qua từng thế hệ.
  * `benchmark_summary.txt` — số liệu tóm tắt.

---

## 10. Tài liệu tham khảo

* Công thức mô hình: `References/system_model_formulas.tex`.
* Kiến trúc: `References/SDN_VANET_UAV_Architecture_Summary.md`.
* Paper chính: Xie et al., *Joint Caching and User Association Optimization for Adaptive Bitrate Video Streaming in UAV-Assisted Cellular Networks*, IEEE Access 2022. DOI: 10.1109/ACCESS.2022.3211940

---

## 11. License & Đồ án

Đồ án tốt nghiệp — Trường Đại học — Ngành Kỹ thuật Máy tính / Điện tử Viễn thông.
Code mô phỏng dùng cho mục đích học tập và nghiên cứu.
