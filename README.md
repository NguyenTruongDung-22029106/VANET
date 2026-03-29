# Tối ưu hóa Caching và User Association cho Video Streaming trong SDN-VANET/UAV

**Graduation Thesis Project**  
Joint Optimization of Caching and User Association for Video Streaming in SDN-VANET/UAV using DRL (D3QN), with QEA as an optional baseline.

![Architecture](https://img.shields.io/badge/Architecture-SDN%20%7C%20UAV%20%7C%20VANET-blue)
![RL](https://img.shields.io/badge/Algorithm-D3QN-green)
![Simulator](https://img.shields.io/badge/Simulation-Mininet--WiFi-orange)

---

## 1. Mục tiêu

Dự án mô phỏng hệ thống video streaming trong mạng xe có UAV/RSU, với hướng chính là DRL:

- **D3QN online (Ryu SDN controller) — phương pháp chính:** quyết định động theo trạng thái mạng.
- **QEA offline baseline (tùy chọn):** dùng để đối sánh.

Hàm mục tiêu trong implementation hiện tại là **độ trễ phục vụ nội dung (delay)** theo mô hình Xie et al. (IEEE Access 2022).

---

## 2. Kiến trúc triển khai hiện tại

- **Data plane (Mininet-WiFi):** `cars`, `uav*` (AP trên không), `rsu*` (AP mặt đất), switch `s1`.
- `cars` chỉ associate với `uav*`; `rsu*` chỉ dùng cho backhaul trong mô hình delay.
- **Control plane (Ryu):** `src/ryu_app.py` gọi REST tới môi trường trong `main_thesis.py`, sau đó cài flow OpenFlow.
- **Environment RL:** `src/environment.py`.
- **Cost model:** `src/models.py` (`calculate_total_cost`).

### State, Action, Reward

- **State:** vector có kích thước động theo số UAV trong topology.
  - `2 + L×2 + L + L + L×3 + 1 + Z` chiều (với L = số UAV, Z = số bitrate)
  - `z_req` được One-hot encode (Z chiều) để Agent nhận diện rõ mức bitrate yêu cầu
  - Mặc định L=5, Z=4: **state_size = 42**

- **Action:** 3 chiều — `uav_idx × z_cached × cache_decision` (chỉ UAV, theo Paper Xie et al.).
  - Encoding: `a = uav_idx + L × (z_cached + Z × cache_dec)`
  - Tổng action = `#UAV × #bitrate × 2`
  - Với mặc định hiện tại 5 UAV, 4 bitrate: **`5 × 4 × 2 = 40` actions**

- **Reward:** `R = -log(1 + delay)` + out-of-range shaping penalty.

- **Cache Miss:** Khi UAV không có video, dùng đường MBS→UAV (backhaul) → Car (Eq.12 Paper).

Ghi chú coverage: hệ thống không tự chuyển sang UAV gần nhất khi agent chọn sai vùng phủ.

---

## 3. Vị trí RSU và UAV

RSU và UAV **không có vị trí hardcode**. Khi `net.plotGraph()` mở lên, bạn kéo thả node trực tiếp trên GUI để đặt vị trí mong muốn. Mininet-WiFi tự cập nhật `params['position']` theo vị trí đặt, và tất cả hàm tính delay đều đọc từ đó ở runtime.

---

## 4. Chế độ chạy

`algo_mode` trong `src/config.py`:

- `ryu_train`: mở REST environment để Ryu train D3QN (epsilon > 0) — default.
- `ryu_env`: mở REST environment để Ryu eval D3QN (epsilon = 0).
- `qea`: chạy baseline QEA trong `main_thesis.py`.

---

## 5. Cài đặt môi trường

### 5.1 Yêu cầu

- Ubuntu 20.04+ (khuyến nghị)
- Python 3.8+
- Mininet-WiFi
- Ryu SDN framework
- Python packages: `torch`, `numpy`, `matplotlib`, `pandas`

### 5.2 Cài Mininet-WiFi

```bash
git clone https://github.com/intrig-unicamp/mininet-wifi
cd mininet-wifi
sudo util/install.sh -Wlnfv
```

### 5.3 Cài Python dependencies

```bash
pip install torch numpy matplotlib pandas ryu
```

---

## 6. Quick Start

Làm việc trong thư mục `src` để path model/log đúng mặc định:

```bash
cd /home/mec/DATN/src
```

### 6.1 Chạy QEA baseline

1. Sửa `src/config.py`:

```python
algo_mode = 'qea'
```

2. Chạy:

```bash
sudo python3 main_thesis.py
```

Output chính (mặc định ở `src/results/`):

- `qea_result.csv` (QEA convergence)
- `qea_eval.csv` (delay per-request)
- `qea_eval_meta.csv` (uav_idx, f_req, z_req, out_of_range cho từng request)

### 6.2 Train D3QN với Ryu

1. Sửa `src/config.py`:

```python
algo_mode = 'ryu_train'
```

2. Mở 2 terminal:

Terminal A (Mininet + REST env):

```bash
cd /home/mec/DATN/src
sudo python3 main_thesis.py
```

Terminal B (Ryu controller):

```bash
cd /home/mec/DATN/src
ryu-manager ryu_app.py
```

Output chính:

- `src/results/ryu_deploy_training.csv`
- `src/results/ryu_deploy_training.log`
- checkpoint model: `src/agents/models/d3qn.pth`

### 6.3 Eval D3QN (epsilon = 0)

1. Sửa `src/config.py`:

```python
algo_mode = 'ryu_env'
```

2. Chạy lại đúng 2 terminal như bước train.

Output:

- `src/results/ryu_deploy_eval.csv`
- `src/results/ryu_deploy_eval.log`

### 6.4 Benchmark D3QN vs QEA (không cần Mininet)

```bash
cd /home/mec/DATN/src
python3 ../benchmark.py
```

Output:

- `src/results/benchmark_summary.csv`
- `src/results/benchmark_d3qn_vs_qea.png`

---

## 7. Cấu hình quan trọng (`src/config.py`)

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `epochs` | `50` | Số epoch mô phỏng |
| `max_steps_per_epoch` | `1000` | Số step tối đa mỗi epoch |
| `cars`, `uavs`, `rsus` | `10, 5, 1` | Quy mô topology |
| `plot_max` | `400` | Kích thước vùng mô phỏng (m) |
| `algo_mode` | `'qea'` | Chế độ chạy (`qea`, `ryu_train`, `ryu_env`) |
| `eval_steps` | `5000` | Số step chạy ở `ryu_env` (`<=0` để chạy không giới hạn) |
| `rest_host`, `rest_port` | `127.0.0.1`, `8081` | REST env endpoint cho Ryu |
| `cache_uav_MB` | `750` | Dung lượng cache mỗi UAV |
| `model_path` | `'agents/models/d3qn.pth'` | Nơi lưu/load model D3QN |
| `log_dir` | `'results'` | Thư mục output của `main_thesis.py` |

---

## 8. Cấu trúc mã nguồn

```text
DATN/
├── README.md
├── benchmark.py
├── Outline/
├── References/
│   ├── SDN_VANET_UAV_Architecture_Summary.md
│   ├── architecture_mapping.md
│   ├── system_model_formulas.tex
│   └── *.pdf
└── src/
    ├── main_thesis.py
    ├── ryu_app.py
    ├── config.py
    ├── constants.py
    ├── helpers.py
    ├── environment.py
    ├── models.py
    ├── agents/
    │   ├── d3qn_agent.py
    │   ├── qea_joint_ca_ua.py
    │   └── models/d3qn.pth
    └── results/
        └── plot_results.py
```

---

## 9. Mapping nhanh tài liệu -> code

- D3QN agent: `src/agents/d3qn_agent.py`
- QEA baseline: `src/agents/qea_joint_ca_ua.py`
- RL environment: `src/environment.py`
- Delay/cost model: `src/models.py`
- Mininet topology + REST env: `src/main_thesis.py`
- Ryu controller logic: `src/ryu_app.py`
- Benchmark D3QN vs QEA: `benchmark.py`

---

## 10. Tài liệu tham khảo chính

- `References/system_model_formulas.tex`
- `References/SDN_VANET_UAV_Architecture_Summary.md`
- Xie et al., *Joint Caching and User Association Optimization for Adaptive Bitrate Video Streaming in UAV-Assisted Cellular Networks*, IEEE Access 2022. DOI: `10.1109/ACCESS.2022.3211940`

---

## 11. License / Mục đích sử dụng

Mã nguồn phục vụ học tập và nghiên cứu trong đồ án tốt nghiệp.
