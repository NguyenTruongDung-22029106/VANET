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

Lõi mô hình được tinh chỉnh theo thực nghiệm Mininet-WiFi:
- Downlink rate dùng tải runtime (`num_users_per_uav`) thay vì tham số cố định.
- Delay có xét runtime `cpu_load` (queue + compute sharing) để phản ánh trạng thái mạng động.
- Runtime load dùng hybrid metadata+fallback: xe nào có `associatedTo` thì theo metadata, xe nào thiếu metadata thì fallback coverage-distance cho chính xe đó.
- Coverage semantics theo agent (chuẩn 2D horizontal): agent chọn UAV nào thì thử offload UAV đó; nếu ngoài vùng phủ thì `out_of_range=True`, không offload thực tế và dùng fixed penalty (mặc định `no_uav_penalty=1000`).

---

## 2. Kiến trúc triển khai hiện tại

- **Data plane (Mininet-WiFi):** `cars`, `uav*` (AP trên không), `rsu*` (AP mặt đất), switch `s1`.
- **Control plane (Ryu):** `src/ryu_app.py` gọi REST tới môi trường trong `main_thesis.py`, sau đó cài flow OpenFlow.
- **Environment RL:** `src/environment.py`.
- **Cost model:** `src/models.py` (`calculate_total_cost`).

Ghi chú kiến trúc: dự án triển khai theo hướng **DRL-first**; QEA được giữ như baseline để so sánh. Cả DRL và QEA eval cùng dùng cost model Mininet-aware; trong `qea` mode, eval dùng deterministic metadata sync theo `X_best` (không chạy association daemon nền) để fairness theo policy QEA.

### State, Action, Reward

- **State:** vector có kích thước động theo số UAV trong topology.
- **Action:** `uav_idx × cache_decision` **+ 1 action MBS-tier (penalty action)**.
  - Tổng action = `(#UAV × 2) + 1`.
  - Với mặc định hiện tại 5 UAV: `5 × 2 + 1 = 11` actions.
- **Reward:** `R = -log(1 + delay)`.

Ghi chú: bitrate vẫn có trong request/content model, nhưng không là một chiều action độc lập ở code hiện tại.
Ghi chú coverage: hệ thống không tự chuyển sang UAV gần nhất khi agent chọn sai vùng phủ.

---

## 3. Chế độ chạy

`algo_mode` trong `src/config.py`:

- `ryu_train`: mở REST environment để Ryu train D3QN (epsilon > 0) — default.
- `ryu_env`: mở REST environment để Ryu eval D3QN (epsilon = 0).
- `qea`: chạy baseline QEA trong `main_thesis.py`.

---

## 4. Cài đặt môi trường

### 4.1 Yêu cầu

- Ubuntu 20.04+ (khuyến nghị)
- Python 3.8+
- Mininet-WiFi
- Ryu SDN framework
- Python packages: `torch`, `numpy`, `matplotlib`, `pandas`

### 4.2 Cài Mininet-WiFi

```bash
git clone https://github.com/intrig-unicamp/mininet-wifi
cd mininet-wifi
sudo util/install.sh -Wlnfv
```

### 4.3 Cài Python dependencies

```bash
pip install torch numpy matplotlib pandas ryu
```

---

## 5. Quick Start

Làm việc trong thư mục `src` để path model/log đúng mặc định:

```bash
cd /home/mec/DATN/src
```

### 5.1 Chạy QEA baseline

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

### 5.2 Train D3QN với Ryu

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

Nên chạy Terminal A trước, đợi log `REST env server listening...` rồi mới chạy Terminal B.

Output chính:

- `src/results/ryu_deploy_training.csv`
- `src/results/ryu_deploy_training.log`
- checkpoint model: `src/agents/models/d3qn.pth`

### 5.3 Eval D3QN (epsilon = 0)

1. Sửa `src/config.py`:

```python
algo_mode = 'ryu_env'
```

2. Chạy lại đúng 2 terminal như bước train.

Ryu sẽ load model từ `model_path` và chạy ở chế độ eval.
Kết quả eval được ghi vào:

- `src/results/ryu_deploy_eval.csv`
- `src/results/ryu_deploy_eval.log`

---

## 6. Cấu hình quan trọng (`src/config.py`)

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `epochs` | `50` | Số epoch mô phỏng |
| `max_steps_per_epoch` | `1000` | Số step tối đa mỗi epoch |
| `cars`, `uavs`, `rsus` | `20, 5, 1` | Quy mô topology |
| `plot_max` | `400` | Kích thước vùng mô phỏng (m) |
| `algo_mode` | `'ryu_train'` | Chế độ chạy (`qea`, `ryu_train`, `ryu_env`) |
| `eval_steps` | `1000` | Số step chạy ở `ryu_env` (`<=0` để chạy không giới hạn) |
| `no_uav_penalty` | `1000` | Delay penalty khi agent chọn UAV ngoài vùng phủ |
| `rest_host`, `rest_port` | `127.0.0.1`, `8081` | REST env endpoint cho Ryu |
| `cache_uav_MB` | `750` | Dung lượng cache mỗi UAV |
| `model_path` | `'agents/models/d3qn.pth'` | Nơi lưu/load model D3QN |
| `log_dir` | `'results'` | Thư mục output của `main_thesis.py` |

---

## 7. Cấu trúc mã nguồn

```text
DATN/
├── README.md
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

## 8. Mapping nhanh tài liệu -> code

- D3QN agent: `src/agents/d3qn_agent.py`
- QEA baseline: `src/agents/qea_joint_ca_ua.py`
- RL environment: `src/environment.py`
- Delay/cost model: `src/models.py`
- Mininet topology + REST env: `src/main_thesis.py`
- Ryu controller logic: `src/ryu_app.py`

---

## 9. Lưu ý khi vẽ đồ thị

Repo hiện dùng script `src/results/plot_results.py` để vẽ các hình tổng hợp.
Script này đọc các file CSV theo tên cứng (có fallback cho tên cũ). Nếu bạn đổi pipeline log hoặc đổi tên file output, cần sửa lại path trong script.

---

## 10. Tài liệu tham khảo chính

- `References/system_model_formulas.tex`
- `References/SDN_VANET_UAV_Architecture_Summary.md`
- Xie et al., *Joint Caching and User Association Optimization for Adaptive Bitrate Video Streaming in UAV-Assisted Cellular Networks*, IEEE Access 2022. DOI: `10.1109/ACCESS.2022.3211940`

---

## 11. License / Mục đích sử dụng

Mã nguồn phục vụ học tập và nghiên cứu trong đồ án tốt nghiệp.
