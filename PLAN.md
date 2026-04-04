# PLAN: Mở rộng hệ thống SDN‑VANET/UAV sang **video truyền thực** (Best‑of‑breed, production‑grade)

> Mục tiêu: giữ nguyên lõi RL + mô hình toán học hiện có, mở rộng thêm một nhánh **real streaming** để đo/quan sát QoE thực tế (lag, giật, rebuffer) và dùng các số đo này để đánh giá/tinh chỉnh policy.

---

## 0) Tuyên bố lựa chọn kỹ thuật (chốt cứng, không phương án thay thế)

Vì yêu cầu của bạn là “chọn cái tốt nhất, không chọn vì dễ”, kế hoạch này **chốt** stack như sau:

1. **Chuẩn streaming:** **MPEG‑DASH + CMAF/fMP4**
2. **Origin server:** **Nginx (HTTP/2 + sendfile + aio + tuned kernel)**
3. **UAV edge cache:** **Nginx reverse proxy cache** (cache key theo URI segment, cache lock, stale‑while‑revalidate)
4. **ABR player phía xe:** **dash.js** chạy trong **Chromium headless** + Telemetry API
5. **QoE telemetry bus:** **OpenTelemetry + Prometheus + Grafana + Loki**
6. **Per‑segment truth store:** **TimescaleDB (PostgreSQL extension)**
7. **RL orchestration:** Giữ **D3QN/Ryu/Mininet‑WiFi** hiện có, bổ sung **RealQoEAdapter** để bridge reward
8. **Video codec ladder:** **H.265/HEVC + fallback H.264**, fixed GOP 2s, aligned keyframe cho tất cả representation
9. **Test & reproducibility:** **k6 + Playwright + pytest + deterministic seeds + scenario manifests**
10. **Deployment:** **Docker Compose + Ansible** (dev/prod profile), CI trên GitHub Actions

Lý do chọn: đây là tổ hợp mạnh nhất cho nghiên cứu ABR edge caching hiện đại, đủ chuẩn học thuật lẫn demo thực chiến, tối ưu cả quan sát realtime và dữ liệu hậu kiểm.

---

## 1) Mục tiêu nghiệm thu (Definition of Done)

Hệ thống được coi là hoàn thành khi đạt đồng thời toàn bộ tiêu chí:

1. Xe có thể **xem video thực** qua UAV path theo quyết định RL.
2. Quan sát được trực quan hiện tượng **lag/giật/rebuffer** trên player.
3. Log được đầy đủ per‑segment:
   - startup delay, stall count, stall duration,
   - selected bitrate, bitrate switch magnitude,
   - throughput estimate, download time, RTT/jitter/loss.
4. Đối sánh được:
   - `QoE_model` (mô hình toán) vs `QoE_real` (đo thực)
   - sai số MAE/MAPE theo epoch và theo scenario.
5. Có dashboard live và report offline reproducible.
6. Có benchmark chuẩn với baseline QEA và policy D3QN hiện tại.

---

## 2) Kiến trúc mục tiêu (Target Architecture)

### 2.1 Luồng dữ liệu end‑to‑end

`Origin(MBS) -> (backhaul) -> UAV Cache -> Car Player`

- Agent quyết định `(uav_idx, z_cached, cache_dec)` như hiện trạng.
- Nếu cache miss tại UAV:
  - UAV cache proxy fetch segment từ Origin tại MBS,
  - sau đó serve xuống car.
- Player nhận segment DASH, phát realtime, phát sinh telemetry.
- Telemetry được đẩy vào OTEL collector, Prometheus/TimescaleDB.
- RealQoEAdapter tổng hợp thành reward dùng cho eval/fine‑tune.

### 2.2 Thành phần triển khai

- **control/**: Ryu app + policy runtime + REST bridge
- **streaming/origin/**: Nginx origin + DASH assets
- **streaming/uav-cache/**: Nginx cache profile cho mỗi UAV
- **client/car-player/**: dash.js client runner (headless + optional GUI)
- **observability/**: OTEL Collector, Prometheus, Grafana, Loki
- **data/**: TimescaleDB schema cho per‑segment events
- **adapter/**: RealQoEAdapter (Python) nối telemetry -> reward

---

## 3) Thiết kế dữ liệu & metric (bắt buộc)

### 3.1 Schema bảng TimescaleDB

#### `segment_events`
- `ts` timestamptz
- `run_id` text
- `episode_id` int
- `slot_id` int
- `car_id` text
- `uav_id` text
- `video_id` int
- `segment_idx` int
- `bitrate_kbps` int
- `download_ms` double
- `size_bytes` bigint
- `throughput_kbps` double
- `buffer_before_s` double
- `buffer_after_s` double
- `rebuffer_s` double
- `switch_delta_mbps` double
- `cache_hit` boolean
- `cache_mode` smallint
- `http_status` smallint

#### `session_events`
- `ts`, `run_id`, `car_id`, `event_type`
- `startup_delay_ms`, `total_rebuffer_s`, `stall_count`, `avg_bitrate_kbps`, `qoe_real`

### 3.2 QoE thực (chuẩn hóa dùng xuyên suốt)

`QoE_real = U(bitrate) - α * rebuffer_s - β * |Δbitrate_Mbps| - γ * startup_s`

Cố định hệ số:
- `α = 4.3`
- `β = 1.0`
- `γ = 1.5`

Ghi chú: giữ đồng nhất triết lý QoE hiện tại để so sánh công bằng.

---

## 4) Kế hoạch triển khai theo pha (chi tiết)

## PHA 1 — Streaming nền tảng chuẩn production

### 4.1 Chuẩn bị video ladder

- Input: 1–3 clip đại diện (motion thấp/trung bình/cao), mỗi clip 2–5 phút.
- Encode ladder:
  - 240p: 300 kbps
  - 360p: 750 kbps
  - 480p: 1200 kbps
  - 720p: 1850 kbps
- Cấu hình encode:
  - GOP = 2s, keyint đồng bộ giữa all reps
  - scene‑cut control nghiêm ngặt
  - segment duration = 2s
- Đóng gói DASH CMAF với MPD chuẩn, index aligned.

### 4.2 Origin server tại MBS

- Nginx tune:
  - `worker_processes auto`
  - `sendfile on`, `tcp_nopush on`, `aio threads`
  - HTTP/2 bật mặc định
  - access log JSON cho tracing per segment
- Cấu trúc URL cố định:
  - `/dash/{video_id}/manifest.mpd`
  - `/dash/{video_id}/{rep}/{segment}.m4s`

### 4.3 UAV cache reverse proxy

Mỗi UAV chạy 1 Nginx cache instance:
- `proxy_cache_path` riêng theo UAV
- `proxy_cache_lock on`
- `proxy_cache_use_stale updating error timeout http_500 http_502 http_503 http_504`
- `proxy_cache_valid 200 206 1h`
- `slice` theo segment để tối ưu reuse

Nghiệm thu pha 1:
- Car request video qua UAV URL.
- Cache warm/cold behavior đúng.
- So sánh latency hit vs miss rõ ràng.

---

## PHA 2 — Player thật trên xe + QoE telemetry chuẩn

### 4.4 Car Player runtime

- Chromium headless + dash.js custom page.
- Bật APIs và hook events:
  - `PLAYBACK_STARTED`
  - `BUFFER_EMPTY`
  - `BUFFER_LOADED`
  - `QUALITY_CHANGE_RENDERED`
  - `FRAGMENT_LOADING_*`
- Mỗi event gắn metadata:
  - `run_id, episode_id, slot_id, car_id, uav_id, segment_idx`

### 4.5 Telemetry pipeline

- Dash.js client gửi JSON events qua HTTP/gRPC tới OTEL Collector.
- Collector route:
  - Metrics -> Prometheus
  - Logs -> Loki
  - Events -> TimescaleDB (qua ingester service)

### 4.6 Dashboard bắt buộc

Grafana dashboard gồm:
1. Buffer level theo thời gian (mỗi car)
2. Rebuffer spikes & stall count
3. Bitrate timeline và switch magnitude
4. Cache hit ratio theo UAV
5. Segment download time P50/P95/P99
6. QoE_real theo episode

Nghiệm thu pha 2:
- Demo trực tiếp nhìn thấy video phát và stall.
- Dashboard phản ánh đúng hiện tượng trên player theo timestamp.

---

## PHA 3 — Bridge vào RL loop (hybrid reward)

### 4.7 RealQoEAdapter

Tạo module `src/real_qoe_adapter.py` với trách nhiệm:
- Lấy telemetry per segment của request vừa hoàn thành.
- Tính `qoe_real` theo công thức chuẩn.
- Trả về payload cho environment:
  - `download_time_s_real`
  - `rebuffer_s_real`
  - `switch_mbps_real`
  - `qoe_real`

### 4.8 Chế độ reward

Chốt 3 mode trong config:
1. `reward_mode = model` -> chỉ dùng mô hình toán
2. `reward_mode = hybrid` -> `R = λ*R_model + (1-λ)*R_real`, với `λ=0.3`
3. `reward_mode = real` -> chỉ dùng QoE thực

Mặc định nghiên cứu mở rộng dùng `hybrid` để ổn định học.

### 4.9 Đồng bộ action‑decision với stream path

- Sau mỗi action `(uav_idx, z_cached, cache_dec)`:
  - Control plane cập nhật route/car association theo UAV mục tiêu
  - Player session của car chuyển sang endpoint UAV tương ứng
- Gắn `decision_id` để join được action log và telemetry log một‑một.

Nghiệm thu pha 3:
- Reward trong loop phản ánh đúng telemetry thực.
- Hệ thống train/eval chạy liên tục không deadlock.

---

## PHA 4 — Thực nghiệm khoa học & báo cáo

### 4.10 Bộ scenario chuẩn

Thiết kế tối thiểu 12 kịch bản:
- Mật độ xe: thấp/vừa/cao
- Tốc độ xe: thấp/vừa/cao
- Tỷ lệ cache warm: thấp/vừa/cao
- Mức nhiễu vô tuyến: thấp/cao

Mỗi scenario chạy:
- 5 seeds cố định
- thời lượng >= 30 phút hoặc >= N episodes ổn định

### 4.11 Chỉ số đánh giá chính

1. QoE_real mean, P50/P95
2. Stall ratio, stall duration
3. Startup delay
4. Bitrate instability index
5. Cache hit ratio per UAV
6. Backhaul offload reduction (%)
7. Model‑to‑real gap: MAE/MAPE

### 4.12 Bảng so sánh bắt buộc

- QEA baseline
- D3QN (model reward)
- D3QN (hybrid reward)
- D3QN (real reward fine‑tune)

Nghiệm thu pha 4:
- Có bảng/đồ thị đầy đủ, kết luận định lượng rõ ràng.

---

## 5) Kế hoạch công việc theo tuần (10 tuần)

## Tuần 1
- Chuẩn hóa asset pipeline encode + DASH packaging.
- Dựng Origin Nginx + validate throughput.

## Tuần 2
- Dựng UAV cache instances + cache policy tuning.
- Hoàn thiện routing path MBS->UAV->Car.

## Tuần 3
- Tạo player page dash.js + event hooks chuẩn.
- Chạy 1 car end‑to‑end với video thật.

## Tuần 4
- Xây telemetry ingest service + OTEL collector.
- Lưu được per‑segment events vào TimescaleDB.

## Tuần 5
- Dựng Grafana dashboards production.
- Kiểm thử timestamp alignment giữa video và logs.

## Tuần 6
- Viết `RealQoEAdapter` + unit tests.
- Tích hợp `reward_mode=model/hybrid/real`.

## Tuần 7
- Đồng bộ decision/action với endpoint switching.
- Chạy smoke tests nhiều cars, xử lý race conditions.

## Tuần 8
- Chạy battery benchmark trên 12 scenarios.
- Thu thập dữ liệu, cleanup outliers theo protocol.

## Tuần 9
- Phân tích model‑real gap, hiệu chỉnh λ hybrid.
- Chốt bảng so sánh với QEA + D3QN.

## Tuần 10
- Freeze artifacts, viết chapter thực nghiệm,
- Chuẩn bị script demo 15 phút + bản backup offline.

---

## 6) Kế hoạch test/QA chi tiết

### 6.1 Test cấp hệ thống

1. **Functional streaming test**: manifest/segment load đúng bitrate ladder.
2. **Cache correctness test**: hit/miss/transcoding metadata đúng.
3. **QoE integrity test**: event timeline khớp với playback thực.
4. **Reward integrity test**: reward adapter trả giá trị hợp lệ mọi step.
5. **Soak test 6h**: không memory leak, không telemetry drop > 0.1%.

### 6.2 Test cấp hiệu năng

- Segment fetch p95 < ngưỡng theo scenario.
- Dashboard refresh < 2s.
- Ingest throughput >= 5k events/s.

### 6.3 Reproducibility protocol

- Mỗi run có `run_manifest.yaml` chứa:
  - git commit hash
  - seed
  - scenario params
  - model checkpoint
- Artifact lưu bất biến: logs, metrics dump, plots, raw CSV.

---

## 7) Quản trị rủi ro (chốt biện pháp xử lý)

1. **Player drift giữa cars**
   - Biện pháp: NTP nội bộ + server‑side timestamp authority.
2. **Noisy wireless làm reward dao động mạnh**
   - Biện pháp: EWMA smoothing, confidence interval, multi‑seed averaging.
3. **Data pipeline bottleneck**
   - Biện pháp: async batching + backpressure + retention policy.
4. **Demo live thất bại do môi trường**
   - Biện pháp: chuẩn bị rehearsal bundle + recorded fallback có telemetry đồng bộ.

---

## 8) Deliverables cuối cùng

1. Hệ thống chạy video thật trên xe trong topology SDN‑VANET/UAV.
2. Dashboard live QoE + cache + delay.
3. Bộ kết quả benchmark đầy đủ (raw + processed).
4. Báo cáo phân tích model‑real gap và lợi ích hybrid reward.
5. Bộ script tái lập thí nghiệm một lệnh.

---

## 9) Quyết định triển khai ngay (Action List)

1. Tạo thư mục:
   - `streaming/origin/`
   - `streaming/uav-cache/`
   - `client/car-player/`
   - `observability/`
   - `src/real_qoe_adapter.py`
2. Thêm cờ config:
   - `enable_real_streaming=true`
   - `reward_mode=hybrid`
   - `hybrid_lambda=0.3`
3. Dựng compose stack cho origin/cache/otel/prometheus/grafana/timescaledb.
4. Tạo dashboard JSON versioned trong repo.
5. Chạy pilot scenario 1 car -> 2 cars -> 5 cars.

---

## 10) Kỳ vọng kết quả sau mở rộng

- Không chỉ “mô phỏng hợp lý” mà còn chứng minh được “triển khai khả thi”.
- Có bằng chứng thị giác (xem video thật) + định lượng (QoE/lag/stall).
- Tăng độ thuyết phục khi bảo vệ: từ digital twin sang cyber‑physical validation.

---

## 11) Checklist nghiệm thu nhanh

- [ ] Xe xem được video thực qua UAV endpoint
- [ ] Quan sát được lag/giật trực tiếp
- [ ] Thu được telemetry per‑segment đầy đủ
- [ ] Reward mode `hybrid` chạy ổn định
- [ ] Dashboard live hoạt động
- [ ] Có báo cáo so sánh QEA vs D3QN(model/hybrid/real)

---

**Kết luận:** Đây là kế hoạch “best possible” theo tiêu chí chất lượng nghiên cứu + khả năng demo thực chiến. Không ưu tiên dễ làm; ưu tiên đúng, mạnh, và bảo vệ thuyết phục.
