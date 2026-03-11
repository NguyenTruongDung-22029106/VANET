# Tóm tắt kiến trúc SDN–VANET–UAV từ 7 tài liệu tham khảo

Tài liệu này trích xuất và tổng hợp các mô hình kiến trúc SDN, VANET và UAV từ 7 bài báo trong thư mục References, phục vụ luận văn *Tối ưu video streaming trong SDN-VANET sử dụng Multi-UAV & Q-Learning*.

---

## 1. A Survey on Video Streaming for Next-Generation Vehicular Networks

- **Vai trò:** Nền tảng video streaming trong mạng xe (V2X, VANET).
- **Đặc điểm:** Nhấn mạnh QoE, băng thông, độ trễ thấp; 6G, tile-based 360°, volumetric video; kiến trúc mạng xe phù hợp cho truyền video thế hệ tiếp theo.
- **Liên quan SDN-VANET-UAV:** Cung cấp bối cảnh ứng dụng (video trong xe) và yêu cầu mạng (latency, bandwidth, stability) cho kiến trúc tổng thể.

---

## 2. A Novel Cost Optimization Strategy for SDN-Enabled UAV-Assisted Vehicular Computation Offloading (Zhao et al., IEEE TITS 2021)

### Kiến trúc chính (Fig. 1 trong bài báo)

- **Control plane (SDN):**
  - **Controller** thu thập thông tin toàn cục theo thời gian thực: vị trí, tốc độ, độ dài hàng đợi của vehicles, UAV, MEC server.
  - Controller gửi thông tin này xuống từng thiết bị ở data plane để trao đổi thông tin trong vùng phủ.

- **Data plane:**
  - **Legacy vehicles** (tập N): nút có nhu cầu offload, di chuyển động.
  - **MEC server** (1): thực thi task thay vehicle; có thể bị che khuất bởi nhà cao tầng.
  - **UAV:** vừa là **cloudlet** (thực thi task), vừa là **relay** chuyển task lên MEC.
  - UAV bay theo quỹ đạo cố định, độ cao cố định; kênh Vehicle–UAV (V2U) giả định LoS.

- **Quyết định offload (Sn):**
  - Sn = 0: thực thi local;
  - Sn = 1: offload lên UAV;
  - Sn = 2: offload thẳng lên MEC;
  - Sn = 3: offload lên MEC qua UAV relay.

- **Hàng đợi:** UAV và MEC có waiting queue (FCFS); UAV có forwarding queue (FCFS) cho relay.

- **Kênh:** Công thức uplink Vehicle→UAV (Rn,u), Vehicle→MEC (Rn,M); LoS/NLoS/WLoS.

**Tóm tắt:** Kiến trúc SDN-VANET-UAV điển hình: controller tập trung, data plane gồm vehicles + UAV + MEC, offload đa lựa chọn (local / UAV / MEC / relay qua UAV).

---

## 3. DRL-Based Backbone SDN Control Methods in UAV-Assisted Networks (Song et al., Electronics 2023)

### Kiến trúc và vai trò SDN–DRL

- **UARS (Unmanned Aerial Resource System):** Môi trường UAV–MEC; các trạng thái (SINR, độ cao UAV, phổ hiệu dụng, v.v.) được thu thập để đưa vào **agent**.
-- **DRL-SDNC (SDN Controller):**
  - Phân bổ tài nguyên tính toán, băng thông, lưu trữ theo yêu cầu task, độ trễ chịu được và điều kiện mạng.
  - Sử dụng kiến trúc UAV cho trao đổi task giữa các MEC.
  - Cài đặt rule (flow/rule installation) dựa trên quan sát trạng thái và chỉ số đánh giá (tắc nghẽn, năng lực UE, hiệu năng năng lượng — *trong paper; luận văn chỉ dùng chỉ số về độ trễ/cache*).
- **Luồng:** State gathering → Agent (DRL) → Action configuration → Policy installation trong SDN controller.
- **Mục tiêu:** CRE (Computational Resource Efficiency): throughput, latency, energy (theo paper gốc).

**Tóm tắt:** SDN controller được điều khiển bởi DRL; state từ UAV-MEC, action là cấu hình mạng/chính sách. Paper gốc dùng reward delay/energy/utility, **nhưng trong luận văn này chỉ ánh xạ phần delay + caching/social contribution, không tối ưu energy.**

---

## 4. Edge Collaborative Caching Based on Incentive-Driven D3QN (PC-ID3QN) in UAV-Assisted Vehicular Networks (Chen et al., IEEE TITS 2025)

### Kiến trúc mạng và caching

- **Ba lớp nút edge caching:**
  - **MBS (Macro Base Station):** kết nối cloud, quản lý toàn bộ edge server, đồng bộ thông tin lưu trữ; đóng vai trò controller trung tâm.
  - **UAV** (tập U): base station trên không, dung lượng cache hạn chế; mỗi UAV phục vụ một vùng con, không trùng nhau.
  - **Vehicles** (tập V): vừa request vừa cung cấp nội dung (cache trên xe).

- **Bốn đường giao hàng nội dung (Fig. 2):**
  - **V2V:** xe lấy từ xe lân cận.
  - **V2U:** xe lấy từ UAV phía trên.
  - **V2U':** hợp tác giữa các UAV lân cận.
  - **V2B:** xe lấy từ MBS (base station).

- **Mô hình cache:** Zipf popularity; Topic similarity Q(f,f'); Preference level Φ; Caching value V_f = ω1·Φ + ω2·p_f; cơ chế incentive (social contribution) cho D3QN.

**Tóm tắt:** Kiến trúc VANET + UAV + MBS rất gần với đề tài: caching phân tầng (vehicle–UAV–MBS), D3QN + user preference, có thể ánh xạ sang mô hình PC-ID3QN và state/action/reward trong code.

---

## 5. Energy Optimization in Dual-RIS UAV-Aided MEC-Enabled Internet of Vehicles (Michailidis et al., Sensors 2021)

### Kiến trúc IoV–MEC–UAV–RIS

- **Thành phần:**
  - **Vehicles (K):** offload task latency-critical lên MEC.
  - **GRSU (Ground RSU):** MEC tại mặt đất, nguồn lưới.
  - **ARSU (Aerial RSU):** UAV mang MEC, vừa tính toán vừa relay decode-and-forward lên GRSU.
  - **Dual-RIS:** hai RIS hỗ trợ kênh vehicle–ARSU và ARSU–GRSU (phase error được mô hình).

- **Luồng:** Vehicle → (RIS) → ARSU (MEC) → phần còn lại relay → (RIS) → GRSU.
- **Tối ưu:** Giảm WTEC (weighted total energy consumption), ràng buộc công suất, lịch slot, phân bổ task.

**Tóm tắt:** Bổ sung ý tưởng RSU mặt đất + RSU trên không (UAV) + MEC; có thể đối chiếu RSU trong Mininet-WiFi với GRSU, UAV với ARSU (MEC/relay).

---

## 6. Integrating Mobile Edge Computing into UAV Networks: An SDN-Enabled Architecture (Lin et al., IEEE IoT Magazine 2021)

### Bốn kiến trúc UAV–MEC (Fig. 1)

1. **Relay MU–Cloud:** UAV chỉ relay giữa mobile user và cloud (độ trễ lớn).
2. **Relay MU–Terrestrial/Aerial MEC:** UAV relay request tới MEC gần hơn.
3. **UAV as MEC server:** UAV tự xử lý task nhẹ.
4. **Distributed MEC (end–edge–cloud):** Nhiều UAV MEC hợp tác, phân tán task.

### Kiến trúc SDN cho multi-UAV MEC (Fig. 2)

- **Data layer:** Thu thập dữ liệu từ WiFi, ad hoc, vehicular networks; xác định QoS/QoE và vị trí; MU không điều khiển mạng.
- **Edge layer:**
  - **SD-MEC (Software-Defined MEC):** các UAV MEC, giao tiếp với nhau qua kênh riêng, được gán cho các LAN theo chức năng/vị trí.
  - **SDN controller (UAV-based):** một controller quản lý một cụm SD-MEC; không nhận request trực tiếp từ MU, chỉ điều khiển SD-MEC; đồng bộ state giữa các controller qua west/eastbound.
- **Relay layer:** SD-MEC chuyển request/data lên cloud qua LEO hoặc relay (stratosphere), tạo kiến trúc end–edge–cloud.

### Smart function table (tương tự flow table OpenFlow)

- **Smart data delivery table:** Chuyển/đồng bộ dữ liệu giữa các UAV (ví dụ UAV chuyên lưu trữ vs UAV tính toán).
- **Smart operation table:** Triển khai hành động điều khiển (ví dụ flying policy) từ controller xuống SD-MEC.

**Tóm tắt:** Chuẩn hóa kiến trúc SDN cho UAV–MEC: tách data / edge (SD-MEC + controller) / relay; rất hữu ích để mô tả control plane (SDN) và data plane (UAV, RSU, vehicles) trong luận văn.

---

## 7. Joint Caching and User Association Optimization for Adaptive Bitrate Video Streaming in UAV-Assisted Cellular Networks (Xie et al., IEEE Access 2022)

### Hệ thống

- **Thành phần:** 1 BS mặt đất, L UAV, K user. UAV vừa relay vừa có cache + computing.
- **Backhaul:** BS–core qua fiber; BS–UAV qua wireless backhaul hạn chế.
- **Video:** Mỗi chunk có Z bitrate; ba trường hợp: direct hit, transcoding hit, cache miss (công thức delay D^1, D^2, D^3 trong `system_model_formulas.tex`).
- **Tối ưu:** Joint caching + user association → giảm total content delivery delay; NLIP, giải bằng QEA.

**Tóm tắt:** Trực tiếp cho video streaming + caching + user association trong mạng UAV; công thức delay và cache đã được đưa vào `system_model_formulas.tex` và có thể dùng trong `models.py` / `environment.py`.

---

## Bảng đối chiếu nhanh: Kiến trúc SDN–VANET–UAV

| Thành phần        | Vai trò trong kiến trúc                                                                 |
|-------------------|-----------------------------------------------------------------------------------------|
| **SDN Controller**| Tập trung: thu thập state (vị trí, queue, channel), gửi policy/rule xuống data plane.   |
| **Data plane**    | Vehicles, RSU (AP), UAV (MEC/relay/cache), (tùy bài) MEC server / BS.                   |
| **UAV**           | Cloudlet, relay, aerial BS/AP, cache, MEC; quỹ đạo cố định hoặc tối ưu.                |
| **VANET**         | V2V, V2I (RSU), V2U; mobility động; mesh (wlan1) đã xóa khỏi code hiện tại.             |
| **Video/Caching** | Zipf, topic/preference, hit/transcoding/miss; delay D = D_hit + D_transcoding + D_miss.|
| **DRL**           | State: network/cache/queue; Action: offload/cache/association; Reward trong paper: delay/energy/utility; **trong luận văn: delay + caching/social contribution (không có energy).** |

---

## Gợi ý ánh xạ sang mã nguồn luận văn

- **Controller SDN (logic):** Có thể mô phỏng bằng agent DRL: state từ `VanetEnvironment`, action từ `PC_ID3QN_agent`, “rule” là quyết định offload/cache/association.
- **Data plane:** `main_thesis.py`: cars (stations), RSU (AP), UAV (aircrafts), switch; kết nối V2I qua RSU, mesh (wlan1) cho V2V/V2U.
- **Công thức:** `References/system_model_formulas.tex` và `src/models.py` (cost, social welfare, delay).
- **Agent:** `src/agents/d3qn_agent.py` (`D3QNAgent`): lấy ý tưởng từ PC-ID3QN (Chen et al.) và DRL backbone (Song et al.), **rút gọn còn reward = -delay (giây); bỏ energy, social_welfare, w_delay, w_cr.**

---

*Tài liệu được tạo từ nội dung trích xuất (pdftotext) của 7 file PDF trong References. Ngày tham chiếu: 2026-01-31.*
