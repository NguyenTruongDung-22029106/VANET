#!/usr/bin/env python3
"""
benchmark.py — So sánh D3QN vs QEA (offline, không cần Mininet).

Chạy: python3 benchmark.py
Kết quả: results/benchmark_*.png + results/benchmark_summary.txt

Dùng stub nodes giả lập topology 400×400m (khớp main_thesis.py):
  - 10 xe (cars), 3 UAV (tam giác đều), 1 RSU
  - Cùng config.py, cùng models.py → so sánh công bằng

FIXES:
  - Bug 1 FIX: run_qea() dùng F=config.num_videos (100) thay vì F=10
    → QEA và D3QN cùng bài toán F=100 video, so sánh công bằng
  - Bug 3 FIX: run_d3qn() epochs=100, steps_per_epoch=500 (50,000 steps)
    → Đủ để D3QN hội tụ với replay buffer 10,000
"""
import os
import sys
import math
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')   # không cần display
import matplotlib.pyplot as plt
from types import SimpleNamespace

# ── thêm thư mục src vào path (chạy từ bất kỳ đâu) ──────────────────────────
SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
AGENTS = os.path.join(SRC, 'agents')
if AGENTS not in sys.path:
    sys.path.insert(0, AGENTS)

from config import get_config
from environment import VanetEnvironment
from agents.d3qn_agent import D3QNAgent
from agents.qea_joint_ca_ua import QEAJointCAUA


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Tạo stub nodes (giống ryu_app._create_stub_nodes)
# ═══════════════════════════════════════════════════════════════════════════════

def make_stub_nodes(config):
    """Stub topology 400×400m khớp main_thesis.py."""
    plot_max = getattr(config, 'plot_max', 400)
    cx, cy   = plot_max / 2.0, plot_max / 2.0
    r_tri    = plot_max / 4.0
    cos30    = math.cos(math.pi / 6)
    sin30    = math.sin(math.pi / 6)
    uav_verts = [
        (cx,                   cy + r_tri),
        (cx - r_tri * cos30,   cy - r_tri * sin30),
        (cx + r_tri * cos30,   cy - r_tri * sin30),
    ]

    cars = [
        SimpleNamespace(
            name=f'car{i}',
            params={'position': (cx + (i - config.cars // 2) * 25, cy - 60)}
        )
        for i in range(1, config.cars + 1)
    ]
    rsus = [
        SimpleNamespace(
            name=f'rsu{i}',
            params={'position': (50 + (i-1)*300, 50)}
        )
        for i in range(1, config.rsus + 1)
    ]
    uavs = [
        SimpleNamespace(
            name=f'uav{i}',
            params={'position': uav_verts[(i-1) % 3]}
        )
        for i in range(1, config.uavs + 1)
    ]
    return cars, rsus, uavs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Chạy QEA
# ═══════════════════════════════════════════════════════════════════════════════

def run_qea(config, cars, rsus, uavs, t_max=100, pop_size=20):
    # ── BUG 1 FIX: dùng F=config.num_videos thay vì F=10 ──────────────────
    # Trước: F=10  → QEA tối ưu bài toán nhỏ hơn 10× D3QN → so sánh sai
    # Sau:   F=config.num_videos (=100) → cùng bài toán F=100 video
    F = int(getattr(config, 'num_videos', 100))
    Z = 2   # số mức bitrate (khớp NUM_BITRATES trong environment.py)

    print(f"\n{'='*55}")
    print(f"  [QEA] Bắt đầu tối ưu: {pop_size} cá thể × {t_max} thế hệ")
    print(f"  [QEA] F={F} video, Z={Z} bitrate  (khớp D3QN)")
    print(f"{'='*55}")
    t0 = time.time()

    qea = QEAJointCAUA(
        cars=cars, uavs=uavs, rsus=rsus,
        config=config,
        F=F,          # BUG 1 FIX: dùng F từ config thay vì hardcode F=10
        Z=Z,
        pop_size=pop_size,
        t_max=t_max,
        seed=42,
    )
    qea.optimize()

    elapsed = time.time() - t0
    print(f"  [QEA] Xong. f_best = {qea.f_best:.6f} s  ({elapsed:.1f}s)")
    return qea


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Chạy D3QN
# ═══════════════════════════════════════════════════════════════════════════════

def run_d3qn(config, cars, rsus, uavs, epochs=100, steps_per_epoch=500):
    # ── BUG 3 FIX: tăng từ 50×200=10,000 lên 100×500=50,000 steps ─────────
    # D3QN với replay buffer 10,000 cần ít nhất 50,000 steps để hội tụ.
    # 10,000 steps trước đây quá ít → đường delay trông phẳng hoàn toàn.
    print(f"\n{'='*55}")
    print(f"  [D3QN] Bắt đầu training: {epochs} epochs × {steps_per_epoch} steps")
    print(f"  [D3QN] Tổng: {epochs * steps_per_epoch:,} steps")
    print(f"{'='*55}")
    t0 = time.time()

    stations = cars + uavs + rsus
    env = VanetEnvironment(config, stations, aps=rsus, uavs_list=uavs)
    agent = D3QNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        num_offload_targets=env.num_offload_targets,
        config=config,
    )

    # Lưu kết quả
    epoch_rewards    = []   # tổng reward mỗi epoch
    epoch_avg_delay  = []   # avg delay mỗi epoch (delay = -reward)
    epoch_avg_loss   = []   # avg loss mỗi epoch
    epoch_epsilon    = []   # epsilon cuối mỗi epoch
    step_delays      = []   # delay từng step (dùng cho đồ thị chi tiết)

    for epoch in range(1, epochs + 1):
        env.reset()
        total_reward  = 0.0
        delays_epoch  = []

        for step in range(steps_per_epoch):
            state      = env.get_state()
            action_idx = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_idx)
            agent.store_experience(state, action_idx, reward, next_state, done)
            agent.train()

            delay = -reward   # reward = -delay
            total_reward  += reward
            delays_epoch.append(delay)
            step_delays.append(delay)

        avg_delay = float(np.mean(delays_epoch))
        avg_loss  = agent.get_avg_loss(last_n=steps_per_epoch)

        epoch_rewards.append(total_reward)
        epoch_avg_delay.append(avg_delay)
        epoch_avg_loss.append(avg_loss)
        epoch_epsilon.append(agent.epsilon)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [D3QN] Epoch {epoch:3d}/{epochs}  "
                  f"avg_delay={avg_delay:.4f}s  "
                  f"loss={avg_loss:.4f}  "
                  f"ε={agent.epsilon:.3f}")

    elapsed = time.time() - t0
    print(f"  [D3QN] Xong. Final avg_delay = {epoch_avg_delay[-1]:.6f}s  ({elapsed:.1f}s)")

    return {
        'epoch_rewards':   epoch_rewards,
        'epoch_avg_delay': epoch_avg_delay,
        'epoch_avg_loss':  epoch_avg_loss,
        'epoch_epsilon':   epoch_epsilon,
        'step_delays':     step_delays,
        'epochs':          epochs,
        'steps_per_epoch': steps_per_epoch,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Vẽ đồ thị so sánh
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(qea, d3qn_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs     = d3qn_results['epochs']
    epoch_list = list(range(1, epochs + 1))

    qea_delay     = qea.f_best
    d3qn_delays   = d3qn_results['epoch_avg_delay']
    d3qn_rewards  = d3qn_results['epoch_rewards']
    d3qn_loss     = d3qn_results['epoch_avg_loss']
    d3qn_epsilon  = d3qn_results['epoch_epsilon']
    qea_conv      = qea.convergence

    C_D3QN  = '#2196F3'
    C_QEA   = '#F44336'
    C_SHADE = '#BBDEFB'

    # ════════════════════════════════════════════════════════════════════════
    # Hình 1: Delay so sánh D3QN vs đường baseline QEA
    # ════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epoch_list, d3qn_delays, color=C_D3QN, linewidth=2,
            label='D3QN — avg delay/epoch')

    if epochs >= 5:
        smoothed = np.convolve(d3qn_delays, np.ones(5)/5, mode='valid')
        ax.plot(range(3, 3 + len(smoothed)), smoothed,
                color=C_D3QN, linewidth=2.5, linestyle='--', alpha=0.7,
                label='D3QN — smoothed (MA5)')

    ax.axhline(y=qea_delay, color=C_QEA, linewidth=2, linestyle='-.',
               label=f'QEA baseline = {qea_delay:.4f} s')

    d3qn_arr = np.array(d3qn_delays)
    better   = d3qn_arr < qea_delay
    ax.fill_between(epoch_list, d3qn_arr, qea_delay,
                    where=better, alpha=0.15, color=C_D3QN,
                    label='D3QN tốt hơn QEA')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Avg Delay (s)', fontsize=12)
    ax.set_title('So sánh Delay: D3QN vs QEA Baseline', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, epochs)

    plt.tight_layout()
    path1 = os.path.join(out_dir, 'benchmark_delay_comparison.png')
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  → Lưu: {path1}")

    # ════════════════════════════════════════════════════════════════════════
    # Hình 2: 4 subplot chi tiết D3QN
    # ════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('D3QN Training Chi Tiết', fontsize=15, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(epoch_list, d3qn_rewards, color=C_D3QN, linewidth=1.5)
    ax.set_title('Tổng Reward mỗi Epoch')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Reward')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epoch_list, d3qn_delays, color=C_D3QN, linewidth=1.5,
            label='D3QN avg delay')
    ax.axhline(y=qea_delay, color=C_QEA, linewidth=1.8, linestyle='-.',
               label=f'QEA = {qea_delay:.4f}s')
    ax.set_title('Avg Delay so với QEA Baseline')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Delay (s)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epoch_list, d3qn_loss, color='#9C27B0', linewidth=1.5)
    ax.set_title('Training Loss (Huber/SmoothL1)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epoch_list, d3qn_epsilon, color='#FF9800', linewidth=2)
    ax.set_title('Epsilon Decay (Explore → Exploit)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('ε')
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
    ax.fill_between(epoch_list, d3qn_epsilon, alpha=0.2, color='#FF9800')

    plt.tight_layout()
    path2 = os.path.join(out_dir, 'benchmark_d3qn_detail.png')
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"  → Lưu: {path2}")

    # ════════════════════════════════════════════════════════════════════════
    # Hình 3: QEA hội tụ
    # ════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(9, 4))
    gen_list = list(range(1, len(qea_conv) + 1))
    ax.plot(gen_list, qea_conv, color=C_QEA, linewidth=2, label='QEA f_best')
    ax.fill_between(gen_list, qea_conv, alpha=0.15, color=C_QEA)
    ax.axhline(y=qea_delay, color='#333', linewidth=1, linestyle='--',
               alpha=0.5, label=f'Final = {qea_delay:.4f}s')
    ax.set_xlabel('Thế hệ (Generation)', fontsize=12)
    ax.set_ylabel('D_tot (s)', fontsize=12)
    ax.set_title('QEA Convergence — D_tot qua từng thế hệ',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path3 = os.path.join(out_dir, 'benchmark_qea_convergence.png')
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"  → Lưu: {path3}")

    return path1, path2, path3


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Xuất summary text
# ═══════════════════════════════════════════════════════════════════════════════

def write_summary(qea, d3qn_results, out_dir):
    d3qn_delays  = d3qn_results['epoch_avg_delay']
    d3qn_rewards = d3qn_results['epoch_rewards']
    epochs       = d3qn_results['epochs']
    steps        = d3qn_results['steps_per_epoch']

    qea_delay    = qea.f_best
    d3qn_init    = float(np.mean(d3qn_delays[:5]))
    d3qn_final   = float(np.mean(d3qn_delays[-5:]))
    improvement  = (d3qn_init - d3qn_final) / d3qn_init * 100 if d3qn_init > 0 else 0
    vs_qea_pct   = (d3qn_final - qea_delay) / qea_delay * 100

    lines = [
        "=" * 55,
        "  BENCHMARK SUMMARY — D3QN vs QEA",
        "=" * 55,
        "",
        f"  Topology   : {qea.K} xe, {qea.L} UAV, 1 RSU",
        f"  Num videos : {qea.F}  (F — khớp nhau giữa QEA và D3QN)",
        f"  D3QN       : {epochs} epochs × {steps} steps/epoch = {epochs*steps:,} steps",
        f"  QEA        : {qea.t_max} thế hệ × {qea.pop_size} cá thể",
        "",
        "  ── QEA ─────────────────────────────────────",
        f"  D_tot tối ưu (f_best)   : {qea_delay:.6f} s",
        f"  Thế hệ hội tụ (ước tính): {_convergence_gen(qea.convergence)}",
        "",
        "  ── D3QN ────────────────────────────────────",
        f"  Avg delay (5 ep đầu)    : {d3qn_init:.6f} s",
        f"  Avg delay (5 ep cuối)   : {d3qn_final:.6f} s",
        f"  Cải thiện delay         : {improvement:.1f}%",
        f"  So với QEA baseline     : {'+' if vs_qea_pct>0 else ''}{vs_qea_pct:.1f}%",
        f"    (+ = D3QN còn cao hơn QEA, - = D3QN tốt hơn QEA)",
        "",
        "  ── Nhận xét ────────────────────────────────",
    ]

    if d3qn_final < qea_delay:
        lines.append(f"  ✓ D3QN đã vượt QEA sau {epochs} epoch huấn luyện")
        lines.append(f"    → Học online thích nghi tốt với topology động")
    elif improvement > 20:
        lines.append(f"  ~ D3QN đang hội tụ nhưng chưa đạt mức QEA")
        lines.append(f"    → Cần thêm epoch hoặc điều chỉnh hyperparameters")
    else:
        lines.append(f"  ! D3QN chưa hội tụ rõ ràng trong {epochs} epoch")
        lines.append(f"    → Tăng epochs hoặc giảm epsilon_decay")

    lines += [
        "",
        "=" * 55,
    ]

    summary = "\n".join(lines)
    print("\n" + summary)

    path = os.path.join(out_dir, 'benchmark_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(f"\n  → Lưu: {path}")


def _convergence_gen(conv_list):
    """Ước tính thế hệ hội tụ: khi f_best không giảm thêm >1%."""
    if not conv_list:
        return "N/A"
    final = conv_list[-1]
    threshold = final * 1.01
    for i, v in enumerate(conv_list):
        if v <= threshold:
            return str(i + 1)
    return str(len(conv_list))


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═"*55)
    print("  BENCHMARK: D3QN vs QEA (offline, không cần Mininet)")
    print("═"*55)

    config = get_config()

    # Tạo stub topology
    cars, rsus, uavs = make_stub_nodes(config)
    num_videos = getattr(config, 'num_videos', 100)
    print(f"\n  Topology   : {len(cars)} xe, {len(uavs)} UAV, {len(rsus)} RSU")
    print(f"  Num videos : {num_videos}  (F — dùng cho cả QEA và D3QN)")
    print(f"  Action space: {1+len(uavs)+len(rsus)} offload × 2 bitrate × 2 cache"
          f" = {(1+len(uavs)+len(rsus))*2*2} actions")

    # ── Chạy QEA ──────────────────────────────────────────────────────────
    # BUG 1 FIX: run_qea dùng F=config.num_videos bên trong rồi
    qea = run_qea(config, cars, rsus, uavs, t_max=100, pop_size=20)

    # ── Chạy D3QN ─────────────────────────────────────────────────────────
    # BUG 3 FIX: 100 epochs × 500 steps = 50,000 steps → đủ để hội tụ
    d3qn_results = run_d3qn(config, cars, rsus, uavs,
                             epochs=100, steps_per_epoch=500)

    # ── Vẽ đồ thị ─────────────────────────────────────────────────────────
    out_dir = os.path.join(SRC, 'results')
    print(f"\n{'='*55}")
    print(f"  Vẽ đồ thị → {out_dir}/")
    print(f"{'='*55}")
    plot_comparison(qea, d3qn_results, out_dir)

    # ── Summary ───────────────────────────────────────────────────────────
    write_summary(qea, d3qn_results, out_dir)

    print("\n  Hoàn thành. Mở thư mục results/ để xem ảnh.\n")


if __name__ == '__main__':
    main()