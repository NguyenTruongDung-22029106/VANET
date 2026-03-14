#!/usr/bin/env python3
"""
plot_comparison.py — Vẽ đồ thị so sánh D3QN vs QEA từ kết quả đã chạy.

Cách dùng:
    python3 plot_comparison.py

Yêu cầu (chạy theo thứ tự trước):
    1. algo_mode = 'drl'       → results/drl_training.csv
    2. algo_mode = 'drl_eval'  → results/drl_eval.csv
    3. algo_mode = 'qea'       → results/qea_result.csv

Xuất ra:
    results/comparison_delay.png      — D3QN eval vs QEA bar chart + line
    results/comparison_training.png   — D3QN learning curve vs QEA baseline
    results/comparison_summary.txt    — Tóm tắt số liệu
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

C_D3QN = '#2196F3'
C_QEA  = '#F44336'


# ─────────────────────────────────────────────────────────────────────────────
# Đọc CSV
# ─────────────────────────────────────────────────────────────────────────────

def _load_drl_training(path):
    """
    Đọc results/drl_training.csv
    Cột: epoch, steps, total_reward, timestamp
    Trả về: list of (epoch, avg_delay)
    avg_delay = -total_reward / steps
    """
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch        = int(row['epoch'])
                steps        = int(row['steps'])
                total_reward = float(row['total_reward'])
                avg_delay    = -total_reward / max(steps, 1)
                rows.append((epoch, avg_delay))
            except (KeyError, ValueError):
                continue
    return rows


def _load_drl_eval(path):
    """
    Đọc results/drl_eval.csv
    Cột: step, delay
    Trả về: list of delay values
    """
    delays = []
    if not os.path.exists(path):
        return delays
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                delays.append(float(row['delay']))
            except (KeyError, ValueError):
                continue
    return delays


def _load_qea_result(path):
    """
    Đọc results/qea_result.csv
    Cột: generation, f_best
    Trả về: (convergence_list, final_delay)
    """
    convergence = []
    if not os.path.exists(path):
        return convergence, None
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                convergence.append(float(row['f_best']))
            except (KeyError, ValueError):
                continue
    final = convergence[-1] if convergence else None
    return convergence, final


# ─────────────────────────────────────────────────────────────────────────────
# Vẽ đồ thị
# ─────────────────────────────────────────────────────────────────────────────

def plot_bar_comparison(d3qn_eval_delay, qea_delay, out_dir):
    """Hình 1: Bar chart so sánh avg delay cuối cùng."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ['D3QN\n(sau training)', 'QEA\n(offline)']
    values = [d3qn_eval_delay, qea_delay]
    colors = [C_D3QN, C_QEA]

    bars = ax.bar(labels, values, color=colors, width=0.4,
                  edgecolor='black', linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f'{val:.4f}s', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    pct = (d3qn_eval_delay - qea_delay) / qea_delay * 100
    sign = '+' if pct > 0 else ''
    ax.set_title(f'So sánh Avg Delay: D3QN vs QEA\n'
                 f'D3QN {sign}{pct:.1f}% so với QEA',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg Delay (s)', fontsize=12)
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    path = os.path.join(out_dir, 'comparison_delay.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  → Lưu: {path}')
    return path


def plot_training_curve(training_rows, qea_delay, out_dir):
    """Hình 2: D3QN learning curve (avg delay per epoch) vs đường ngang QEA."""
    if not training_rows:
        print('  [SKIP] Không có drl_training.csv')
        return None

    epochs = [r[0] for r in training_rows]
    delays = [r[1] for r in training_rows]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, delays, color=C_D3QN, linewidth=1.5, alpha=0.7,
            label='D3QN avg delay/epoch')

    # Moving average smoothing
    if len(delays) >= 5:
        w = max(5, len(delays) // 20)
        smoothed = np.convolve(delays, np.ones(w) / w, mode='valid')
        ax.plot(epochs[w-1:], smoothed, color=C_D3QN, linewidth=2.5,
                linestyle='--', label=f'D3QN MA({w})')

    if qea_delay is not None:
        ax.axhline(y=qea_delay, color=C_QEA, linewidth=2,
                   linestyle='-.', label=f'QEA = {qea_delay:.4f}s')
        # Tô vùng D3QN tốt hơn QEA
        arr = np.array(delays)
        better = arr < qea_delay
        ax.fill_between(epochs, delays, qea_delay,
                        where=better, alpha=0.12, color=C_D3QN,
                        label='D3QN tốt hơn QEA')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Avg Delay (s)', fontsize=12)
    ax.set_title('D3QN Learning Curve vs QEA Baseline', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, 'comparison_training.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  → Lưu: {path}')
    return path


def plot_qea_convergence(convergence, out_dir):
    """Hình 3: QEA convergence curve."""
    if not convergence:
        print('  [SKIP] Không có qea_result.csv')
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    gens = list(range(1, len(convergence) + 1))
    ax.plot(gens, convergence, color=C_QEA, linewidth=2, label='QEA f_best')
    ax.fill_between(gens, convergence, alpha=0.15, color=C_QEA)
    ax.axhline(y=convergence[-1], color='#333', linewidth=1,
               linestyle='--', alpha=0.5,
               label=f'Final = {convergence[-1]:.4f}s')
    ax.set_xlabel('Thế hệ (Generation)', fontsize=12)
    ax.set_ylabel('D_tot (s)', fontsize=12)
    ax.set_title('QEA Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, 'comparison_qea_convergence.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  → Lưu: {path}')
    return path


def plot_eval_distribution(eval_delays, qea_delay, out_dir):
    """Hình 4: Phân phối delay của D3QN eval (histogram) vs QEA."""
    if not eval_delays:
        print('  [SKIP] Không có drl_eval.csv')
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(eval_delays, bins=40, color=C_D3QN, alpha=0.7, edgecolor='white',
            label=f'D3QN eval (n={len(eval_delays)})')
    ax.axvline(x=np.mean(eval_delays), color=C_D3QN, linewidth=2,
               linestyle='--', label=f'D3QN avg = {np.mean(eval_delays):.4f}s')
    if qea_delay is not None:
        ax.axvline(x=qea_delay, color=C_QEA, linewidth=2,
                   linestyle='-.', label=f'QEA = {qea_delay:.4f}s')
    ax.set_xlabel('Delay (s)', fontsize=12)
    ax.set_ylabel('Số lượng step', fontsize=12)
    ax.set_title('Phân phối Delay — D3QN Eval vs QEA', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, 'comparison_eval_dist.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  → Lưu: {path}')
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(training_rows, eval_delays, qea_delay, out_dir):
    lines = [
        '=' * 55,
        '  COMPARISON SUMMARY — D3QN vs QEA',
        '=' * 55,
        '',
    ]

    # D3QN training
    if training_rows:
        first5 = np.mean([r[1] for r in training_rows[:5]])
        last5  = np.mean([r[1] for r in training_rows[-5:]])
        impr   = (first5 - last5) / first5 * 100 if first5 > 0 else 0
        lines += [
            '  ── D3QN Training ──────────────────────────',
            f'  Epochs              : {len(training_rows)}',
            f'  Avg delay (5 ep đầu): {first5:.4f}s',
            f'  Avg delay (5 ep cuối): {last5:.4f}s',
            f'  Cải thiện qua training: {impr:.1f}%',
            '',
        ]

    # D3QN eval
    if eval_delays:
        d3qn_avg = np.mean(eval_delays)
        d3qn_std = np.std(eval_delays)
        lines += [
            '  ── D3QN Eval (load .pth, ε=0) ──────────────',
            f'  Số steps eval     : {len(eval_delays)}',
            f'  Avg delay         : {d3qn_avg:.4f}s',
            f'  Std delay         : {d3qn_std:.4f}s',
            f'  Min / Max         : {min(eval_delays):.4f}s / {max(eval_delays):.4f}s',
            '',
        ]
    else:
        d3qn_avg = None

    # QEA
    if qea_delay is not None:
        lines += [
            '  ── QEA (offline) ────────────────────────────',
            f'  D_tot tối ưu (f_best): {qea_delay:.4f}s',
            '',
        ]

    # So sánh
    if d3qn_avg is not None and qea_delay is not None:
        pct = (d3qn_avg - qea_delay) / qea_delay * 100
        sign = '+' if pct > 0 else ''
        lines += [
            '  ── Kết quả so sánh ──────────────────────────',
            f'  D3QN eval avg : {d3qn_avg:.4f}s',
            f'  QEA f_best    : {qea_delay:.4f}s',
            f'  Chênh lệch    : {sign}{pct:.1f}%  (- = D3QN tốt hơn)',
            '',
        ]
        if d3qn_avg < qea_delay:
            lines.append('  ✓ D3QN vượt QEA sau training!')
        else:
            lines.append('  ~ D3QN chưa vượt QEA — thử train thêm epoch.')

    lines += ['', '=' * 55]
    summary = '\n'.join(lines)
    print('\n' + summary)

    path = os.path.join(out_dir, 'comparison_summary.txt')
    os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\n  → Lưu: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '═' * 55)
    print('  COMPARISON: D3QN vs QEA')
    print('═' * 55)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Đọc dữ liệu
    train_path = os.path.join(RESULTS_DIR, 'drl_training.csv')
    eval_path  = os.path.join(RESULTS_DIR, 'drl_eval.csv')
    qea_path   = os.path.join(RESULTS_DIR, 'qea_result.csv')

    training_rows = _load_drl_training(train_path)
    eval_delays   = _load_drl_eval(eval_path)
    qea_conv, qea_delay = _load_qea_result(qea_path)

    print(f'\n  drl_training.csv : {len(training_rows)} epochs'
          if training_rows else '\n  drl_training.csv : CHƯA CÓ')
    print(f'  drl_eval.csv     : {len(eval_delays)} steps'
          if eval_delays else '  drl_eval.csv     : CHƯA CÓ')
    print(f'  qea_result.csv   : f_best = {qea_delay:.4f}s'
          if qea_delay else '  qea_result.csv   : CHƯA CÓ')

    if not training_rows and not eval_delays and not qea_conv:
        print('\n  [ERROR] Chưa có dữ liệu nào. Chạy main_thesis.py trước.')
        print('    1. algo_mode = "drl"      → train D3QN')
        print('    2. algo_mode = "drl_eval" → eval D3QN')
        print('    3. algo_mode = "qea"      → chạy QEA')
        sys.exit(1)

    # Vẽ
    print(f'\n  Vẽ đồ thị → {RESULTS_DIR}/')
    print('  ' + '-' * 40)

    d3qn_eval_avg = np.mean(eval_delays) if eval_delays else None

    if d3qn_eval_avg is not None and qea_delay is not None:
        plot_bar_comparison(d3qn_eval_avg, qea_delay, RESULTS_DIR)

    plot_training_curve(training_rows, qea_delay, RESULTS_DIR)
    plot_qea_convergence(qea_conv, RESULTS_DIR)
    plot_eval_distribution(eval_delays, qea_delay, RESULTS_DIR)

    write_summary(training_rows, eval_delays, qea_delay, RESULTS_DIR)

    print('\n  Hoàn thành!\n')


if __name__ == '__main__':
    main()