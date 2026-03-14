#!/usr/bin/env python3
"""
Main entry point for the Thesis Simulation (Mininet-WiFi Graph version).

Architecture (SDN–VANET–UAV):
  - Control plane (logic): DRL agent (D3QN) + VanetEnvironment; agent observes state,
    selects action (offload + cache), env computes reward (cost/welfare).
  - Data plane: Mininet-WiFi net = cars (stations), RSUs (APs), UAVs (aircrafts), switch s1,
    controller c1.

THAY ĐỔI SO VỚI BẢN CŨ:
  - Thêm _write_delay_csv(): ghi delay từng step khi drl_eval
  - run_simulation_loop() thêm tham số eval_log_path
  - Sau qea.optimize() → ghi results/qea_result.csv
  - Gọi run_simulation_loop() truyền eval_log_path khi drl_eval
"""
import os
import time
import atexit
import threading
import math
from datetime import datetime
import matplotlib.pyplot as plt
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.log import setLogLevel, info
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference

from config import get_config
from environment import VanetEnvironment
from agents.d3qn_agent import D3QNAgent
from control_layer import ControlLayer
from agents.qea_joint_ca_ua import QEAJointCAUA

# Patch tkinter để tránh RuntimeError khi thoát
try:
    import tkinter as _tk
    _after_cancel_orig = _tk.Misc.after_cancel
    def _after_cancel_safe(self, id):
        try:
            return _after_cancel_orig(self, id)
        except RuntimeError as e:
            if 'main thread' not in str(e) and 'main loop' not in str(e):
                raise
    _tk.Misc.after_cancel = _after_cancel_safe
    _destroy_orig = _tk.Misc.destroy
    def _destroy_safe(self):
        try:
            return _destroy_orig(self)
        except RuntimeError as e:
            if 'main thread' in str(e) or 'main loop' in str(e):
                return
            raise
    _tk.Misc.destroy = _destroy_safe
except Exception:
    pass

stop_event = threading.Event()

_vanet_start_orig = None
_mobility_parameters_patched = False

# TABLE II SIMULATION PARAMETERS
UAV_RANGE = 100
MBS_RANGE = 250
VEHICLE_RANGE = 50
UAV_ALTITUDE = 100
UAV_TRAJECTORY_SPEED_MS = 20.0
UAV_TRAJECTORY_RADIUS = 200.0


def _demo_ffmpeg_streaming(net, cars, uavs, rsus, env, agent):
    """Demo Hybrid: agent chọn offload target cho car1, tự chạy ffmpeg/cvlc nếu có."""
    pass  # giữ nguyên như cũ


# ── (các hàm khác giữ nguyên: _patch_vanet_clear_lists_only,
#    _patch_mobility_parameters, _car_ap_distance, _log_assoc_change,
#    update_car_ap_association, start_assoc_daemon,
#    _uav_fixed_trajectory_position) ──────────────────────────────────────────


def _write_training_log(log_path, epoch, steps, total_reward):
    if not log_path:
        return
    try:
        write_header = not os.path.exists(log_path)
        with open(log_path, 'a', encoding='utf-8') as f:
            if write_header:
                f.write("epoch,steps,total_reward,timestamp\n")
            f.write(f"{epoch},{steps},{total_reward:.4f},{datetime.now().isoformat()}\n")
    except Exception:
        pass


def _write_run_log(run_log_path, message):
    if not run_log_path:
        return
    try:
        with open(run_log_path, 'a', encoding='utf-8') as f:
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass


# ── MỚI: ghi delay từng step khi drl_eval ────────────────────────────────────
def _write_delay_csv(path, epoch, step, delay):
    """
    Ghi delay từng step ra CSV.
    Dùng cho drl_eval mode để so sánh với QEA.
    Format: epoch,step,delay
    """
    if not path:
        return
    try:
        write_header = not os.path.exists(path)
        with open(path, 'a', encoding='utf-8') as f:
            if write_header:
                f.write("epoch,step,delay\n")
            f.write(f"{epoch},{step},{delay:.6f}\n")
    except Exception:
        pass


# ── SỬATHAM SỐ: thêm eval_log_path=None ──────────────────────────────────────
def run_simulation_loop(net, config, env, agent, cars, uavs,
                        plot_queue=None, uav_mode='hover',
                        plot_max=1200, log_path=None,
                        run_log_path=None, eval_log_path=None):
    """
    Epoch/step loop. ControlLayer chạy D3QN mỗi bước.
    UAV không do AI điều khiển: hover hoặc fixed_trajectory.

    eval_log_path: nếu truyền vào, ghi delay mỗi step ra CSV
                   (dùng cho algo_mode='drl_eval')
    """
    use_plot = plot_queue is not None
    use_plot_config = getattr(config, 'plot', False)
    center_x = plot_max / 2.0
    center_y = plot_max / 2.0

    def _log(msg):
        if run_log_path:
            _write_run_log(run_log_path, msg.strip())

    control_layer = ControlLayer(env, agent)
    _log("*** Loop start, log: %s\n" % (log_path or ""))

    try:
        for epoch in range(1, config.epochs + 1):
            env.reset()
            done = False
            total_reward = 0
            step = 0

            while not done:
                step += 1
                if step % 20 == 1:
                    try:
                        update_car_ap_association(net)
                    except AssertionError:
                        pass

                if getattr(config, 'uav_mode', 'hover') == 'fixed_trajectory':
                    for i, uav in enumerate(uavs):
                        x, y = _uav_fixed_trajectory_position(
                            step, i, len(uavs), center_x, center_y,
                            UAV_TRAJECTORY_RADIUS,
                            speed_ms=UAV_TRAJECTORY_SPEED_MS, dt=0.1
                        )
                        uav.position = [x, y, 50.0]
                        if hasattr(uav, 'pos'):
                            uav.pos = uav.position

                if use_plot:
                    plot_queue.put(True)

                action_idx, reward = control_layer.step()

                # Agent-driven association với range check
                try:
                    ap_name = control_layer.get_forced_ap_name(action_idx, cars, uavs)
                    forced  = getattr(net, '_car_forced_ap', None)
                    if forced is None:
                        net._car_forced_ap = {}
                        forced = net._car_forced_ap
                    req_car_name = getattr(
                        getattr(control_layer.env, 'requesting_car', None),
                        'name', cars[0].name
                    )
                    if ap_name:
                        forced[req_car_name] = ap_name
                    else:
                        forced.pop(req_car_name, None)
                except Exception:
                    pass

                total_reward += reward

                # ── MỚI: ghi delay mỗi step khi eval mode ────────────────
                if eval_log_path:
                    _write_delay_csv(eval_log_path, epoch, step, -reward)

                if step % 100 == 1 or step >= config.max_steps_per_epoch - 1:
                    decision = control_layer.get_decision(action_idx)
                    _log("  step %d offload→%s bitrate=%s cache=%s R=%.6f" % (
                        step, decision['offload_name'], decision['bitrate_label'],
                        decision['cache'], reward))

                if step >= config.max_steps_per_epoch:
                    done = True
                if done:
                    _log("Epoch %d/%d steps=%d R=%.6f\n" % (
                        epoch, config.epochs, step, total_reward))
                    _write_training_log(log_path, epoch, step, total_reward)
                    agent.save_model()
                    break
                time.sleep(0.05 if use_plot_config else 0.1)

    except KeyboardInterrupt:
        _log("*** Stopped.\n")
    if use_plot:
        plot_queue.put(None)


def run_simulation(config):
    """Thiết lập và chạy mô phỏng."""
    use_plot = getattr(config, 'plot', True)

    # (phần setup Mininet-WiFi, nodes, links giữ nguyên như cũ)
    # ...

    # ── Phần chạy thuật toán ─────────────────────────────────────────────────
    try:
        algo_mode = getattr(config, 'algo_mode', 'drl')
        log_dir   = os.path.abspath(getattr(config, 'log_dir', 'results'))
        os.makedirs(log_dir, exist_ok=True)

        # ── QEA ──────────────────────────────────────────────────────────────
        if algo_mode in ('qea', 'both'):
            info("*** Running QEA baseline (offline optimization)…\n")
            qea = QEAJointCAUA(cars=cars, uavs=uavs, rsus=rsus, config=config)
            X_best, Y_best = qea.optimize()
            info("*** QEA best total cost: %.4f\n" % qea.f_best)

            # ── MỚI: ghi qea_result.csv ──────────────────────────────────
            qea_csv = os.path.join(log_dir, 'qea_result.csv')
            try:
                with open(qea_csv, 'w', encoding='utf-8') as _f:
                    _f.write("generation,f_best\n")
                    for _g, _v in enumerate(qea.convergence, start=1):
                        _f.write(f"{_g},{_v:.6f}\n")
                info("*** QEA CSV saved: %s\n" % qea_csv)
            except Exception as _e:
                info("*** QEA CSV error: %s\n" % _e)

        # ── DRL (train hoặc eval) ─────────────────────────────────────────
        if algo_mode in ('drl', 'both', 'drl_eval'):
            info("*** Running DRL (D3QN) loop…\n")
            stations = list(cars) + list(uavs) + list(rsus)
            env = VanetEnvironment(config, stations, aps=rsus, uavs_list=uavs)

            agent = D3QNAgent(
                state_size=env.state_size,
                action_size=env.action_size,
                num_offload_targets=env.num_offload_targets,
                config=config,
            )

            if algo_mode == 'drl_eval':
                agent.load_model()
                agent.set_eval_mode()

            train_log = os.path.join(log_dir, 'drl_training.csv')
            run_log   = os.path.join(log_dir, 'drl_run.log')

            # ── MỚI: chỉ ghi eval CSV khi drl_eval mode ──────────────────
            eval_log  = os.path.join(log_dir, 'drl_eval.csv') \
                        if algo_mode == 'drl_eval' else None

            run_simulation_loop(
                net, config, env, agent, cars, uavs,
                plot_queue=None,
                uav_mode=getattr(config, 'uav_mode', 'hover'),
                plot_max=getattr(config, 'plot_max', 1200),
                log_path=train_log,
                run_log_path=run_log,
                eval_log_path=eval_log,          # ← MỚI
            )

            if algo_mode == 'drl_eval':
                _demo_ffmpeg_streaming(net, cars, uavs, rsus, env, agent)

    except Exception as e:
        info("*** Error while running algorithms (QEA/DRL): %s\n" % e)

    # (phần CLI, cleanup giữ nguyên như cũ)


def _cleanup_plot_before_exit():
    try:
        plt.close('all')
    except (RuntimeError, KeyboardInterrupt):
        pass
    except Exception:
        pass
    try:
        import matplotlib
        if matplotlib.get_backend().lower().find('tk') >= 0:
            for _ in range(3):
                try:
                    import tkinter as tk
                    root = tk._default_root
                    if root is not None:
                        root.destroy()
                        tk._default_root = None
                    break
                except (RuntimeError, KeyboardInterrupt):
                    pass
                except Exception:
                    pass
    except Exception:
        pass


def _run_exitfuncs_safe():
    try:
        _cleanup_plot_before_exit()
    except BaseException:
        pass
    while atexit._exithandlers:
        handler, args, kwargs = atexit._exithandlers.pop()
        try:
            handler(*args, **kwargs)
        except (RuntimeError, KeyboardInterrupt):
            pass
        except BaseException as e:
            if 'main thread' in str(e) or 'main loop' in str(e):
                pass
            else:
                try:
                    import sys
                    import traceback
                    sys.__excepthook__(type(e), e, e.__traceback__)
                except Exception:
                    pass


atexit._run_exitfuncs = _run_exitfuncs_safe


if __name__ == '__main__':
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('agents/models'):
        os.makedirs('agents/models')
    setLogLevel('info')
    cfg = get_config()
    atexit.register(_cleanup_plot_before_exit)
    run_simulation(cfg)