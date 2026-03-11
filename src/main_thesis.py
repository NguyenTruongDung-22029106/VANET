#!/usr/bin/env python3
"""
Main entry point for the Thesis Simulation (Mininet-WiFi Graph version).

Architecture (SDN–VANET–UAV):
  - Control plane (logic): DRL agent (D3QN) + VanetEnvironment; agent observes state,
    selects action (offload + cache), env computes reward (cost/welfare).
  - Data plane: Mininet-WiFi net = cars (stations), RSUs (APs), UAVs (aircrafts), switch s1,
    controller c1.
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
from agents.control_layer import ControlLayer          # FIX: import từ file chung
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
    if not cars:
        return
    car = cars[0]
    state = env.get_state()
    action_idx = agent.select_action(state)
    off_idx, z_req, cache_01 = agent.get_action_vector(action_idx)

    # off_idx: 0=Local, 1..len(uavs)=UAV, len(uavs)+1..=RSU
    target = None
    if off_idx == 0:
        target = car
    elif off_idx <= len(uavs):
        target = uavs[off_idx - 1]
    else:
        rsu_idx = off_idx - 1 - len(uavs)
        if 0 <= rsu_idx < len(rsus):
            target = rsus[rsu_idx]

    if target is None:
        return

    try:
        car_ip = car.IP()
    except Exception:
        car_ip = "10.0.0.1"
    target_name = getattr(target, "name", "node")
    cache_flag = "YES" if cache_01 == 1 else "NO"
    bitrate_label = ['low(480p)', 'high(1080p)'][z_req]

    info("\n*** Hybrid demo: Agent chọn offload → %s bitrate=%s cache=%s cho car1 (IP %s)\n"
         % (target_name, bitrate_label, cache_flag, car_ip))

    if target is not car and hasattr(target, "cmd"):
        ffmpeg_cmd = (
            "ffmpeg -y -i /videos/input_1080p.mp4 "
            "-vf scale=640:360 -b:v 800k -preset ultrafast /videos/output_360p.mp4"
        )
        cvlc_cmd = (
            f'cvlc /videos/output_360p.mp4 --sout "#rtp{{dst={car_ip},port=5004,mux=ts}}" '
            "--no-sout-all --sout-keep &"
        )
        info(f"*** Chạy ffmpeg trên {target_name}:\n    {target_name} {ffmpeg_cmd}\n")
        target.cmd(ffmpeg_cmd)
        info(f"*** Chạy cvlc trên {target_name} → stream về car1 ({car_ip}):\n    {target_name} {cvlc_cmd}\n")
        target.cmd(cvlc_cmd)
    else:
        info("*** Node đích là car hoặc không hỗ trợ cmd(); bỏ qua ffmpeg/cvlc.\n")

    info("*** Trên car1, tự chạy: car1 vlc rtp://@:5004 &\n\n")


def _patch_mobility_parameters():
    global _mobility_parameters_patched
    from mn_wifi.mobility import Mobility
    from time import sleep
    if _mobility_parameters_patched:
        return

    def _patched_parameters(self):
        mob_nodes = list(set(self.mobileNodes) - set(self.aps))
        while getattr(self.thread_, '_keep_alive', False):
            try:
                self.config_links(mob_nodes)
            except AssertionError:
                pass
            sleep(0.0001)

    Mobility.parameters = _patched_parameters
    _mobility_parameters_patched = True


def _patch_vanet_clear_lists_only():
    global _vanet_start_orig
    import mn_wifi.vanet as vanet_module
    from mn_wifi.mobility import Mobility
    if _vanet_start_orig is not None:
        return
    _vanet_start_orig = vanet_module.vanet.start

    def _patched_start(self, cars, **kwargs):
        aps = kwargs['aps']
        roads = kwargs.get('roads', 8)
        Mobility.stations = cars
        Mobility.mobileNodes = cars
        Mobility.aps = aps
        self.roads.clear()
        self.points.clear()
        self.all_points = []
        self.interX = {}
        self.interY = {}
        num_points = max(2 + roads + len(aps), 20)
        while len(self.points) < num_points:
            self.points.append(len(self.points))
        return _vanet_start_orig(self, cars, **kwargs)

    vanet_module.vanet.start = _patched_start


def _car_ap_distance(car, ap):
    if hasattr(car, 'get_distance_to'):
        return car.get_distance_to(ap)
    try:
        pc = getattr(car, 'position', None) or (car.params.get('position') if hasattr(car, 'params') else None) or getattr(car, 'pos', None)
        pa = getattr(ap, 'position', None) or (ap.params.get('position') if hasattr(ap, 'params') else None) or getattr(ap, 'pos', None)
        if pc is not None and pa is not None:
            return math.sqrt(sum((float(pc[i]) - float(pa[i])) ** 2 for i in range(min(2, len(pc), len(pa)))))
    except Exception:
        pass
    return float('inf')


def _log_assoc_change(msg):
    try:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__) or '.', '..', 'results', 'association.log'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write('%s %s\n' % (datetime.now().strftime('%H:%M:%S'), msg))
    except Exception:
        pass


def update_car_ap_association(net):
    """Car–AP association by distance. RSU/MBS dùng MBS_RANGE, UAV dùng UAV_RANGE; ngoài vùng phủ → ad-hoc."""
    aps_for_assoc = list(net.aps)
    car_mode = getattr(net, '_car_mode', None)
    if car_mode is None:
        net._car_mode = {}
        car_mode = net._car_mode
    car_ap = getattr(net, '_car_ap', None)
    if car_ap is None:
        net._car_ap = {}
        car_ap = net._car_ap
    forced = getattr(net, '_car_forced_ap', None)
    if forced is None:
        net._car_forced_ap = {}
        forced = net._car_forced_ap

    for car in net.cars:
        test = 0
        chosen_ap_idx = None
        chosen_ssid = None
        forced_name = forced.get(car.name)
        if forced_name:
            for ap in aps_for_assoc:
                if getattr(ap, 'name', '') == forced_name:
                    try:
                        test = 1
                        chosen_ap_idx = aps_for_assoc.index(ap) + 1
                        chosen_ssid = ap.params.get('ssid', 'AP%d' % chosen_ap_idx)
                    except Exception:
                        pass
                    break
        if not test:
            for ap in aps_for_assoc:
                try:
                    name_l = getattr(ap, 'name', '').lower()
                    this_range = MBS_RANGE if ('rsu' in name_l or 'mbs' in name_l) else UAV_RANGE
                    if _car_ap_distance(car, ap) <= this_range:
                        test = 1
                        chosen_ap_idx = aps_for_assoc.index(ap) + 1
                        chosen_ssid = ap.params.get('ssid', 'AP%d' % chosen_ap_idx)
                        break
                except Exception:
                    pass

        if test == 1:
            was_adhoc = car_mode.get(car.name) == 'adhoc'
            same_ap = (car_ap.get(car.name) == chosen_ap_idx) and (car_mode.get(car.name) == 'ap')
            if same_ap and not was_adhoc:
                continue
            car_mode[car.name] = 'ap'
            car_ap[car.name] = chosen_ap_idx
            ap_idx = chosen_ap_idx
            ssid = chosen_ssid
            car_idx = net.cars.index(car) + 1
            try:
                car.cmd('ip addr flush dev %s-wlan0 2>/dev/null' % car.name)
                car.cmd('iw dev %s-wlan0 set type managed 2>/dev/null' % car.name)
                car.cmd('iw dev %s-wlan0 connect %s 2>/dev/null' % (car.name, ssid))
                time.sleep(0.15)
                ip_host = 100 + car_idx
                car.setIP('192.168.%s.%s/24' % (ap_idx, ip_host), intf='%s-wlan0' % car.name)
                car.cmd('ip route replace default via 192.168.%s.1 dev %s-wlan0 2>/dev/null' % (ap_idx, car.name))
                car.cmd('ip route replace 192.168.0.0/16 via 192.168.%s.1 dev %s-wlan0 2>/dev/null' % (ap_idx, car.name))
                car.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward 2>/dev/null')
                if was_adhoc or not same_ap:
                    _log_assoc_change('%s -> AP (192.168.%s.%s)' % (car.name, ap_idx, ip_host))
            except Exception:
                car_ap[car.name] = None
            continue

        if car_mode.get(car.name) == 'adhoc':
            continue
        car_mode[car.name] = 'adhoc'
        car_ap[car.name] = None
        try:
            _log_assoc_change('%s -> ad-hoc 10.10.0.%s' % (car.name, net.cars.index(car) + 1))
            car.cmd('ip addr flush dev %s-wlan0 2>/dev/null' % car.name)
            car.cmd('iwconfig %s-wlan0 mode ad-hoc essid MyAdHocNetwork channel 1 2>/dev/null' % car.name)
            car.cmd('ip link set %s-wlan0 up 2>/dev/null' % car.name)
            time.sleep(0.1)
            car.setIP('10.10.0.%s/24' % (net.cars.index(car) + 1), intf='%s-wlan0' % car.name)
            car.cmd('ip route add 192.168.0.0/16 via 10.10.0.%s 2>/dev/null' % (net.cars.index(car) + 1))
            car.cmd('ip route add 10.10.0.0/16 via 10.10.0.%s 2>/dev/null' % (net.cars.index(car) + 1))
            car.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward 2>/dev/null')
        except Exception:
            car_mode[car.name] = ''


def start_assoc_daemon(net, interval=1.0):
    def _loop():
        while not stop_event.is_set():
            try:
                update_car_ap_association(net)
            except Exception:
                pass
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name='car-ap-assoc')
    t.start()
    return t


def _uav_fixed_trajectory_position(step, uav_idx, num_uavs, center_x, center_y, radius, speed_ms=20.0, dt=0.1):
    dist_per_step = speed_ms * dt
    angle_per_step = dist_per_step / radius
    angle = step * angle_per_step + (uav_idx * 2 * math.pi / max(num_uavs, 1))
    x = center_x + radius * math.cos(angle)
    y = center_y + radius * math.sin(angle)
    return float(x), float(y)


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


def run_simulation_loop(net, config, env, agent, cars, uavs, plot_queue=None, uav_mode='hover', plot_max=1200, log_path=None, run_log_path=None):
    """
    Epoch/step loop. ControlLayer chạy D3QN mỗi bước.
    UAV không do AI điều khiển: hover hoặc fixed_trajectory.
    """
    use_plot = plot_queue is not None
    use_plot_config = getattr(config, 'plot', False)
    center_x = plot_max / 2.0
    center_y = plot_max / 2.0

    def _log(msg):
        if run_log_path:
            _write_run_log(run_log_path, msg.strip())

    # FIX: dùng ControlLayer từ agents/control_layer.py (không còn duplicate)
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
                            step, i, len(uavs), center_x, center_y, UAV_TRAJECTORY_RADIUS,
                            speed_ms=UAV_TRAJECTORY_SPEED_MS, dt=0.1
                        )
                        uav.position = [x, y, 50.0]
                        if hasattr(uav, 'pos'):
                            uav.pos = uav.position

                if use_plot:
                    plot_queue.put(True)

                action_idx, reward = control_layer.step()

                # FIX 4: Agent-driven association với range check
                # Chỉ ép car1 bám AP khi xe thực sự trong vùng phủ của node đó
                try:
                    ap_name = control_layer.get_forced_ap_name(action_idx, cars, uavs)
                    forced  = getattr(net, '_car_forced_ap', None)
                    if forced is None:
                        net._car_forced_ap = {}
                        forced = net._car_forced_ap
                    if ap_name:
                        forced[cars[0].name] = ap_name
                    else:
                        # Xóa forced để update_car_ap_association fallback theo khoảng cách
                        forced.pop(cars[0].name, None)
                except Exception:
                    pass

                total_reward += reward

                if step % 100 == 1 or step >= config.max_steps_per_epoch - 1:
                    decision = control_layer.get_decision(action_idx)
                    _log("  step %d offload→%s bitrate=%s cache=%s R=%.6f" % (
                        step, decision['offload_name'], decision['bitrate_label'],
                        decision['cache'], reward))

                if step >= config.max_steps_per_epoch:
                    done = True
                if done:
                    _log("Epoch %d/%d steps=%d R=%.6f\n" % (epoch, config.epochs, step, total_reward))
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
    use_plot = True
    num_roads = min(getattr(config, 'roads', 8), 8)
    net = Mininet_wifi(controller=RemoteController, roads=num_roads, link=wmediumd, wmediumd_mode=interference)

    info("*** Creating nodes\n")
    cars = []
    speed_ms = getattr(config, 'vehicle_speed_kmh', 120) / 3.6
    car_range = getattr(config, 'vehicle_range_m', 50)
    for i in range(1, config.cars + 1):
        min_ = max(1, int(speed_ms - 3))
        max_ = int(speed_ms + 3)
        cars.append(net.addCar(f'car{i}', wlans=2, min_speed=min_, max_speed=max_, range=car_range))

    plot_max_val = getattr(config, 'plot_max', 1200)
    channels = ['1', '6', '11']
    rsus = [
        net.addAccessPoint(
            f'rsu{i}', ssid=f'RSU{10+i}', mode='g',
            channel=channels[(i - 1) % 3], range=MBS_RANGE,
            position=f'{100+((i-1)%3)*400},{100+((i-1)//3)*400},0'
        )
        for i in range(1, config.rsus + 1)
    ]
    cx, cy = plot_max_val / 2.0, plot_max_val / 2.0
    r_tri = plot_max_val / 4.0
    cos30 = math.cos(math.pi / 6)
    sin30 = math.sin(math.pi / 6)
    uav_triangle_verts = [
        (cx, cy + r_tri, UAV_ALTITUDE),
        (cx - r_tri * cos30, cy - r_tri * sin30, UAV_ALTITUDE),
        (cx + r_tri * cos30, cy - r_tri * sin30, UAV_ALTITUDE),
    ]
    uav_pos_list = [uav_triangle_verts[i % 3] for i in range(config.uavs)]
    uavs = [
        net.addAccessPoint(
            f'uav{i}', ssid=f'UAV{i}', mode='g',
            channel='5', range=UAV_RANGE,
            position=f'{int(uav_pos_list[i-1][0])},{int(uav_pos_list[i-1][1])},{UAV_ALTITUDE}'
        )
        for i in range(1, config.uavs + 1)
    ]
    s1 = net.addSwitch('s1', cls=OVSKernelSwitch)
    c1 = net.addController('c1', controller=RemoteController, ip='127.0.0.1', port=6653)
    info("*** Using Ryu SDN controller (127.0.0.1:6653).\n")

    net.setPropagationModel(model="logDistance", exp=4)
    net.configureWifiNodes()

    info("*** Associating and Creating links\n")
    for rsu in rsus:
        net.addLink(rsu, s1)
    for uav in uavs:
        net.addLink(uav, s1)

    plot_max = getattr(config, 'plot_max', 1200)
    mobility_time = getattr(config, 'mobility_time', 1)
    if use_plot:
        net.plotGraph(max_x=plot_max, max_y=plot_max)
        _patch_vanet_clear_lists_only()
    net.startMobility(time=mobility_time)
    _patch_mobility_parameters()
    info("*** Starting network build\n")
    net.build()
    c1.start()
    for rsu in rsus:
        rsu.start([c1])
    for uav in uavs:
        uav.start([c1])
    s1.start([c1])

    for uav in uavs:
        try:
            pos = getattr(uav, 'position', None) or (uav.params.get('position') if hasattr(uav, 'params') else None) or [0, 0, 0]
            x, y = float(pos[0]), float(pos[1])
            uav.position = [x, y, float(UAV_ALTITUDE)]
            if hasattr(uav, 'pos'):
                uav.pos = uav.position
            if hasattr(uav, 'set_pos_wmediumd'):
                uav.set_pos_wmediumd((x, y, UAV_ALTITUDE))
        except Exception:
            pass

    aps_order = list(rsus) + list(uavs)
    for ap_idx, ap in enumerate(aps_order, start=1):
        try:
            ap.setIP('192.168.%s.1/24' % ap_idx, intf='%s-wlan0' % ap.name)
        except Exception:
            try:
                ap.setIP('192.168.%s.1/24' % ap_idx, intf='%s-wlan1' % ap.name)
            except Exception:
                pass
        try:
            ap.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward 2>/dev/null')
        except Exception:
            pass

    time.sleep(1.5)
    update_car_ap_association(net)

    try:
        if net.cars and aps_order:
            out = net.cars[0].cmd('ping -c 1 -W 2 192.168.1.1 2>&1')
            if '1 received' in out or '1 packets received' in out:
                info("*** Ping car1 -> 192.168.1.1 OK ***\n")
            else:
                info("*** Ping car1 -> 192.168.1.1 FAIL (xe có thể ngoài vùng phủ) ***\n")
    except Exception:
        pass
    info("*** Car–AP association done. ***\n")
    start_assoc_daemon(net, interval=0.5)

    try:
        algo_mode = getattr(config, 'algo_mode', 'drl')
        log_dir = os.path.abspath(getattr(config, 'log_dir', 'results'))
        os.makedirs(log_dir, exist_ok=True)

        if algo_mode in ('qea', 'both'):
            info("*** Running QEA baseline (offline optimization)…\n")
            qea = QEAJointCAUA(cars=cars, uavs=uavs, rsus=rsus, config=config)
            X_best, Y_best = qea.optimize()
            info("*** QEA best total cost: %.4f\n" % qea.f_best)

        if algo_mode in ('drl', 'both', 'drl_eval'):
            info("*** Running DRL (D3QN) loop…\n")
            stations = list(cars) + list(uavs) + list(rsus)
            env = VanetEnvironment(config, stations, aps=rsus, uavs_list=uavs)

            # FIX: truyền đủ 4 tham số — num_offload_targets bắt buộc
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
            run_log = os.path.join(log_dir, 'drl_run.log')
            run_simulation_loop(
                net, config, env, agent, cars, uavs,
                plot_queue=None,
                uav_mode=getattr(config, 'uav_mode', 'hover'),
                plot_max=getattr(config, 'plot_max', 1200),
                log_path=train_log,
                run_log_path=run_log,
            )

            if algo_mode == 'drl_eval':
                _demo_ffmpeg_streaming(net, cars, uavs, rsus, env, agent)

    except Exception as e:
        info("*** Error while running algorithms (QEA/DRL): %s\n" % e)

    try:
        CLI(net)
        while True:
            update_car_ap_association(net)
            try:
                a = int(input("nhap dau vao: "))
            except (ValueError, EOFError, KeyboardInterrupt):
                break
            if a == 1:
                CLI(net)
            else:
                break
    except KeyboardInterrupt:
        info("*** Ctrl+C\n")

    info("*** Stopping network\n")
    try:
        stop_event.set()
        if use_plot:
            net.stop_graph_params()
            time.sleep(0.5)
            try:
                plt.close('all')
            except Exception:
                pass
        net.stop()
    except KeyboardInterrupt:
        info("*** Interrupted during cleanup.\n")
    except Exception as e:
        if 'main thread is not in main loop' not in str(e):
            info("*** Error during stop: %s\n" % e)


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