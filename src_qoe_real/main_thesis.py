#!/usr/bin/env python3
"""
Main entry point for the Thesis Simulation (Mininet-WiFi).

Architecture (SDN–VANET–UAV):
  - Control plane : D3QN agent + VanetEnvironment (via Ryu SDN controller)
  - Data plane    : Mininet-WiFi — cars (stations), RSUs/UAVs (APs), switch, controller

algo_mode trong config.py:
  'ryu_train' — Mininet REST server + Ryu trains D3QN, lưu d3qn.pth
  'ryu_env'   — Mininet REST server + Ryu eval (ε=0)
  'qea'       — chạy QEA offline, ghi qea_result.csv
"""
import os
import time
import atexit
import threading
import math
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from types import SimpleNamespace
import matplotlib.pyplot as plt
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.log import setLogLevel, info
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference

from config import get_config
from environment import VanetEnvironment
from agents.qea_joint_ca_ua import QEAJointCAUA
from qoe_runtime import QoESessionStore
from constants import UAV_RANGE, MBS_RANGE, UAV_ALTITUDE
from helpers import (
    estimate_runtime_users_for_uav,
    get_node_xy,
    dist_2d,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Patch tkinter ─────────────────────────────────────────────────────────────
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

# ── Patch Mininet-WiFi mobility: suppress AssertionError khi disconnect ──────
# Xảy ra khi mobility thread cố disconnect xe đang bận (shell.waiting=True)
# D3QN vẫn chạy đúng vì update_car_ap_association() tự quản lý association
try:
    from mn_wifi import mobility as _mn_mobility
    _orig_ap_out_of_range = _mn_mobility.Mobility.ap_out_of_range

    def _safe_ap_out_of_range(self, intf, ap_intf):
        try:
            return _orig_ap_out_of_range(self, intf, ap_intf)
        except Exception:
            pass

    _mn_mobility.Mobility.ap_out_of_range = _safe_ap_out_of_range
except Exception:
    pass

# ── Patch Mininet-WiFi: tắt handover trong thread mobility để tránh BlockingIOError ─
# Thread wifiParameters gọi iwconfig liên tục -> fork nhiều process -> EAGAIN (errno 11).
# Association đã được xử lý bởi start_assoc_daemon() + update_car_ap_association().
try:
    from mn_wifi import mobility as _mn_mob
    _orig_set_handover = getattr(_mn_mob.Mobility, 'set_handover', None)
    if _orig_set_handover is not None:
        def _noop_set_handover(self, intf, aps):
            pass  # skip iwconfig in mobility thread; we use car-ap-assoc daemon
        _mn_mob.Mobility.set_handover = _noop_set_handover
except Exception:
    pass

stop_event                   = threading.Event()
_mobility_parameters_patched = False
_assoc_lock                  = threading.RLock()
_assoc_log_path              = None


# ═══════════════════════════════════════════════════════════════════════════════
# Patch Mininet-WiFi VANET
# ═══════════════════════════════════════════════════════════════════════════════



def _patch_mobility_parameters():
    """Đặt giá trị mặc định cho các tham số mobility có thể bị None."""
    global _mobility_parameters_patched
    if _mobility_parameters_patched:
        return
    try:
        from mn_wifi.mobility import Mobility
        if getattr(Mobility, 'speed', None) is None:
            Mobility.speed = 1.0
        if getattr(Mobility, 'min_x', None) is None:
            Mobility.min_x = 0
        if getattr(Mobility, 'min_y', None) is None:
            Mobility.min_y = 0
        _mobility_parameters_patched = True
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers: khoảng cách, log, association
# ═══════════════════════════════════════════════════════════════════════════════

def _car_ap_distance(car, ap):
    """Coverage distance chuẩn hóa theo mặt phẳng 2D cho toàn pipeline."""
    try:
        return dist_2d(car, ap)
    except Exception:
        return float('inf')


def _log_assoc_change(msg):
    try:
        path = _assoc_log_path
        if not path:
            path = os.path.abspath(os.path.join(THIS_DIR, 'results', 'association.log'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write('%s %s\n' % (datetime.now().strftime('%H:%M:%S'), msg))
    except Exception:
        pass


def _set_assoc_log_path(config):
    """Align association telemetry file with configured log_dir."""
    global _assoc_log_path
    raw = getattr(config, 'log_dir', 'results')
    if os.path.isabs(raw):
        log_dir = raw
    else:
        log_dir = os.path.join(THIS_DIR, raw)
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    _assoc_log_path = os.path.join(log_dir, 'association.log')


def _is_shell_alive(node):
    """Kiểm tra shell process của Mininet node còn sống không."""
    try:
        shell = getattr(node, 'shell', None)
        if shell is None:
            return False
        # Nếu process đã chết, poll() trả exit code (!= None)
        if hasattr(shell, 'poll') and shell.poll() is not None:
            return False
        # Kiểm tra waiting flag — nếu đang waiting thì không gửi cmd
        if getattr(node, 'waiting', False):
            return False
        return True
    except Exception:
        return False


def _assoc_name_from_node(node):
    assoc = None
    params = getattr(node, 'params', None)
    if isinstance(params, dict):
        assoc = params.get('associatedTo', None)
    if assoc is None:
        assoc = getattr(node, 'associatedTo', None)
    if isinstance(assoc, (list, tuple)):
        assoc = assoc[0] if assoc else None
    if isinstance(assoc, str):
        return assoc
    return getattr(assoc, 'name', None)


def _snapshot_nodes(nodes):
    """Snapshot tối thiểu của node list để objective tĩnh trong QEA optimize."""
    snaps = []
    for n in nodes:
        x, y = get_node_xy(n)
        params = {'position': (float(x), float(y))}
        assoc_name = _assoc_name_from_node(n)
        if assoc_name:
            params['associatedTo'] = assoc_name
        snaps.append(SimpleNamespace(name=getattr(n, 'name', ''), params=params))
    return snaps


def update_car_ap_association(net):
    """Car–UAV association theo khoảng cách (RSU không phục vụ trực tiếp car)."""
    with _assoc_lock:
        aps_for_assoc = [
            (idx, ap)
            for idx, ap in enumerate(getattr(net, 'aps', []), start=1)
            if getattr(ap, 'name', '').lower().startswith('uav')
        ]
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
            test           = 0
            chosen_ap_idx  = None
            chosen_ssid    = None
            chosen_ap_name = None
            forced_name    = forced.get(car.name)

            if forced_name:
                for idx, ap in aps_for_assoc:
                    if getattr(ap, 'name', '') == forced_name:
                        try:
                            if _car_ap_distance(car, ap) <= float(UAV_RANGE):
                                test           = 1
                                chosen_ap_idx  = idx
                                chosen_ssid    = ap.params.get('ssid', 'AP%d' % chosen_ap_idx)
                                chosen_ap_name = getattr(ap, 'name', None)
                        except Exception:
                            pass
                        break

            if not test:
                best = None
                for idx, ap in aps_for_assoc:
                    try:
                        d = _car_ap_distance(car, ap)
                        if d <= float(UAV_RANGE) and (best is None or d < best[0]):
                            best = (d, idx, ap)
                    except Exception:
                        pass
                if best is not None:
                    _, chosen_ap_idx, chosen_ap = best
                    test = 1
                    chosen_ssid = chosen_ap.params.get('ssid', 'AP%d' % chosen_ap_idx)
                    chosen_ap_name = getattr(chosen_ap, 'name', None)

            if test == 1:
                # Cập nhật metadata ngay để runtime load đọc được trong cùng step.
                try:
                    if isinstance(getattr(car, 'params', None), dict):
                        car.params['associatedTo'] = chosen_ap_name
                except Exception:
                    pass

                same_ap = (car_ap.get(car.name) == chosen_ap_idx
                           and car_mode.get(car.name) == 'ap')
                if same_ap:
                    continue
                car_mode[car.name] = 'ap'
                car_ap[car.name]   = chosen_ap_idx
                _log_assoc_change('%s -> AP (192.168.%s.1%02d)' % (
                    car.name, chosen_ap_idx, net.cars.index(car) + 1))
                if _is_shell_alive(car):
                    try:
                        car.cmd('iw dev %s-wlan0 connect %s 2>/dev/null' % (car.name, chosen_ssid))
                        car.cmd('pkill -9 -f "dhclient %s-wlan0" 2>/dev/null' % car.name)
                        car.cmd('dhclient %s-wlan0 2>/dev/null &' % car.name)
                        car.cmd('ip route add 10.10.0.0/16 via 10.10.0.%s 2>/dev/null'
                                % (net.cars.index(car) + 1))
                        car.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward 2>/dev/null')
                    except (Exception, BaseException):
                        car_mode[car.name] = ''
            else:
                # Không có AP phù hợp: clear metadata để helper không đếm tải ảo.
                try:
                    if isinstance(getattr(car, 'params', None), dict):
                        car.params['associatedTo'] = None
                except Exception:
                    pass

                if car_mode.get(car.name) != 'adhoc':
                    car_mode[car.name] = 'adhoc'
                    _log_assoc_change('%s -> ad-hoc 10.10.0.%s' % (
                        car.name, net.cars.index(car) + 1))
                    if _is_shell_alive(car):
                        try:
                            car.cmd('iw dev %s-wlan0 ibss join adhoc 2412 2>/dev/null' % car.name)
                            car.cmd('ip addr add 10.10.0.%s/16 dev %s-wlan0 2>/dev/null' % (
                                net.cars.index(car) + 1, car.name))
                        except (Exception, BaseException):
                            pass


def start_assoc_daemon(net, interval=1.0):
    err_count = 0

    def _loop():
        nonlocal err_count
        while not stop_event.is_set():
            try:
                update_car_ap_association(net)
            except Exception as e:
                err_count += 1
                if err_count == 1 or err_count % 50 == 0:
                    info("*** Assoc daemon warning (%d): %s\n" % (err_count, e))
            time.sleep(interval)
    t = threading.Thread(target=_loop, daemon=True, name='car-ap-assoc')
    t.start()
    return t

# ═══════════════════════════════════════════════════════════════════════════════
# REST Environment Server (for Ryu training/deploy)
# ═══════════════════════════════════════════════════════════════════════════════

def run_rest_env_server(net, config, env, cars, uavs, host="127.0.0.1", port=8081):
    """
    Expose VanetEnvironment over REST.

    Core endpoints:
      - POST /reset         -> {state}
      - POST /step          -> {next_state, reward, done, info, decision, requesting_car}
      - GET  /meta          -> metadata for Ryu agent + QoE telemetry contract

    QoE telemetry endpoints (player -> environment bridge):
      - POST /qoe/session/start
      - POST /qoe/segment
    """
    lock = threading.Lock()
    qoe_store = QoESessionStore(config)
    env.attach_qoe_store(qoe_store)
    env.set_qoe_session(getattr(config, 'qoe_default_session_id', 'default'))

    def _json_response(handler, code, payload):
        raw = json.dumps(payload).encode("utf-8")
        handler.send_response(code)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(raw)))
        handler.end_headers()
        try:
            handler.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/meta":
                with lock:
                    ap_port_map = {}
                    ap_subnet_idx = {}
                    for idx, ap in enumerate(getattr(net, 'aps', []), start=1):
                        name = getattr(ap, 'name', '')
                        if not name:
                            continue
                        ap_port_map[name] = idx
                        ap_subnet_idx[name] = idx
                    payload = {
                        "state_size": int(env.state_size),
                        "action_size": int(env.action_size),
                        "num_offload_targets": int(env.num_offload_targets),
                        "ap_port_map": ap_port_map,
                        "ap_subnet_idx": ap_subnet_idx,
                        "objective_profile": str(getattr(env, 'objective_profile', 'qoe_real')),
                        "qoe_contract": {
                            "default_session_id": str(getattr(env, 'qoe_session_id', 'default')),
                            "focus_car_name": str(getattr(env, 'qoe_focus_car_name', '') or ''),
                            "rl_step_interval_s": float(getattr(config, 'rl_step_interval_s', 0.10)),
                            "segment_index_source": "environment-per-car",
                            "required_segment_event_fields": [
                                "session_id", "car_name", "segment_idx",
                                "startup_sec", "rebuffer_sec", "rebuffer_count",
                                "quality_index", "switch_magnitude",
                                "segment_download_sec", "buffer_sec",
                            ],
                        },
                    }
                return _json_response(self, 200, payload)
            return _json_response(self, 404, {"error": "not_found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(n) if n > 0 else b"{}"
            try:
                req = json.loads(body.decode("utf-8") or "{}")
            except Exception:
                req = {}

            path = self.path.rstrip("/")

            if path == "/qoe/session/start":
                sid = qoe_store.start_session(
                    session_id=req.get("session_id"),
                    car_name=req.get("car_name"),
                    meta=req.get("meta", {}),
                )
                env.set_qoe_session(sid)
                return _json_response(self, 200, {"ok": True, "session_id": sid})

            if path == "/qoe/segment":
                ev = dict(req or {})
                if "car_name" not in ev and "car" in ev:
                    ev["car_name"] = ev.get("car")
                if "session_id" not in ev:
                    ev["session_id"] = getattr(env, "qoe_session_id", "default")
                metric = qoe_store.ingest_segment_event(ev)
                return _json_response(self, 200, {"ok": True, "metric": metric})

            if path == "/reset":
                with lock:
                    state = env.reset()
                    payload = {"state": state.tolist()}
                return _json_response(self, 200, payload)

            if path == "/step":
                action_idx = int(req.get("action_idx", 0))
                qoe_session_id = req.get("qoe_session_id")
                if qoe_session_id:
                    env.set_qoe_session(qoe_session_id)

                with lock:
                    cur_car = getattr(getattr(env, "requesting_car", None), "name", None)
                    with _assoc_lock:
                        next_state, reward, done, step_info = env.step(action_idx)
                    step_info = step_info or {}

                    try:
                        decision = dict(step_info.get("decision", {}) or {})
                        if not decision:
                            decision = env.get_action_components(action_idx)

                        tier = str(decision.get("tier", "uav"))
                        uav_idx = int(decision.get("uav_idx", 0))

                        ap_name = ""
                        if tier == "uav" and 0 <= uav_idx < len(uavs):
                            ap_name = uavs[uav_idx].name

                        with _assoc_lock:
                            forced = getattr(net, "_car_forced_ap", None)
                            if forced is None:
                                net._car_forced_ap = {}
                                forced = net._car_forced_ap
                            if cur_car and ap_name:
                                forced[cur_car] = ap_name
                            elif cur_car:
                                forced.pop(cur_car, None)
                                ap_name = ""
                    except Exception as e:
                        info("*** REST /step decision warning: %s\n" % e)
                        decision = {}
                        ap_name = ""

                    payload = {
                        "requesting_car": cur_car,
                        "next_state": next_state.tolist(),
                        "reward": float(reward),
                        "done": bool(done),
                        "info": step_info,
                        "decision": decision,
                        "ap_name": ap_name,
                    }
                return _json_response(self, 200, payload)

            return _json_response(self, 404, {"error": "not_found"})

    httpd = ThreadingHTTPServer((host, int(port)), _Handler)
    info(f"*** REST env server listening on http://{host}:{port}\\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Setup Mininet-WiFi
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(config):
    """Thiết lập Mininet-WiFi và chạy thuật toán theo algo_mode."""
    # Hỗ trợ chạy nhiều lần trong cùng process: reset stop flag trước khi start daemon.
    stop_event.clear()
    use_plot  = getattr(config, 'plot', True)
    num_roads = min(getattr(config, 'roads', 8), 8)
    algo_mode = getattr(config, 'algo_mode', 'ryu_env')
    _set_assoc_log_path(config)

    net = Mininet_wifi(
        controller=RemoteController,
        roads=num_roads,
        link=wmediumd,
        wmediumd_mode=interference,
    )

    info("*** Creating nodes\n")
    cars      = []
    speed_ms  = getattr(config, 'vehicle_speed_kmh', 20) / 3.6
    car_range = getattr(config, 'vehicle_range_m', 50)
    for i in range(1, config.cars + 1):
        min_ = max(1, int(speed_ms - 3))
        max_ = int(speed_ms + 3)
        cars.append(net.addCar(
            f'car{i}', wlans=2,
            min_speed=min_, max_speed=max_, range=car_range
        ))

    plot_max_val = getattr(config, 'plot_max', 400)
    channels     = ['1', '6', '11']
    rsus = [
        net.addAccessPoint(
            f'rsu{i}', ssid=f'RSU{10+i}', mode='g',
            channel=channels[(i - 1) % 3], range=MBS_RANGE
        )
        for i in range(1, config.rsus + 1)
    ]

    cx, cy = plot_max_val / 2.0, plot_max_val / 2.0
    r_poly = plot_max_val / 4.0
    uav_pos_list = []
    for i in range(config.uavs):
        angle = 2 * math.pi * i / max(config.uavs, 1) + math.pi / 2
        uav_pos_list.append((cx + r_poly * math.cos(angle), cy + r_poly * math.sin(angle), UAV_ALTITUDE))
    uavs = [
        net.addAccessPoint(
            f'uav{i}', ssid=f'UAV{i}', mode='g',
            channel='5', range=UAV_RANGE
        )
        for i in range(1, config.uavs + 1)
    ]

    s1 = net.addSwitch('s1', cls=OVSKernelSwitch)
    c1 = net.addController('c1', controller=RemoteController,
                            ip='127.0.0.1', port=6653)

    net.setPropagationModel(model="logDistance", exp=4)
    net.configureWifiNodes()

    info("*** Creating links\n")
    for rsu in rsus:
        net.addLink(rsu, s1)
    for uav in uavs:
        net.addLink(uav, s1)

    plot_max      = getattr(config, 'plot_max', 400)
    mobility_time = getattr(config, 'mobility_time', 1)
    if use_plot:
        net.plotGraph(max_x=plot_max, max_y=plot_max)
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

    # Set vị trí UAV theo layout đã tính sẵn để đồng bộ model/plot/runtime.
    for idx, uav in enumerate(uavs):
        try:
            if idx < len(uav_pos_list):
                x, y, z = uav_pos_list[idx]
            else:
                x, y = get_node_xy(uav)
                z = float(UAV_ALTITUDE)
            # Mininet-WiFi requires tuples for internal position logic
            uav.position = (float(x), float(y), float(z))
            if hasattr(uav, 'pos'):
                uav.pos = uav.position
            if isinstance(getattr(uav, 'params', None), dict):
                uav.params['position'] = uav.position
            if hasattr(uav, 'set_pos_wmediumd'):
                uav.set_pos_wmediumd((float(x), float(y), float(z)))
        except Exception:
            pass

    # Set IP cho APs
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
    if algo_mode in ('ryu_env', 'ryu_train'):
        update_car_ap_association(net)

        try:
            if net.cars and aps_order:
                out = net.cars[0].cmd('ping -c 1 -W 2 192.168.1.1 2>&1')
                if '1 received' in out or '1 packets received' in out:
                    info("*** Ping car1 -> 192.168.1.1 OK ***\n")
                else:
                    info("*** Ping car1 -> 192.168.1.1 FAIL ***\n")
        except Exception:
            pass

        info("*** Car–AP association done. ***\n")
        start_assoc_daemon(net, interval=0.5)
    else:
        info("*** QEA mode: association daemon disabled (deterministic eval sync). ***\n")

    # ── Chạy thuật toán ───────────────────────────────────────────────────────
    try:
        raw_log_dir = getattr(config, 'log_dir', 'results')
        if os.path.isabs(raw_log_dir):
            log_dir = raw_log_dir
        else:
            log_dir = os.path.join(THIS_DIR, raw_log_dir)
        log_dir = os.path.abspath(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        if algo_mode == 'qea':
            info("*** Running QEA baseline...\n")
            # Dùng snapshot tĩnh để objective QEA optimize không bị drift do mobility runtime.
            _cars_snap = _snapshot_nodes(cars)
            _uavs_snap = _snapshot_nodes(uavs)
            _rsus_snap = _snapshot_nodes(rsus)
            qea = QEAJointCAUA(
                cars=_cars_snap, uavs=_uavs_snap, rsus=_rsus_snap, config=config,
                F=getattr(config, 'num_videos', 100),
                Z=4,
                t_max=getattr(config, 'qea_generations', 100),
            )
            qea.optimize()
            info("*** QEA best calibrated QoE cost: %.4f\n" % qea.f_best)

            qea_csv = os.path.join(log_dir, 'qea_result.csv')
            try:
                with open(qea_csv, 'w', encoding='utf-8') as _f:
                    _f.write("generation,f_best\n")
                    for _g, _v in enumerate(qea.convergence, start=1):
                        _f.write(f"{_g},{_v:.6f}\n")
                info("*** QEA CSV saved: %s\n" % qea_csv)
            except Exception as _e:
                info("*** QEA CSV error: %s\n" % _e)

            # ── QEA eval: chạy QEA solution qua random requests ──────────
            info("*** Running QEA eval (per-request calibrated QoE)...\n")
            import random as _rnd
            import numpy as _np
            from models import calculate_total_cost as _calc_cost

            _joint_probs = _np.asarray(qea.p_fz, dtype=_np.float64)
            if _joint_probs.size == 0 or float(_joint_probs.sum()) <= 0.0:
                _joint_probs = _np.full(
                    (max(int(qea.F), 1), max(int(qea.Z), 1)),
                    1.0,
                    dtype=_np.float64,
                )
            _joint_probs /= float(_joint_probs.sum())
            _joint_flat = _joint_probs.reshape(-1)
            _qea_eval_csv = os.path.join(log_dir, 'qea_eval.csv')
            _qea_eval_meta_csv = os.path.join(log_dir, 'qea_eval_meta.csv')
            _play_t = max(float(getattr(config, 'chunk_playback_sec', 2.0)), 1e-6)
            _w_delay = float(getattr(config, 'qoe_w_delay', 1.0))
            _w_stall = float(getattr(config, 'qoe_w_stall', 1.5))
            _w_smooth = float(getattr(config, 'qoe_w_smooth', 0.5))
            _reward_min = float(getattr(config, 'reward_clip_min', -25.0))
            _reward_max = float(getattr(config, 'reward_clip_max', 0.0))
            _oor_alpha = float(getattr(config, 'oor_penalty_alpha', 10.0))
            _oor_beta = float(getattr(config, 'oor_penalty_beta', 1.25))
            _oor_cap = float(getattr(config, 'oor_penalty_cap', 5.0))
            _prev_delay_by_car = {}

            def _sync_qea_eval_metadata():
                """Deterministic sync: metadata association cho toàn bộ cars theo X_best."""
                x_best = getattr(qea, 'X_best', None)
                if x_best is None:
                    for car_k in cars:
                        if isinstance(getattr(car_k, 'params', None), dict):
                            car_k.params['associatedTo'] = None
                        else:
                            setattr(car_k, 'associatedTo', None)
                    return
                if x_best.shape[0] <= 0 or x_best.shape[1] <= 0:
                    for car_k in cars:
                        if isinstance(getattr(car_k, 'params', None), dict):
                            car_k.params['associatedTo'] = None
                        else:
                            setattr(car_k, 'associatedTo', None)
                    return

                for k, car_k in enumerate(cars):
                    assoc_name = None
                    if k < x_best.shape[1]:
                        assigned_row = int(_np.argmax(x_best[:, k]))
                        if 0 <= assigned_row < len(uavs):
                            target_k = uavs[assigned_row]
                            if _car_ap_distance(car_k, target_k) <= float(UAV_RANGE):
                                assoc_name = getattr(target_k, 'name', None)
                    if isinstance(getattr(car_k, 'params', None), dict):
                        car_k.params['associatedTo'] = assoc_name
                    else:
                        setattr(car_k, 'associatedTo', assoc_name)

            try:
                with open(_qea_eval_csv, 'w', encoding='utf-8') as _ef, \
                     open(_qea_eval_meta_csv, 'w', encoding='utf-8') as _mf:
                    _ef.write("epoch,step,delay,delay_term,stall_norm,smooth_norm,qoe_cost,qoe_mos,reward_final\n")
                    _mf.write(
                        "epoch,step,uav_idx,f_req,z_req,out_of_range,delay,delay_term,stall_norm,smooth_norm,"
                        "qoe_cost,qoe_mos,reward_final\n"
                    )
                    _total_delay = 0.0
                    _total_steps = 0
                    _out_count = 0
                    _total_qoe_cost = 0.0
                    for _ep in range(1, config.epochs + 1):
                        for _st in range(1, config.max_steps_per_epoch + 1):
                            _sync_qea_eval_metadata()

                            _valid_cars = []
                            for _car_k in cars:
                                _covered_uav = any(_car_ap_distance(_car_k, _u) <= float(UAV_RANGE) for _u in uavs)
                                if _covered_uav:
                                    _valid_cars.append(_car_k)
                            _car_pool = _valid_cars if _valid_cars else cars

                            _car = _rnd.choice(_car_pool)
                            _car_idx = cars.index(_car)
                            _req_idx = int(_np.random.choice(_joint_flat.size, p=_joint_flat))
                            _f_req = int(_req_idx // qea.Z)
                            _z_req = int(_req_idx % qea.Z)

                            _uav_l = int(_np.argmax(qea.X_best[:, _car_idx]))
                            _out_of_range = False
                            _users_rt = 0

                            _target = uavs[_uav_l]
                            _distance = float(_car_ap_distance(_car, _target))
                            _out_of_range = (_distance > float(UAV_RANGE))
                            # Paper: sharing factor depends on number of users assigned to UAV l.
                            # In QEA, this is given by the optimized association matrix X_best.
                            _users_rt = int(_np.sum(qea.X_best[_uav_l, :]))

                            _f_mod = _f_req % qea.F
                            if qea.Y_best[_uav_l, _f_mod, _z_req] == 1:
                                _cm, _zr, _zc = 1, _z_req, _z_req
                            else:
                                _z_plus = None
                                for _z2 in range(_z_req + 1, qea.Z):
                                    if qea.Y_best[_uav_l, _f_mod, _z2] == 1:
                                        _z_plus = _z2
                                        break
                                if _z_plus is not None:
                                    _cm, _zr, _zc = 2, _z_req, _z_plus
                                else:
                                    _cm, _zr, _zc = 0, _z_req, _z_req

                            if _out_of_range:
                                _out_count += 1

                            # --- PHYSICS-BASED DELAY (SAME AS D3QN) ---
                            _delay = _calc_cost(
                                _car, _target, config,
                                cache_mode=_cm, all_uavs=uavs,
                                z_req=_zr, z_cached=_zc,
                                num_uavs=len(uavs),
                                rsus=rsus,
                                num_users_per_uav=_users_rt,
                            )
                            # -----------------------------------------
                            _car_name = getattr(_car, 'name', f'car{_car_idx+1}')
                            _prev = float(_prev_delay_by_car.get(_car_name, _delay))
                            _delay_term = math.log1p(max(float(_delay), 0.0))
                            _stall_norm = max(0.0, (float(_delay) - _play_t) / _play_t)
                            _smooth_norm = abs(float(_delay) - _prev) / _play_t
                            _proxy_cost = (
                                _w_delay * _delay_term
                                + _w_stall * _stall_norm
                                + _w_smooth * _smooth_norm
                            )
                            _qoe_cost = qea._apply_qoe_calibration(_proxy_cost, float(_delay), _smooth_norm)
                            _qoe_mos = max(1.0, min(5.0, 5.0 - _qoe_cost))

                            _overshoot = max(0.0, (_distance / float(UAV_RANGE)) - 1.0)
                            _oor_penalty = min(_oor_alpha * (_overshoot ** _oor_beta), _oor_cap)
                            _reward_unclipped = -_qoe_cost - _oor_penalty
                            _reward_final = max(_reward_min, min(_reward_max, _reward_unclipped))
                            _prev_delay_by_car[_car_name] = float(_delay)

                            _ef.write(
                                f"{_ep},{_st},{_delay:.6f},{_delay_term:.6f},{_stall_norm:.6f},{_smooth_norm:.6f},"
                                f"{_qoe_cost:.6f},{_qoe_mos:.6f},{_reward_final:.6f}\n"
                            )
                            _mf.write(
                                f"{_ep},{_st},{_uav_l},{_f_req},{_z_req},{int(_out_of_range)},{_delay:.6f},"
                                f"{_delay_term:.6f},{_stall_norm:.6f},{_smooth_norm:.6f},"
                                f"{_qoe_cost:.6f},{_qoe_mos:.6f},{_reward_final:.6f}\n"
                            )
                            _total_delay += _delay
                            _total_steps += 1
                            _total_qoe_cost += _qoe_cost

                    _avg = _total_delay / max(_total_steps, 1)
                    _oor_rate = float(_out_count) / max(_total_steps, 1)
                    _avg_qoe = _total_qoe_cost / max(_total_steps, 1)
                    info("*** QEA eval: %d steps, avg delay = %.4fs\n" % (_total_steps, _avg))
                    info("*** QEA eval avg calibrated QoE cost = %.4f\n" % _avg_qoe)
                    info("*** QEA eval out_of_range: %d (%.2f%%)\n" % (_out_count, 100.0 * _oor_rate))
                    info("*** QEA eval CSV saved: %s\n" % _qea_eval_csv)
                    info("*** QEA eval META saved: %s\n" % _qea_eval_meta_csv)
            except Exception as _e:
                info("*** QEA eval error: %s\n" % _e)

        if algo_mode in ('ryu_env', 'ryu_train'):
            info("*** Starting Mininet REST Environment Server...\n")
            stations = list(cars) + list(uavs) + list(rsus)
            env = VanetEnvironment(config, stations, aps=rsus, uavs_list=uavs)
            host = getattr(config, 'rest_host', '127.0.0.1')
            port = getattr(config, 'rest_port', 8081)
            run_rest_env_server(net, config, env, cars, uavs, host=host, port=port)

    except Exception as e:
        info("*** Error while running algorithms: %s\n" % e)

    # ── CLI và cleanup ────────────────────────────────────────────────────────
    if algo_mode == 'qea':
        info("*** QEA mode finished. Skip interactive CLI.\n")
    else:
        try:
            CLI(net)
            while True:
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


# ═══════════════════════════════════════════════════════════════════════════════
# atexit cleanup
# ═══════════════════════════════════════════════════════════════════════════════

def _cleanup_plot_before_exit():
    try:
        plt.close('all')
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
                    import sys, traceback
                    sys.__excepthook__(type(e), e, e.__traceback__)
                except Exception:
                    pass


atexit._run_exitfuncs = _run_exitfuncs_safe


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    os.makedirs('results',       exist_ok=True)
    os.makedirs('agents/models', exist_ok=True)
    setLogLevel('info')
    cfg = get_config()
    atexit.register(_cleanup_plot_before_exit)
    run_simulation(cfg)
