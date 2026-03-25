#!/usr/bin/env python3
"""
Ryu SDN controller for VANET-UAV with D3QN-driven flow control.

Architecture (SDN southbound):
  - OVS switch s1 connects all APs (rsu1, uav1-3) via OpenFlow 1.3
  - D3QN agent (eval mode) makes offload decisions every step
  - Controller translates per-car decisions into proactive flow rules on s1

Port mapping on switch s1:
  - lấy động từ REST /meta của main_thesis.py (khớp thứ tự AP trong topology runtime)

Per-car flow strategy:
  - PacketIn learns car MAC addresses → mac_to_car mapping
  - D3QN decision: car_name → target AP
  - Controller installs: match eth_src=car_mac → output target_ap_port
                         match eth_dst=car_mac → output car's_ingress_port
  - Cookie = car_index → delete old rules before installing new ones
  - idle_timeout=0: controller manages lifecycle, no flickering

Workflow:
  1. Train:   algo_mode='ryu_train' → sudo python3 main_thesis.py &
              ryu-manager ryu_app.py           → d3qn.pth
  2. Eval:    algo_mode='ryu_env'   → sudo python3 main_thesis.py &
              ryu-manager ryu_app.py           (epsilon=0)
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import sys
import time
import threading
import json
import urllib.request
import urllib.error
from datetime import datetime
import torch

# Keep PyTorch single-threaded inside controller process
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# Ryu dùng eventlet monkey-patch threading → green thread.
# PyTorch C extensions KHÔNG tương thích với green thread → SIGSEGV.
# Dùng eventlet.tpool.execute() để chạy PyTorch trong real native thread.
try:
    import eventlet.tpool as _tpool
except ImportError:
    # Fallback: chạy trực tiếp nếu không có eventlet
    class _tpool:
        @staticmethod
        def execute(fn, *a, **kw):
            return fn(*a, **kw)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4

from config import get_config
from agents.d3qn_agent import D3QNAgent
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Default topology constants (fallback nếu /meta không cung cấp map)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_AP_PORT_MAP = {
    'rsu1': 1,
    'uav1': 2,
    'uav2': 3,
    'uav3': 4,
}

DEFAULT_AP_SUBNET_IDX = {
    'rsu1': 1,
    'uav1': 2,
    'uav2': 3,
    'uav3': 4,
}

PROACTIVE_PRIORITY  = 200
L2_PRIORITY         = 100
ARP_PRIORITY        = 50
TABLE_MISS_PRIORITY = 0

L2_IDLE_TIMEOUT     = 60
L2_HARD_TIMEOUT     = 300

COOKIE_BASE         = 0x00D3000000000000


def _car_cookie(car_name):
    """car1 → cookie 1, car2 → cookie 2, ..."""
    try:
        return COOKIE_BASE | int(car_name.replace('car', ''))
    except (ValueError, AttributeError):
        return COOKIE_BASE


class SdnVanetRyuApp(app_manager.RyuApp):
    """
    Ryu SDN controller: L2 learning switch + D3QN per-car proactive flow control.

    Per-car flow lifecycle:
      1. PacketIn learns car MAC (mac_to_car)
      2. D3QN decides car → target AP
      3. Delete old flows for this car (by cookie)
      4. Install new per-car flows:
         - eth_src=car_mac, IP → output target_ap_port  (uplink)
         - eth_dst=car_mac, IP → output car_ingress_port (downlink)
      5. idle_timeout=0 — only controller deletes, no flickering
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._control_thread = None
        self._control_stop   = threading.Event()
        self._agent          = None
        self._config         = None
        self._rest_base      = None

        self.datapaths      = {}      # dpid → datapath
        self.mac_to_port    = {}      # dpid → {mac → port}
        self.mac_to_car     = {}      # mac → car_name  (learned from PacketIn IP)
        self.car_to_mac     = {}      # car_name → mac
        self._offload_table = {}      # car_name → ap_name
        self._offload_lock  = threading.Lock()

        self._log_dir  = os.path.join(THIS_DIR, 'results')
        self._log_path = os.path.join(self._log_dir, 'ryu_deploy.log')
        self._csv_path = os.path.join(self._log_dir, 'ryu_deploy.csv')
        self.ap_port_map = dict(DEFAULT_AP_PORT_MAP)
        self.ap_subnet_idx = dict(DEFAULT_AP_SUBNET_IDX)

    # ══════════════════════════════════════════════════════════════════════
    # Logging
    # ══════════════════════════════════════════════════════════════════════

    def _configure_output_paths(self, algo_mode):
        suffix = 'training' if algo_mode == 'ryu_train' else 'eval'
        self._log_path = os.path.join(self._log_dir, f'ryu_deploy_{suffix}.log')
        self._csv_path = os.path.join(self._log_dir, f'ryu_deploy_{suffix}.csv')

    def _init_csv(self):
        os.makedirs(self._log_dir, exist_ok=True)
        with open(self._csv_path, 'w') as f:
            f.write('timestamp,step,car,offload_target,tier,out_of_range,fallback,disconnected,bitrate,z_cached,cache,delay\\n')
        with open(self._log_path, 'w') as f:
            ts = datetime.now().isoformat()
            f.write(f'[{ts}] RUN_START csv={os.path.basename(self._csv_path)}\n')

    def _write_csv(self, step, car, target, tier, out_of_range, fallback, disconnected, bitrate, z_cached, cache, delay):
        try:
            with open(self._csv_path, 'a') as f:
                ts = datetime.now().isoformat()
                f.write(f'{ts},{step},{car},{target},{tier},{out_of_range},{fallback},{disconnected},{bitrate},{z_cached},{cache},{delay:.6f}\n')
        except Exception:
            pass

    def _write_log(self, msg):
        try:
            os.makedirs(self._log_dir, exist_ok=True)
            with open(self._log_path, 'a') as f:
                ts = datetime.now().strftime('%H:%M:%S')
                f.write(f'[{ts}] {msg}\n')
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # Start / Stop
    # ══════════════════════════════════════════════════════════════════════

    def start(self):
        super().start()
        self.logger.info("*** SDN-VANET Ryu app started (REST env). ***")

        self._config = get_config()
        cfg_log_dir = getattr(self._config, 'log_dir', 'results')
        if os.path.isabs(cfg_log_dir):
            self._log_dir = cfg_log_dir
        else:
            self._log_dir = os.path.join(THIS_DIR, cfg_log_dir)
        host = getattr(self._config, 'rest_host', '127.0.0.1')
        port = int(getattr(self._config, 'rest_port', 8081))
        self._rest_base = f"http://{host}:{port}"

        meta = self._rest_get_with_retry("/meta", retries=20, sleep_s=0.5)
        if meta is None:
            msg = f"REST env not ready at {self._rest_base}/meta. Stop controller startup."
            self.logger.error(msg)
            self._write_log(msg)
            return
        state_size = int(meta.get("state_size", 15))
        action_size = int(meta.get("action_size", 6))
        num_targets = int(meta.get("num_offload_targets", 3))
        meta_port_map = meta.get("ap_port_map", {}) or {}
        meta_subnet_idx = meta.get("ap_subnet_idx", {}) or {}
        if isinstance(meta_port_map, dict) and meta_port_map:
            try:
                self.ap_port_map = {
                    str(k): int(v) for k, v in meta_port_map.items()
                }
            except Exception:
                self.ap_port_map = dict(DEFAULT_AP_PORT_MAP)
        if isinstance(meta_subnet_idx, dict) and meta_subnet_idx:
            try:
                self.ap_subnet_idx = {
                    str(k): int(v) for k, v in meta_subnet_idx.items()
                }
            except Exception:
                self.ap_subnet_idx = dict(DEFAULT_AP_SUBNET_IDX)

        agent = D3QNAgent(
            state_size=state_size,
            action_size=action_size,
            num_offload_targets=num_targets,
            config=self._config,
        )

        algo_mode = getattr(self._config, 'algo_mode', 'ryu_train')
        self._configure_output_paths(algo_mode)
        self._init_csv()

        model_path = getattr(self._config, 'model_path', 'agents/models/d3qn.pth')
        if not os.path.isabs(model_path):
            model_path = os.path.join(THIS_DIR, model_path)
        agent.model_path = model_path

        if algo_mode == 'ryu_train':
            self.logger.info("*** Ryu mode: TRAIN (epsilon>0). ***")
        else:
            if not os.path.exists(model_path):
                msg = f"Model not found at '{model_path}'. Stop ryu_env to avoid random-weight evaluation."
                self.logger.error(msg)
                self._write_log(msg)
                return
            if not agent.load_model():
                msg = f"Failed to load model from '{model_path}'. Stop ryu_env to avoid invalid evaluation."
                self.logger.error(msg)
                self._write_log(msg)
                return
            agent.set_eval_mode()
            self.logger.info("*** Ryu mode: EVAL (epsilon=0). ***")

        self._agent = agent
        self._control_stop.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop, daemon=True,
        )
        self._control_thread.start()
        self.logger.info("*** D3QN control loop started. ***")

    def stop(self):
        self._control_stop.set()
        super().stop()

    # ══════════════════════════════════════════════════════════════════════
    # REST helpers
    # ══════════════════════════════════════════════════════════════════════

    def _rest_get(self, path, timeout=30.0):
        url = self._rest_base + path
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _rest_get_with_retry(self, path, retries=10, sleep_s=0.5, timeout=5.0):
        last_err = None
        for _ in range(max(int(retries), 1)):
            try:
                return self._rest_get(path, timeout=timeout)
            except Exception as e:
                last_err = e
                time.sleep(float(sleep_s))
        self.logger.error("REST GET %s failed after retries: %s", path, last_err)
        return None

    def _rest_post(self, path, payload, timeout=30.0):
        url = self._rest_base + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _rest_post_with_retry(self, path, payload, retries=10, sleep_s=0.5, timeout=5.0):
        last_err = None
        for _ in range(max(int(retries), 1)):
            try:
                return self._rest_post(path, payload, timeout=timeout)
            except Exception as e:
                last_err = e
                time.sleep(float(sleep_s))
        self.logger.error("REST POST %s failed after retries: %s", path, last_err)
        return None

    # ══════════════════════════════════════════════════════════════════════
    # Flow rule helpers
    # ══════════════════════════════════════════════════════════════════════

    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0, cookie=0):
        parser = datapath.ofproto_parser
        inst   = [parser.OFPInstructionActions(
            datapath.ofproto.OFPIT_APPLY_ACTIONS, actions,
        )]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            cookie=cookie,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        datapath.send_msg(mod)

    def _del_flows_by_cookie(self, datapath, cookie):
        """Delete ALL flows matching this cookie on the given switch."""
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            cookie=cookie,
            cookie_mask=0xFFFFFFFFFFFFFFFF,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=parser.OFPMatch(),
        )
        datapath.send_msg(mod)

    def _push_car_flows(self, car_name, ap_name):
        """
        Per-car proactive flow rules:
          1. Delete old flows for this car (by cookie)
          2. If car MAC is known:
               - Uplink:   eth_src=car_mac, IP → output target_ap_port
               - Downlink: eth_dst=car_mac, IP → output car_ingress_port
          3. If MAC unknown: fallback to subnet-based forwarding
        """
        out_port = self.ap_port_map.get(ap_name)
        if out_port is None:
            return

        cookie    = _car_cookie(car_name)
        car_mac   = self.car_to_mac.get(car_name)
        has_mac   = car_mac is not None

        for dpid, datapath in list(self.datapaths.items()):
            # Chỉ push lên switch chính s1 (dpid=1) — AP bridges không cần flow rules
            if dpid != 1:
                continue
            if not getattr(datapath, 'is_active', True):
                continue
            parser  = datapath.ofproto_parser

            # Step 1: delete old flows for this car
            self._del_flows_by_cookie(datapath, cookie)

            if has_mac:
                # Step 2a: per-car uplink (car → target AP)
                match_up = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    eth_src=car_mac,
                )
                actions_up = [parser.OFPActionOutput(out_port)]
                self._add_flow(
                    datapath, PROACTIVE_PRIORITY, match_up, actions_up,
                    cookie=cookie,
                )

                # Step 2b: per-car downlink (target AP → car)
                ingress = self.mac_to_port.get(dpid, {}).get(car_mac)
                if ingress is not None:
                    match_down = parser.OFPMatch(
                        eth_type=ether_types.ETH_TYPE_IP,
                        eth_dst=car_mac,
                    )
                    actions_down = [parser.OFPActionOutput(ingress)]
                    self._add_flow(
                        datapath, PROACTIVE_PRIORITY, match_down, actions_down,
                        cookie=cookie,
                    )

                self._write_log(
                    f'FLOW_PER_CAR: {car_name}(mac={car_mac})→{ap_name} '
                    f'port={out_port} dpid={hex(dpid)}'
                )
            else:
                # Step 3: fallback — subnet-based (until MAC is learned)
                subnet_idx = self.ap_subnet_idx.get(ap_name)
                if subnet_idx is None:
                    continue
                match_fwd = parser.OFPMatch(
                    eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_dst=(f'192.168.{subnet_idx}.0', '255.255.255.0'),
                )
                actions_fwd = [parser.OFPActionOutput(out_port)]
                self._add_flow(
                    datapath, PROACTIVE_PRIORITY - 10, match_fwd, actions_fwd,
                    cookie=cookie,
                )

                # Return rules for other subnets
                for other_ap, other_port in self.ap_port_map.items():
                    if other_ap == ap_name:
                        continue
                    other_idx = self.ap_subnet_idx.get(other_ap)
                    if other_idx is None:
                        continue
                    match_ret = parser.OFPMatch(
                        eth_type=ether_types.ETH_TYPE_IP,
                        in_port=out_port,
                        ipv4_dst=(f'192.168.{other_idx}.0', '255.255.255.0'),
                    )
                    actions_ret = [parser.OFPActionOutput(other_port)]
                    self._add_flow(
                        datapath, PROACTIVE_PRIORITY - 11, match_ret, actions_ret,
                        cookie=cookie,
                    )

                self._write_log(
                    f'FLOW_SUBNET_FALLBACK: {car_name}→{ap_name} '
                    f'port={out_port} dpid={hex(dpid)} (MAC not yet learned)'
                )

    def _upgrade_car_flows(self, car_name, car_mac):
        """Khi MAC mới được learn, upgrade flows từ subnet-based → per-car."""
        with self._offload_lock:
            ap_name = self._offload_table.get(car_name)
        if ap_name:
            self._push_car_flows(car_name, ap_name)
            self._write_log(
                f'FLOW_UPGRADE: {car_name}(mac={car_mac}) upgraded to per-car rules'
            )

    # ══════════════════════════════════════════════════════════════════════
    # MAC → car name mapping (learned from IP src in PacketIn)
    # ══════════════════════════════════════════════════════════════════════

    def _try_learn_car_mac(self, pkt, src_mac, in_port):
        """
        Identify car from IP src address.
        IP format: 192.168.{ap_idx}.1{car_idx:02d}
        Example: 192.168.1.101 = car1 on rsu1 (ap_idx=1)
        """
        ip_hdr = pkt.get_protocol(ipv4.ipv4)
        if ip_hdr is None:
            return

        src_ip = ip_hdr.src
        parts  = src_ip.split('.')
        if len(parts) != 4 or parts[0] != '192' or parts[1] != '168':
            return

        try:
            host = int(parts[3])
        except ValueError:
            return

        if 101 <= host <= 199:
            car_idx  = host - 100
            car_name = f'car{car_idx}'

            already_known = car_name in self.car_to_mac
            self.mac_to_car[src_mac] = car_name
            self.car_to_mac[car_name] = src_mac

            if not already_known:
                self.logger.info(
                    "*** Learned: %s = MAC %s (from IP %s, port %d) ***",
                    car_name, src_mac, src_ip, in_port,
                )
                self._upgrade_car_flows(car_name, src_mac)

    # ══════════════════════════════════════════════════════════════════════
    # D3QN inference loop
    # ══════════════════════════════════════════════════════════════════════

    def _control_loop(self):
        """
        Mỗi step (REST env):
          1. state (từ /reset hoặc response /step trước) → agent action
          2. POST /step → nhận next_state, reward, raw_delay, requesting_car, ap_name
          3. (optional) push per-car flows nếu biết MAC
          4. If training: store experience + train + save checkpoint định kỳ
        """
        agent = self._agent
        step  = 0
        training = (getattr(self._config, 'algo_mode', 'ryu_train') == 'ryu_train')
        if training:
            epochs = max(int(getattr(self._config, 'epochs', 1)), 1)
            max_steps_per_epoch = max(int(getattr(self._config, 'max_steps_per_epoch', 1)), 1)
            max_steps = epochs * max_steps_per_epoch
        else:
            max_steps = int(getattr(
                self._config, 'eval_steps',
                getattr(self._config, 'max_steps_per_epoch', 1000),
            ))
            if max_steps <= 0:
                max_steps = None

        reset = self._rest_post_with_retry("/reset", {}, retries=20, sleep_s=0.5)
        if reset is None:
            self._write_log("REST reset failed after retries")
            return
        state = np.array(reset.get("state", []), dtype=np.float32)

        self._write_log('REST control loop started')
        if max_steps is None:
            self.logger.info("*** REST control loop running... (unbounded) ***")
        else:
            self.logger.info("*** REST control loop running... max_steps=%d ***", max_steps)

        while not self._control_stop.is_set() and (max_steps is None or step < max_steps):
            try:
                step += 1
                action_idx = int(_tpool.execute(agent.select_action, state))
                resp = self._rest_post("/step", {"action_idx": action_idx})

                next_state = np.array(resp.get("next_state", []), dtype=np.float32)
                reward = float(resp.get("reward", 0.0))
                info = resp.get("info", {}) or {}
                delay = float(info.get("raw_delay", -reward))
                car_name = resp.get("requesting_car", "") or "car1"
                ap_name = resp.get("ap_name", "") or ""
                decision = resp.get("decision", {}) or {}
                tier = decision.get("tier", "")
                out_of_range = bool(info.get("out_of_range", False))
                fallback = bool(info.get("fallback", False))
                disconnected = bool(info.get("disconnected", False))

                if training:
                    _tpool.execute(agent.store_experience, state, action_idx, reward, next_state, False)
                    _tpool.execute(agent.train)
                    if step % 1000 == 0:
                        _tpool.execute(agent.save_model)

                with self._offload_lock:
                    prev = self._offload_table.get(car_name)

                if ap_name in self.ap_port_map:
                    if prev != ap_name:
                        with self._offload_lock:
                            self._offload_table[car_name] = ap_name
                        try:
                            self._push_car_flows(car_name, ap_name)
                        except Exception as e:
                            self.logger.warning(
                                "Failed to push per-car flows for %s->%s: %s",
                                car_name, ap_name, e,
                            )
                elif not ap_name and prev is not None:
                    with self._offload_lock:
                        self._offload_table.pop(car_name, None)
                    cookie = _car_cookie(car_name)
                    for dpid, datapath in list(self.datapaths.items()):
                        if dpid == 1:
                            self._del_flows_by_cookie(datapath, cookie)
                    self._write_log(f'FLOW_REMOVED: {car_name} (selected tier unavailable in current coverage)')

                self._write_csv(
                    step, car_name, ap_name,
                    tier, out_of_range, fallback, disconnected,
                    decision.get("z_req", ""),
                    decision.get("z_cached", ""),
                    decision.get("cache", ""),
                    delay,
                )

                if step % 100 == 1:
                    tier = decision.get("tier", "")
                    out_of_range = bool(info.get("out_of_range", False))
                    self.logger.info(
                        "step %d  %s->%s  tier=%s  cache=%s  z_cached=%s  out_of_range=%s  delay=%.4fs",
                        step, car_name, ap_name, tier, decision.get("cache", ""), decision.get("z_cached", ""), out_of_range, delay,
                    )
                    with self._offload_lock:
                        tbl = dict(self._offload_table)
                    self.logger.info("  offload_table: %s", tbl)
                    self.logger.info("  known MACs: %s", dict(self.car_to_mac))

                state = next_state
                time.sleep(0.10)  # 100ms — giảm áp lực lên OVS

            except (KeyboardInterrupt, SystemExit):
                self._write_log('REST loop stopped (interrupt)')
                break
            except Exception as e:
                self.logger.warning("Control loop skip step %d due to error: %s", step, e)

        if training:
            try:
                _tpool.execute(agent.save_model)
            except Exception as e:
                self.logger.warning("Failed to save model at loop end: %s", e)
        self._write_log(f'REST loop ended after {step} steps')
        self.logger.info("*** Control loop stopped after %d steps. ***", step)

    # ══════════════════════════════════════════════════════════════════════
    # OpenFlow event handlers
    # ══════════════════════════════════════════════════════════════════════

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Switch kết nối → cài default flows:
          1. Table-miss → controller (PacketIn)
          2. ARP → flood
          3. Replay current offload table
        """
        datapath = ev.msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        dpid     = datapath.id

        self.datapaths[dpid] = datapath
        self.mac_to_port.setdefault(dpid, {})

        self._add_flow(
            datapath, TABLE_MISS_PRIORITY,
            parser.OFPMatch(),
            [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                    ofproto.OFPCML_NO_BUFFER)],
        )

        self._add_flow(
            datapath, ARP_PRIORITY,
            parser.OFPMatch(eth_type=ether_types.ETH_TYPE_ARP),
            [parser.OFPActionOutput(ofproto.OFPP_FLOOD)],
        )

        self.logger.info(
            "*** Switch dpid=%s connected — table-miss + ARP flood installed ***",
            hex(dpid),
        )
        self._write_log(f'Switch connected: dpid={hex(dpid)}')

        with self._offload_lock:
            # Chỉ replay offload table lên switch chính s1
            if dpid == 1:
                for car, ap in self._offload_table.items():
                    try:
                        self._push_car_flows(car, ap)
                    except Exception as e:
                        self.logger.warning(
                            "Replay flow push failed for %s->%s on %s: %s",
                            car, ap, hex(dpid), e,
                        )

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        PacketIn:
          1. MAC learning (src → in_port)
          2. Try identify car from IP src → learn MAC-to-car mapping
          3. Unicast forward or flood
          4. Install L2 flow for known unicast
        """
        msg      = ev.msg
        datapath = msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        dpid     = datapath.id
        in_port  = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            return

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        src_mac = eth.src
        dst_mac = eth.dst

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port

        self._try_learn_car_mac(pkt, src_mac, in_port)

        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
                in_port=in_port,
                eth_dst=dst_mac,
                eth_src=src_mac,
            )
            self._add_flow(
                datapath, L2_PRIORITY, match, actions,
                idle_timeout=L2_IDLE_TIMEOUT,
                hard_timeout=L2_HARD_TIMEOUT,
            )

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None,
        )
        datapath.send_msg(out)
