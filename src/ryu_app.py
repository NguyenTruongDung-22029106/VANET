#!/usr/bin/env python3
"""
Ryu SDN application: SDN-VANET controller với D3QN (Offload + Caching).

Chạy Ryu: ryu-manager ryu_app.py
Chạy mạng: sudo python3 main_thesis.py
"""
import threading
import time
from types import SimpleNamespace

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_0

from environment import VanetEnvironment
from agents.d3qn_agent import D3QNAgent
# FIX: dùng ControlLayer từ file chung, không định nghĩa lại
from agents.control_layer import ControlLayer


def _create_stub_config():
    """Config mặc định khi chạy Ryu — đồng bộ với get_config()."""
    return SimpleNamespace(
        cars=10, uavs=3, rsus=1,
        plot_max=400,
        epochs=100, max_steps_per_epoch=1000,
        model_path='agents/models/d3qn.pth',
        # Communication params (khớp config.py)
        B=160e6, Bh=60e6, M=30,
        PUAV_dBm=30, PBS_dBm=35, sigma2_dBm=-95,
        H=100.0, fc=5e9, d0=1.0,
        nLoS=2.0, nNLoS=2.4, sLoS=5.3, sNLoS=5.27,
        kappa=11.9, zeta=0.13,
        gamma_bs=3.5, eta_bs=100.0,
        w0=1.0, C_comp=3.4e9, chunk_size_MB=8.0,
    )


def _create_stub_nodes(config):
    """
    Tạo stub nodes với vị trí tam giác đều — khớp với main_thesis.py.
    UAV1, UAV2, UAV3 cách đều nhau trong vùng 400×400m.
    """
    import math
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
        SimpleNamespace(name=f'car{i}', params={'position': (cx + (i - config.cars // 2) * 30, cy - 50)})
        for i in range(1, config.cars + 1)
    ]
    rsus = [
        SimpleNamespace(name=f'rsu{i}', params={'position': (50 + (i - 1) * 150, 50)})
        for i in range(1, config.rsus + 1)
    ]
    uavs = [
        SimpleNamespace(name=f'uav{i}', params={'position': uav_verts[(i - 1) % 3]})
        for i in range(1, config.uavs + 1)
    ]
    stations = cars + uavs
    return stations, rsus, uavs


class SdnVanetRyuApp(app_manager.RyuApp):
    """Ryu app: SDN controller cho VANET, chạy D3QN (offload + cache)."""

    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SdnVanetRyuApp, self).__init__(*args, **kwargs)
        self._control_thread = None
        self._control_stop = threading.Event()
        self._control_layer = None

    def start(self):
        super(SdnVanetRyuApp, self).start()
        self.logger.info("*** SDN-VANET Ryu app started (D3QN). ***")

        config = _create_stub_config()
        stations, rsus, uavs = _create_stub_nodes(config)
        env = VanetEnvironment(config, stations, aps=rsus, uavs_list=uavs)

        # FIX: truyền đủ 4 tham số — num_offload_targets bắt buộc
        agent = D3QNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            num_offload_targets=env.num_offload_targets,
            config=config,
        )
        agent.load_model()

        # FIX: dùng ControlLayer từ agents/control_layer.py
        self._control_layer = ControlLayer(env, agent)
        self._control_stop.clear()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        self.logger.info("*** Control Layer (D3QN) loop started. ***")

    def _control_loop(self):
        """Vòng lặp control plane: mỗi bước gọi control_layer.step() và log ra Ryu."""
        step = 0
        while not self._control_stop.is_set():
            try:
                step += 1
                action_idx, reward = self._control_layer.step()
                # get_decision() trả dict (khớp environment.get_action_components)
                decision = self._control_layer.get_decision(action_idx)
                if step % 50 == 1:
                    self.logger.info(
                        "step %d offload → %s bitrate = %s cache = %s R= %.2f",
                        step,
                        decision['offload_name'],
                        decision['bitrate_label'],
                        decision['cache'],
                        reward,
                    )
            except (KeyboardInterrupt, SystemExit):
                # Ctrl+C hoặc ryu-manager dừng → thoát sạch, không in traceback
                self.logger.info("*** Control loop stopping (KeyboardInterrupt). ***")
                self._control_stop.set()
                break
            except Exception as e:
                self.logger.exception("Control step error: %s", e)
            time.sleep(0.5)

    def stop(self):
        self._control_stop.set()
        super(SdnVanetRyuApp, self).stop()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Khi switch kết nối tới Ryu."""
        datapath = ev.msg.datapath
        self.logger.info("*** Switch connected: datapath_id=%s ***", hex(datapath.id))