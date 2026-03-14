#!/usr/bin/env python3
"""
Ryu SDN application: SDN-VANET controller với D3QN (Offload + Caching).

Chạy Ryu: ryu-manager ryu_app.py
Chạy mạng: sudo python3 main_thesis.py

FIXES:
  - Bug 4 FIX: _create_stub_config() thêm num_videos=100 và zipf_exponent=0.7
    → nhất quán với config.py và environment.py
"""
import os
import sys
import threading
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_0

from config import get_config
from environment import VanetEnvironment
from agents.d3qn_agent import D3QNAgent
from control_layer import ControlLayer


def _create_stub_config():
    """
    Config mặc định khi chạy Ryu — đồng bộ hoàn toàn với get_config().

    Dùng get_config() trực tiếp thay vì định nghĩa lại từng tham số,
    tránh hai file bị lệch nhau khi config.py thay đổi.
    """
    return get_config()


def _create_stub_nodes(config):
    """
    Tạo stub nodes với vị trí tam giác đều — khớp với main_thesis.py.
    UAV1, UAV2, UAV3 cách đều nhau trong vùng 400×400m.
    """
    import math
    from types import SimpleNamespace
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
        """
        Vòng lặp control plane — đồng bộ với run_simulation_loop() trong main_thesis.py.

        Cấu trúc giống hệt main_thesis:
          - Có epoch loop (epochs × max_steps_per_epoch)
          - step reset về 0 đầu mỗi epoch
          - In log mỗi 100 bước (khớp main_thesis)
          - In tổng kết cuối mỗi epoch
          - Lưu model sau mỗi epoch
        """
        config   = self._control_layer.env.config
        epochs   = int(getattr(config, 'epochs',               100))
        max_steps = int(getattr(config, 'max_steps_per_epoch', 1000))

        for epoch in range(1, epochs + 1):
            if self._control_stop.is_set():
                break

            self._control_layer.env.reset()
            total_reward = 0.0
            step         = 0

            while step < max_steps and not self._control_stop.is_set():
                try:
                    step        += 1
                    action_idx, reward = self._control_layer.step()
                    total_reward += reward
                    decision     = self._control_layer.get_decision(action_idx)

                    # Khớp main_thesis: in mỗi 100 bước và bước cuối
                    if step % 100 == 1 or step >= max_steps:
                        self.logger.info(
                            "step %d offload→%s bitrate=%s cache=%s R=%.6f",
                            step,
                            decision['offload_name'],
                            decision['bitrate_label'],
                            decision['cache'],
                            reward,
                        )
                except (KeyboardInterrupt, SystemExit):
                    self.logger.info("*** Control loop stopping. ***")
                    self._control_stop.set()
                    break
                except Exception as e:
                    self.logger.exception("Control step error: %s", e)

            # Tổng kết epoch — khớp main_thesis
            self.logger.info(
                "Epoch %d/%d steps=%d R=%.6f",
                epoch, epochs, step, total_reward,
            )
            # Lưu model sau mỗi epoch — khớp main_thesis
            try:
                self._control_layer.agent.save_model()
            except Exception as e:
                self.logger.warning("save_model failed: %s", e)

        self.logger.info("*** Training hoàn thành %d epochs × %d steps = %d steps tổng. ***",
                         epochs, max_steps, epochs * max_steps)

    def stop(self):
        self._control_stop.set()
        super(SdnVanetRyuApp, self).stop()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Khi switch kết nối tới Ryu."""
        datapath = ev.msg.datapath
        self.logger.info("*** Switch connected: datapath_id=%s ***", hex(datapath.id))
