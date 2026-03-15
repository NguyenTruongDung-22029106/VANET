#!/usr/bin/env python3
"""
Ryu SDN application: SDN-VANET controller với D3QN (Deploy mode).

Workflow:
  Bước 1 — Train (chạy riêng):
    algo_mode = 'drl'
    sudo python3 main_thesis.py   → lưu agents/models/d3qn.pth

  Bước 2 — Deploy (file này):
    sudo python3 main_thesis.py &   # khởi động Mininet
    ryu-manager ryu_app.py          # load .pth → điều khiển

Thay đổi so với bản cũ:
  - BỎ: _create_stub_nodes(), _create_stub_config()
  - BỎ: epoch loop, save_model(), store_experience(), train()
  - Dùng VanetEnvironment.from_config(config) — tạo env trực tiếp từ config
  - _control_loop chỉ inference: get_state → select_action → env.step → log
  - agent.set_eval_mode() → ε=0, greedy hoàn toàn
"""
import os
import sys
import threading

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


class SdnVanetRyuApp(app_manager.RyuApp):
    """
    Ryu SDN controller cho VANET-UAV.
    Deploy mode: load model đã train → inference only, không train lại.
    """

    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SdnVanetRyuApp, self).__init__(*args, **kwargs)
        self._control_thread = None
        self._control_stop   = threading.Event()
        self._control_layer  = None

    def start(self):
        super(SdnVanetRyuApp, self).start()
        self.logger.info("*** SDN-VANET Ryu app started (D3QN deploy mode). ***")

        config = get_config()

        # Tạo env trực tiếp từ config — không cần stub nodes ngoài
        env = VanetEnvironment.from_config(config)

        agent = D3QNAgent(
            state_size          = env.state_size,
            action_size         = env.action_size,
            num_offload_targets = env.num_offload_targets,
            config              = config,
        )

        # Load model đã train từ main_thesis.py
        model_path = getattr(config, 'model_path', 'agents/models/d3qn.pth')
        if not os.path.exists(model_path):
            self.logger.warning(
                "*** Model không tìm thấy tại '%s'. "
                "Hãy train trước: algo_mode='drl' sudo python3 main_thesis.py ***",
                model_path
            )
        agent.load_model()

        # Tắt hoàn toàn exploration — greedy inference
        agent.set_eval_mode()
        self.logger.info("*** Model loaded. Eval mode (ε=0). ***")

        self._control_layer = ControlLayer(env, agent)
        self._control_stop.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop, daemon=True
        )
        self._control_thread.start()
        self.logger.info("*** Control loop started. ***")

    def _control_loop(self):
        """
        Vòng lặp inference liên tục — KHÔNG train, KHÔNG lưu model.

        Mỗi bước:
          1. get_state()      — lấy state từ VanetEnvironment
          2. select_action()  — greedy (ε=0)
          3. env.step()       — cập nhật cache/CPU state, tính delay
          4. log              — offload target, bitrate, cache, delay
        """
        env   = self._control_layer.env
        agent = self._control_layer.agent
        step  = 0

        env.reset()
        self.logger.info("*** Inference loop running... ***")

        while not self._control_stop.is_set():
            try:
                step += 1
                state      = env.get_state()
                action_idx = agent.select_action(state)
                _, reward, _, _ = env.step(action_idx)

                if step % 100 == 1:
                    decision = self._control_layer.get_decision(action_idx)
                    self.logger.info(
                        "step %d  offload→%s  bitrate=%s  cache=%d  delay=%.4fs",
                        step,
                        decision['offload_name'],
                        decision['bitrate_label'],
                        decision['cache'],
                        -reward,
                    )

            except (KeyboardInterrupt, SystemExit):
                self.logger.info("*** Control loop stopping. ***")
                self._control_stop.set()
                break
            except Exception as e:
                self.logger.exception("Control loop error at step %d: %s", step, e)

        self.logger.info("*** Control loop stopped after %d steps. ***", step)

    def stop(self):
        self._control_stop.set()
        super(SdnVanetRyuApp, self).stop()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Khi OpenFlow switch kết nối tới Ryu."""
        datapath = ev.msg.datapath
        self.logger.info(
            "*** Switch connected: datapath_id=%s ***", hex(datapath.id)
        )

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Bắt packet từ switch — có thể mở rộng để cài flow rule theo quyết định D3QN."""
        msg      = ev.msg
        datapath = msg.datapath
        self.logger.debug(
            "PacketIn: datapath=%s in_port=%s",
            hex(datapath.id), msg.match.get('in_port', '?')
        )