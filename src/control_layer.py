#!/usr/bin/env python3
"""
ControlLayer: SDN control plane logic dùng chung cho main_thesis.py và ryu_app.py.

Tách ra file riêng để tránh duplicate code giữa 2 entry point.
"""
import math


def _node_distance(n1, n2):
    """Tính khoảng cách 2D giữa 2 node từ params['position']."""
    def _pos(n):
        if hasattr(n, 'params') and 'position' in n.params:
            p = n.params['position']
            return float(p[0]), float(p[1])
        pos = getattr(n, 'position', None) or getattr(n, 'pos', None)
        if pos:
            return float(pos[0]), float(pos[1])
        return 0.0, 0.0
    x1, y1 = _pos(n1)
    x2, y2 = _pos(n2)
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Phạm vi phủ sóng — phải khớp với hằng số trong main_thesis.py
UAV_RANGE = 100
MBS_RANGE = 250


class ControlLayer:
    """
    Mỗi bước: state → action → env.step → store_experience → train.
    Trả về (action_idx, reward).
    """

    def __init__(self, env, agent):
        self.env   = env
        self.agent = agent

    def step(self):
        state                          = self.env.get_state()
        action_idx                     = self.agent.select_action(state)
        next_state, reward, done, _    = self.env.step(action_idx)
        self.agent.store_experience(state, action_idx, reward, next_state, done)
        self.agent.train()
        return action_idx, reward

    def get_decision(self, action_idx):
        """
        Trả về dict mô tả action để log.
        Dùng get_action_components() từ environment — khớp với action space mới.
        """
        return self.env.get_action_components(action_idx)

    def get_forced_ap_name(self, action_idx, cars, uavs):
        """
        Trả về tên AP mà car0 nên bám theo quyết định của agent,
        CHỈ khi car0 thực sự nằm trong vùng phủ của AP đó.
        Nếu ngoài vùng phủ → trả về None.

        Dùng get_action_vector() 3-dim: (offload, z_req, cache).
        """
        if not cars:
            return None

        # get_action_vector trả (offload, z_req, cache) — chỉ cần offload
        action_tuple = self.agent.get_action_vector(action_idx)
        off_idx = action_tuple[0]

        if off_idx == 0:
            return None   # local, không cần bám AP

        car = cars[0]
        target_node = None
        tmp = off_idx - 1
        if tmp < len(uavs):
            target_node = uavs[tmp]
        else:
            rsu_idx = tmp - len(uavs)
            rsus    = self.env.rsus
            if 0 <= rsu_idx < len(rsus):
                target_node = rsus[rsu_idx]

        if target_node is None:
            return None

        name_l     = getattr(target_node, 'name', '').lower()
        this_range = MBS_RANGE if ('rsu' in name_l or 'mbs' in name_l) else UAV_RANGE

        dist = _node_distance(car, target_node)
        if dist <= this_range:
            return getattr(target_node, 'name', None)
        return None