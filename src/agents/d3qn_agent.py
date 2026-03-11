#!/usr/bin/env python3
"""
D3QN agent for SDN–VANET–UAV (control plane logic).

Action space: offload target (local / UAV_1..UAV_n / RSU_1..RSU_m)
              × bitrate (low/high) × cache (no/yes).
State: from VanetEnvironment (positions, CPU load, cache status, video popularity).
"""
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DuelingDQN(nn.Module):
    """
    Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        features   = self.feature_layer(x)
        values     = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class D3QNAgent:
    """Agent: state -> action (offload_idx, cache_01); experience replay; epsilon-greedy."""

    def __init__(self, state_size, action_size, num_offload_targets, config):
        """
        Args:
            state_size          : kích thước vector state
            action_size         : num_offload_targets × num_bitrates × num_cache_actions
            num_offload_targets : số đích offload (local + UAV + RSU)
            config              : SimpleNamespace hoặc dict
        """
        self.state_size          = state_size
        self.action_size         = action_size
        self.num_offload_targets = num_offload_targets
        self.num_bitrates        = 2   # khớp environment.NUM_BITRATES

        self.memory        = deque(maxlen=10_000)
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size    = 64

        self.policy_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        # Hỗ trợ cả SimpleNamespace lẫn dict
        if isinstance(config, dict):
            self.model_path = config.get("model_path", "agents/models/d3qn.pth")
        else:
            self.model_path = getattr(config, "model_path", "agents/models/d3qn.pth")

        self.train_steps            = 0
        self.target_update_interval = 200
        self.losses                 = []

    def select_action(self, state):
        """
        Epsilon-greedy.
        FIX 2: dùng was_training để không phá set_eval_mode().
          - Nếu đang eval (set_eval_mode đã gọi): luôn giữ eval sau inference.
          - Nếu đang train: tạm eval để inference rồi restore lại train.
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        was_training = self.policy_net.training   # lưu trạng thái hiện tại
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        if was_training:                           # chỉ restore nếu trước đó là train
            self.policy_net.train()
        return q_values.argmax().item()

    def get_action_vector(self, action_idx):
        """
        action_idx -> (offload_idx, z_req, cache_01)

        Encoding khớp với environment._decode_action():
          action_idx = offload + num_offload * (z + num_bitrates * cache)
        """
        num_bitrates = getattr(self, 'num_bitrates', 2)
        a         = int(action_idx)
        offload   = a % self.num_offload_targets
        remainder = a // self.num_offload_targets
        z_req     = remainder % num_bitrates
        cache     = remainder // num_bitrates
        return (offload, z_req, cache)

    @staticmethod
    def _to_numpy_state(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().to("cpu").numpy()
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)
        return np.array(x, dtype=np.float32)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((
            self._to_numpy_state(state),
            int(action), float(reward),
            self._to_numpy_state(next_state),
            bool(done),
        ))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch       = random.sample(self.memory, self.batch_size)
        states      = torch.as_tensor(np.stack([x[0] for x in batch]), dtype=torch.float32, device=device)
        actions     = torch.as_tensor([x[1] for x in batch], dtype=torch.long,    device=device).unsqueeze(1)
        rewards     = torch.as_tensor([x[2] for x in batch], dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.as_tensor(np.stack([x[3] for x in batch]), dtype=torch.float32, device=device)
        dones       = torch.as_tensor([x[4] for x in batch], dtype=torch.float32, device=device).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions)
            target_q     = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.losses.append(loss.item())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.policy_net.state_dict(), self.model_path)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        if not os.path.exists(self.model_path):
            return
        try:
            # FIX 1: fallback cho PyTorch < 2.0 không có tham số weights_only
            try:
                state_dict = torch.load(
                    self.model_path, map_location=device, weights_only=True
                )
            except TypeError:
                state_dict = torch.load(self.model_path, map_location=device)

            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def set_eval_mode(self):
        """
        Tắt exploration và giữ network ở eval mode.
        select_action sẽ không restore train mode vì was_training = False.
        """
        self.epsilon     = 0.0
        self.epsilon_min = 0.0
        self.policy_net.eval()

    def set_train_mode(self):
        """Khôi phục train mode sau eval."""
        self.policy_net.train()

    def get_avg_loss(self, last_n=100):
        if not self.losses:
            return 0.0
        return float(np.mean(self.losses[-last_n:]))