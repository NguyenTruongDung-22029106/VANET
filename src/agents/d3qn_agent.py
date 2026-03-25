#!/usr/bin/env python3
"""
D3QN agent for SDN–VANET–UAV (control plane logic).

Action space (3-chiều):
  UAV tier : a = uav_idx + L*(z_cached + Z*cache_dec)
  MBS tier : a = L*Z*2
State: from VanetEnvironment (see environment.py).
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import random
from collections import deque

import numpy as np
import torch
torch.set_num_threads(1)
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
    """D3QN agent for tier decision + bitrate selection + caching.

    Action layout (3-chiều, khớp VanetEnvironment):
      - UAV tier actions: a = uav_idx + L*(z_cached + Z*cache_dec)
          uav_idx  in [0..L-1]
          z_cached in [0..Z-1]  ← bitrate cần cache
          cache_dec in {0, 1}
      - MBS tier action : a = L*Z*2  (= _uav_action_size)

    Tổng: action_size = L*Z*2 + 1
    Với mặc định L=5, Z=4: action_size = 41
    """

    def __init__(
        self,
        state_size,
        action_size,
        num_offload_targets,
        config,
        num_bitrates: int = None,
    ):
        """
        Args:
            state_size          : kích thước vector state
            action_size         : L*Z*2 + 1 (tổng số actions)
            num_offload_targets : số UAV (đích offload) trong environment  = L
            config              : SimpleNamespace hoặc dict
            num_bitrates        : số mức bitrate = Z (mặc định lấy từ config, fallback 4)
        """
        self.state_size          = state_size
        self.action_size         = action_size
        self.num_offload_targets = num_offload_targets
        self.num_cache_actions   = 2
        backup_z = 4
        if isinstance(config, dict):
            backup_z = config.get('num_bitrates', 4)
        else:
            backup_z = getattr(config, 'num_bitrates', 4)
        self.num_bitrates = int(num_bitrates if num_bitrates is not None else backup_z)

        self.memory        = deque(maxlen=50_000)
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 5e-4
        self.batch_size    = 256

        self.policy_net = DuelingDQN(state_size, action_size, hidden_size=256).to(device)
        self.target_net = DuelingDQN(state_size, action_size, hidden_size=256).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        if isinstance(config, dict):
            self.model_path = config.get("model_path", "agents/models/d3qn.pth")
        else:
            self.model_path = getattr(config, "model_path", "agents/models/d3qn.pth")

        self.train_steps            = 0
        self.target_update_interval = 1000
        self.losses                 = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mbs_action_idx(self) -> int:
        """Index của MBS-tier action: L * Z * 2."""
        return int(self.num_offload_targets) * int(self.num_bitrates) * self.num_cache_actions

    def select_action(self, state):
        """
        Epsilon-greedy.
        Dùng was_training để không phá set_eval_mode().
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        was_training = self.policy_net.training
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state)
        if was_training:
            self.policy_net.train()
        return q_values.argmax().item()

    def get_action_vector(self, action_idx):
        """
        Decode action_idx thành tuple mô tả quyết định — 3-chiều.

        Returns:
          ('mbs', -1, -1, 0)               — MBS/RSU tier
          ('uav', uav_idx, z_cached, cache) — UAV tier
        where:
          uav_idx  : UAV được chọn       [0..L-1]
          z_cached : bitrate cần cache   [0..Z-1]
          cache    : 0=no_cache, 1=cache
        """
        a = int(action_idx)
        L = max(int(self.num_offload_targets), 1)
        Z = max(int(self.num_bitrates), 1)                 # ← FIX: dùng Z thay vì bỏ qua
        mbs_idx = self._mbs_action_idx()

        if a == mbs_idx:
            return ('mbs', -1, -1, 0)

        # ← FIX: decode đúng encoding 3 chiều của VanetEnvironment
        uav_idx  = a % L
        t        = a // L
        z_cached = int(t % Z)
        cache    = int(t // Z)
        return ('uav', int(uav_idx), int(z_cached), int(cache))

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
            return False
        try:
            try:
                state_dict = torch.load(
                    self.model_path, map_location=device, weights_only=True
                )
            except TypeError:
                state_dict = torch.load(self.model_path, map_location=device)

            self.policy_net.load_state_dict(state_dict, strict=False)
            self.target_net.load_state_dict(self.policy_net.state_dict(), strict=False)
            print(f"Model loaded from {self.model_path} (strict=False)")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def set_eval_mode(self):
        """Tắt exploration và giữ network ở eval mode."""
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