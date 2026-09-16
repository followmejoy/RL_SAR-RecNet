import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import numpy as np


# ---------------- Replay Buffer ----------------
class ReplayBuffer:
    def __init__(self, capacity=100000, device="cpu"):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state.detach().cpu(),
            action.detach().cpu(),
            reward,
            next_state.detach().cpu(),
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states).to(self.device),
            torch.stack(actions).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device),
            torch.stack(next_states).to(self.device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )

    def __len__(self):
        return len(self.buffer)


# ---------------- Actor Network ----------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        h = self.backbone(state)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state, deterministic=False, eps=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = mean if deterministic else normal.rsample()
        raw_action = torch.tanh(z)
        if deterministic:
            log_prob = None
        else:
            log_prob = normal.log_prob(z) - torch.log(1 - raw_action.pow(2) + eps)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        return raw_action, log_prob


# ---------------- Critic Network ----------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ---------------- SAC Agent (最终修复版) ----------------
class SACAgent:
    def __init__(self, state_dim, action_dim, max_lambda_dn, max_lambda_pr, device,
                 gamma=0.99, tau=0.005, lr=3e-4, alpha=0.2, automatic_entropy_tuning=True):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(100000, device=device)
        self.gamma = gamma
        self.tau = tau
        self.max_lambda_dn = max_lambda_dn
        self.max_lambda_pr = max_lambda_pr
        self.automatic_entropy_tuning = automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=device, requires_grad=True)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        else:
            self.alpha = alpha

    @property
    def current_alpha(self):
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return torch.tensor(self.alpha, dtype=torch.float32, device=self.device)

    def map_action(self, raw_actions):
        min_lambda_dn = 0.01
        min_lambda_pr = 0.01
        lambda_dn = min_lambda_dn + ((raw_actions[:, 0] + 1) / 2) * (self.max_lambda_dn - min_lambda_dn)
        lambda_pr = min_lambda_pr + ((raw_actions[:, 1] + 1) / 2) * (self.max_lambda_pr - min_lambda_pr)
        return torch.stack([lambda_dn, lambda_pr], dim=-1)

    def select_raw_action(self, state, deterministic=False):
        state = state.to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            raw_action, _ = self.actor.sample(state, deterministic=deterministic)
        return raw_action.squeeze(0)

    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def update(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return None

        states, raw_actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # 确保所有张量类型正确
        states = states.float().to(self.device)
        raw_actions = raw_actions.float().to(self.device)
        next_states = next_states.float().to(self.device)
        rewards = rewards.float().to(self.device)
        dones = dones.float().to(self.device)

        # 关键修复1：强制启用梯度计算（覆盖外部no_grad上下文）
        with torch.enable_grad():
            # 确保所有SAC网络处于训练模式
            self.actor.train()
            self.critic1.train()
            self.critic2.train()
            self.target_critic1.eval()
            self.target_critic2.eval()

            # 显式确保Critic参数需要梯度
            for param in self.critic1.parameters():
                param.requires_grad = True
            for param in self.critic2.parameters():
                param.requires_grad = True

            # 计算目标Q值
            with torch.no_grad():
                next_raw_actions, next_log_probs = self.actor.sample(next_states, deterministic=False)
                next_q1 = self.target_critic1(next_states, next_raw_actions)
                next_q2 = self.target_critic2(next_states, next_raw_actions)
                next_q = torch.min(next_q1, next_q2)
                target_q = rewards + self.gamma * (1 - dones) * (next_q - self.current_alpha.detach() * next_log_probs)

            # Critic更新
            q1 = self.critic1(states, raw_actions)
            q2 = self.critic2(states, raw_actions)

            # 梯度检查（如果仍然出错，会打印详细信息）
            if not q1.requires_grad:
                print(f"ERROR: q1 has no gradient! q1.requires_grad={q1.requires_grad}, q1.grad_fn={q1.grad_fn}")
                print(f"critic1 training mode: {self.critic1.training}")
                print(f"First param requires_grad: {next(self.critic1.parameters()).requires_grad}")
                raise RuntimeError("Critic1 output has no gradient")

            loss_q1 = nn.MSELoss()(q1, target_q)
            loss_q2 = nn.MSELoss()(q2, target_q)

            self.critic1_opt.zero_grad()
            loss_q1.backward()
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
            self.critic1_opt.step()

            self.critic2_opt.zero_grad()
            loss_q2.backward()
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
            self.critic2_opt.step()

            # Actor更新
            raw_actions_new, log_probs = self.actor.sample(states, deterministic=False)
            q1_new = self.critic1(states, raw_actions_new)
            q2_new = self.critic2(states, raw_actions_new)
            q_values = torch.min(q1_new, q2_new)
            actor_loss = (self.current_alpha.detach() * log_probs - q_values).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            # Alpha更新
            alpha_loss = None
            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()

            # 软更新目标网络
            self.soft_update(self.target_critic1, self.critic1)
            self.soft_update(self.target_critic2, self.critic2)

        return {
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.current_alpha.item(),
            "alpha_loss": None if alpha_loss is None else alpha_loss.item()
        }