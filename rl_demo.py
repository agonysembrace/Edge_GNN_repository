import torch
import torch.nn as nn
import torch.optim as optim

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, 128)
        self.policy_head = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return torch.softmax(self.policy_head(x), dim=-1)

# 初始化
state_dim = 4  # 状态维度
action_dim = 2  # 动作维度
policy_net = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
epsilon = 0.2

# 采样数据（这里假设有一批样本数据）
states = torch.rand(10, state_dim)
actions = torch.randint(0, action_dim, (10,))
advantages = torch.rand(10)

# 计算旧策略的动作概率
with torch.no_grad():
    old_probs = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze()

# PPO更新
for i in range(4):  # Typically we run multiple epochs
    action_probs = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze()
    ratio = action_probs / old_probs
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages
    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("PPO Update Done!")
