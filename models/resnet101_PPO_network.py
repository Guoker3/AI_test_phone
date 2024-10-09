import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from WZ_1v1_env import wz_1v1_env


class resnet101(nn.Module):
    def __init__(self, resnet):
        super(resnet101, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=6):
        x = img

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()  # Tensor:(2048,)
        # att_size=6:输出的特征图将是6x6的尺寸
        att = F.adaptive_avg_pool2d(x, (att_size, att_size)).squeeze().permute(1, 2, 0)  # Tensor:(6,6,2048)

        return fc, att


class ppo_policy_net(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(ppo_policy_net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b, n_actions]
        x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
        return x


class ppo_value_net(nn.Module):
    # TODO(fyt) for quickly start, I use red-judge to generate rewards from envs. And we could use multi-rewards or added rewards later.
    def __init__(self, n_states, n_hiddens):
        super(ppo_value_net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x


class PPO:
    def __init__(self, n_states, n_actions, gpu, n_hiddens=16,
                 actor_lr=1e-3, critic_lr=1e-2, lmbda=0.95, epochs=10, eps=0.2, gamma=0.9):
        # 实例化策略网络
        self.actor = ppo_policy_net(n_states, n_hiddens, n_actions).to(gpu)
        # 实例化价值网络
        self.critic = ppo_value_net(n_states, n_hiddens).to(gpu)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.gpu = gpu

    # 动作选择
    def take_action(self, state_tensor):
        # 维度变换 [n_state]-->tensor[1,n_states]
        # state_tensor = torch.tensor(state[0], dtype=torch.float).to(self.gpu)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state_tensor)
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action

    def learn(self, transition_dict):
        # 提取数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.gpu)
        actions = torch.tensor(transition_dict['actions']).to(self.gpu).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.gpu).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.gpu)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.gpu).view(-1, 1)

        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value

        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.gpu)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 梯度清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class resnet101_PPO_network(nn.Module):
    def __init__(self, device_serial='356a7b7'):
        super(resnet101_PPO_network, self).__init__()
        self.gpu = torch.device("cuda")
        self.resnet = resnet101(
            torchvision.models.resnet101(pretrained=True).eval().cuda(self.gpu).requires_grad_(False))
        self.resnet_output_size = 6  # x^2
        n_states = self.resnet_output_size * self.resnet_output_size * 2048
        n_hiddens = 16
        n_actions = 4
        actor_lr = 1e-3  # 策略网络的学习率
        critic_lr = 1e-2  # 价值网络的学习率
        gamma = 0.9  # 折扣因子
        self.agent = PPO(gpu=self.gpu,
                         n_states=n_states,  # 状态数
                         n_hiddens=n_hiddens,  # 隐含层数
                         n_actions=n_actions,  # 动作数
                         actor_lr=actor_lr,  # 策略网络学习率
                         critic_lr=critic_lr,  # 价值网络学习率
                         lmbda=0.95,  # 优势函数的缩放因子
                         epochs=10,  # 一组序列训练的轮次
                         eps=0.2,  # PPO中截断范围的参数
                         gamma=gamma,  # 折扣因子
                         )
        self.total_reward = 0
        self.environment = wz_1v1_env(device_serial, self.gpu, self.resnet)
        self.transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

    def train_start(self):
        screen_torch = self.environment.device.UI_device.get_screen_torch()
        self.environment.reset_env(screen_torch)

    def train_step(self):
        screen_torch = self.environment.device.UI_device.get_screen_torch()
        _, state = self.resnet(screen_torch)
        state = torch.reshape(state, (1, 6 * 6 * 2048))
        action = self.agent.take_action(state)  # 动作选择
        time_start_action = time.time()
        next_state, reward = self.environment.step_env(action)  # 环境更新
        print("reward : %s, action time cost : %s" % (reward, time.time() - time_start_action))
        # 保存每个时刻的状态\动作\...
        self.transition_dict['states'].append(state)
        self.transition_dict['actions'].append(action)
        self.transition_dict['next_states'].append(next_state)
        self.transition_dict['rewards'].append(reward)
        # self.transition_dict['dones'].append(done)
        # 更新状态
        # state = next_state
        # 累计回合奖励
        self.total_reward += reward


if __name__ == "__main__":
    train_network = resnet101_PPO_network()
    train_network.train_start()
    for i in range(100):
        train_network.train_step()
    train_network.environment.device.stop()
