import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from src.IP import *
from src.LTNN import *
from tqdm import tqdm
import random
import torch.nn.functional as F

class MaskedSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.sigmoid()
        output[input > 0.9] = 1
        output[input < 0.1] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        sigmoid_backward = input.sigmoid() * (1 - input.sigmoid())
        return grad_input * sigmoid_backward

class MaskedSigmoid(nn.Module):
    def forward(self, input):
        return MaskedSigmoidFunction.apply(input)

class OUNoiseParam:
    def __init__(self, model, theta=0.15, sigma=0.2, decay_rate=0.999):
        self.theta = theta
        self.decay_rate = decay_rate
        self.sigma = sigma
        self.model = model
        self.noise_state = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        self.reset()

    def reset(self):
        for name in self.noise_state:
            self.noise_state[name].fill_(0)

    def noise(self):
        for name, param in self.model.named_parameters():
            delta = self.theta * -self.noise_state[name] + self.sigma * torch.randn_like(param)
            self.noise_state[name] += delta
        self.sigma *= self.decay_rate
        self.sigma = max(self.sigma, 0.01)
        return self.noise_state

class Actor(nn.Module):
    def __init__(self, num_constraints, num_variables, N=16, K=2, num_cuts=1, hidden_channels=[32,32], use_shortcut=False, activation='StepSigmoid', squeeze='Sigmoid', use_bn=False, problem_type='knapsack'):
        super(Actor, self).__init__()
        self.N = N
        self.K = K
        self.num_constraints = num_constraints
        self.num_variables = num_variables
        self.num_cuts = num_cuts
        self.state_dim = N + K
        self.action_dim = num_constraints * num_cuts + num_cuts * (num_cuts - 1) // 2
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.MaskedSigmoid = MaskedSigmoid()
        self.use_shortcut = use_shortcut
        self.use_bn = use_bn
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_bn else []
        self.bn_output = nn.BatchNorm1d(self.action_dim)
        num_hidden_layers = len(hidden_channels)
        self.hidden_layers.append(nn.Linear(self.state_dim, hidden_channels[0]))
        if use_bn:
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels[0]))
        for i in range(1, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_channels[i-1], hidden_channels[i]))
            if use_bn:
                self.bn_layers.append(nn.BatchNorm1d(hidden_channels[i]))
        self.output_layer = nn.Linear(hidden_channels[-1], self.action_dim)
        self.activation = get_activation(activation)
        self.squeeze = get_activation(squeeze)

    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
        x = self.output_layer(x)
        x = self.squeeze(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_channels=[128,128], activation='ReLU', use_bn=False):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ReLU = nn.ReLU()
        self.use_bn = use_bn
        self.bn_0 = nn.BatchNorm1d(self.state_dim + self.action_dim)
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_bn else []
        num_hidden_layers = len(hidden_channels)
        self.hidden_layers.append(nn.Linear(self.state_dim + self.action_dim, hidden_channels[0]))
        if use_bn:
            self.bn_layers.append(nn.BatchNorm1d(hidden_channels[0]))
        for i in range(1, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_channels[i-1], hidden_channels[i]))
            if use_bn:
                self.bn_layers.append(nn.BatchNorm1d(hidden_channels[i]))
        self.output_layer = nn.Linear(hidden_channels[-1], 1)
        self.activation = get_activation(activation)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x

class DDPG:
    def __init__(
                    self,
                    N,
                    K,
                    num_cuts=1,
                    hidden_channels_actor=[128,128],
                    hidden_channels_critic=[128,128],
                    activation="StepSigmoid",
                    squeeze="Sigmoid",
                    use_bn=False,
                    sigma=0.01,
                    actor_lr=3e-4,
                    critic_lr=3e-3,
                    tau=0.005,
                    gamma=0.98,
                    exploration_rate=1.0,
                    device=torch.device("cpu"),
                    problem_type='knapsack',
                    ):
        self.problem_type = problem_type
        num_variables = N * K
        num_constraints = N + K + 2 * num_variables

        self.actor = Actor(num_constraints=num_constraints, num_variables=num_variables, num_cuts=num_cuts, hidden_channels=hidden_channels_actor, activation=activation, squeeze=squeeze, use_bn=use_bn, problem_type=problem_type).to(device)
        self.critic1 = Critic(state_dim=self.actor.state_dim, action_dim=self.actor.action_dim, hidden_channels=hidden_channels_critic, activation="ReLU", use_bn=use_bn).to(device)
        self.target_actor = Actor(num_constraints=num_constraints, num_variables=num_variables, num_cuts=num_cuts, hidden_channels=hidden_channels_actor, activation=activation, squeeze=squeeze, use_bn=use_bn, problem_type=problem_type).to(device)
        self.target_critic1 = Critic(state_dim=self.actor.state_dim, action_dim=self.actor.action_dim, hidden_channels=hidden_channels_critic, activation="ReLU", use_bn=use_bn).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2 = Critic(state_dim=self.actor.state_dim, action_dim=self.actor.action_dim, hidden_channels=hidden_channels_critic, activation="ReLU", use_bn=use_bn).to(device)
        self.target_critic2 = Critic(state_dim=self.actor.state_dim, action_dim=self.actor.action_dim, hidden_channels=hidden_channels_critic, activation="ReLU", use_bn=use_bn).to(device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.action_dim = self.actor.action_dim
        self.param_noise = OUNoiseParam(self.actor)
        self.exploration_rate = exploration_rate
        self.device = device

    def take_action(self, state, problem_type='knapsack'):
        with torch.no_grad():
            state = state.to(self.device)
            state = torch.cat((state[:, 0:16], state[:, -82:-80]), dim=1)
            action = self.actor(state)
            if random.random() < self.exploration_rate:
                action = torch.tensor(np.random.rand(self.action_dim), dtype=torch.float).to(self.device)
            self.exploration_rate *= 0.995
            self.exploration_rate = max(self.exploration_rate, 0.1)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        with torch.no_grad():
            states = transition_dict['states'].to(self.device)
            actions = transition_dict['actions'].to(self.device)
            rewards = transition_dict['rewards'].to(self.device).unsqueeze(1)
            next_states = transition_dict['next_states'].to(self.device)
            dones = transition_dict['dones'].to(self.device)
            states = torch.cat((states[:, 0:16], states[:, -82:-80]), dim=1)
            next_states = torch.cat((next_states[:, 0:16], next_states[:, -82:-80]), dim=1)
            next_q1 = self.target_critic1(next_states, self.target_actor(next_states))
            next_q2 = self.target_critic2(next_states, self.target_actor(next_states))
            next_q = torch.min(next_q1, next_q2)
            q_targets = rewards + self.gamma * next_q * (1 - dones)
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss1 = F.mse_loss(current_q1, q_targets.detach())
        critic_loss2 = F.mse_loss(current_q2, q_targets.detach())
        self.critic1_optimizer.zero_grad()
        critic_loss1.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic_loss2.backward()
        self.critic2_optimizer.step()
        actor_loss = -self.critic1(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        self.soft_update(self.actor, self.target_actor)
        return ((critic_loss1 + critic_loss2) / 2).item()


