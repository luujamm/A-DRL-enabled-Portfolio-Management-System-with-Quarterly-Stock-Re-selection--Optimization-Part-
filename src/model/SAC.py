import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import Normal
from .base import StochasticRL, PolicyBased, Qnet
from .models import CNN_tcn, FNN


EPS = 1e-8
epsilon = 1e-6


class SAC(StochasticRL):
    def __init__(self, args, action_dim):
        super().__init__(args, action_dim)
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_scale = 0.5
        self.action_bias = 0.5

        self.policy = PolicyBased(args, self.state_length, self.action_dim).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.args.lra)
        self.critic = Qnet(args, self.state_length, self.action_dim).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.args.lrv)
        self.target_critic = Qnet(args, self.state_length, self.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=args.lrv)

    def get_action_dist(self, action_mean, action_log_std):
        action_std = action_log_std.exp()
        normal = Normal(action_mean, action_std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(action_mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def choose_action(self, state, weight):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.device)
            weight = torch.FloatTensor(weight[np.newaxis, :]).to(self.device)     
            action_mean, aciton_log_std = self.policy(state, weight)
            action, action_logprob, mean = self.get_action_dist(action_mean, aciton_log_std)
            
            use_action = self.relu(action).cpu().numpy().flatten() + EPS
            use_action /= np.sum(use_action)
            
            return use_action, action_mean.cpu().numpy().flatten(), action_logprob.cpu().numpy().flatten()
        
    def evaluate(self, state, weight):  
        action_mean, aciton_log_std = self.policy(state, weight)
        action, action_logprob, mean = self.get_action_dist(action_mean, aciton_log_std)   
        return action, action_logprob

    def append(self, state, state_, weight, weight_, action, reward, done):
        self.basic_append(state, action, weight, reward, done)
        self.buffer.states_.append(torch.tensor(state_, dtype=torch.float))        
        self.buffer.weights_.append(torch.tensor(weight_, dtype=torch.float))
        
    def update(self):
        loss_fn = nn.MSELoss()
        total_loss = 0

        states = self.change_form(self.buffer.states)
        actions = self.change_form(self.buffer.actions)
        weights = self.change_form(self.buffer.weights)
        states_ = self.change_form(self.buffer.states_)
        weights_ = self.change_form(self.buffer.weights_)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.buffer.is_terminals, dtype=torch.float32).to(self.device)
        
        idx_gen = self.batch_idx_generator(self.buffer.__len__(), self.batch_size)
        idx = next(idx_gen)

        action = actions[idx]
        state = states[idx]
        state_ = states_[idx]
        weight = weights[idx]
        weight_ = weights_[idx]
        reward = rewards[idx].unsqueeze(1)
        done = dones[idx].unsqueeze(1)

        a = 0.1

        with torch.no_grad():
            action_, logprob_ = self.evaluate(state_, weight_) 
            new_q1_value, new_q2_value = self.target_critic(state_, weight_, action_) 
            min_q_value = a * torch.min(new_q1_value, new_q2_value) - self.alpha * logprob_ 
            next_q_value = reward + (1 - done) * self.gamma * min_q_value
        
        q1_value, q2_value = self.critic(state, weight, action)
        
        critic_loss = loss_fn(q1_value, next_q_value) + loss_fn(q2_value, next_q_value)
        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action, logprob = self.evaluate(state, weight)
        q1_value, q2_value = self.critic(state, weight, action)
        
        # update policy
        policy_loss = (self.alpha * logprob - a * torch.min(q1_value, q2_value)).mean()
        total_loss += policy_loss.sum().item()
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        # update alpha
        alpha_loss = -(self.log_alpha * (logprob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp()

        self.train_loss.append(total_loss)

        # Copy new weights into old policy
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1-self.tau) * target_param.data)
        
        # std decay
        self.std_train = max(self.std_train * self.std_decay, 1e-4)
        
        # clear buffer
        self.buffer.clear()
        
    def save(self, model_path):
        torch.save(self.policy.state_dict(), model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        self.policy.load_state_dict(model)