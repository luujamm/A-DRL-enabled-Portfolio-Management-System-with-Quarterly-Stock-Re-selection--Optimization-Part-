import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import MultivariateNormal, Normal
from .base import PolicyBased, Qnet, setup_seed
from .models import CNN_tcn, FNN


EPS = 1e-8
STEP = 0.01
epsilon = 1e-6

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_ = []
        self.weights = []
        self.weights_ = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.states_[:]
        del self.weights[:]
        del self.weights_[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.actions)

class SAC(nn.Module):
    def __init__(self, args, action_dim):
        super(SAC, self).__init__()
        setup_seed(args.seed)
        self.action_dim = action_dim
        self.args = args
        self.device = args.device
        self.day_length = args.state_length
        self.std = args.action_std_test if self.args.test or self.args.backtest else args.action_std_train
        self.std_train = args.action_std_train
        self.std_decay = args.action_std_decay_rate
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy = PolicyBased(args, self.day_length, self.action_dim).to(self.device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=self.args.lra)
        self.critic = Qnet(args, self.day_length, self.action_dim).to(self.device)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.args.lrv)
        self.target_critic = Qnet(args, self.day_length, self.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=args.lrv)

        self.relu = nn.ReLU()
        self.step = torch.FloatTensor([STEP]).to(self.device)
        self.buffer = RolloutBuffer()
        
        #recorder
        self.train_reward = []
        self.train_value = []
        self.val_reward = []
        self.val_value = []
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []
        #recorder

        self.action_scale = 0.5
        self.action_bias = 0.5

    def setup_seed_(self, seed):
        setup_seed(seed)

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
        self.buffer.states.append(torch.tensor(state, dtype=torch.float))
        self.buffer.states_.append(torch.tensor(state_, dtype=torch.float))
        self.buffer.weights.append(torch.tensor(weight, dtype=torch.float))
        self.buffer.weights_.append(torch.tensor(weight_, dtype=torch.float))
        self.buffer.actions.append(torch.tensor(action, dtype=torch.float))
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(int(done))
        
    def shuffle_idx(self, num):
        p = np.random.permutation(num)
        return p
    
    def batch_idx_generator(self, data_len, batch_size, shuffle=True):
        if shuffle:
            idx = self.shuffle_idx(data_len)
        batch_count = 0
        while True:
            if (batch_count + 1) * batch_size > data_len:
                batch_count = 0
                if shuffle:
                    idx = self.shuffle_idx(data_len)
            start = batch_count * batch_size
            end = start + batch_size
            batch_count += 1
            yield idx[start: end]        
        
    def update(self):
        loss_fn = nn.MSELoss()
        total_loss = 0

        states = torch.squeeze(torch.stack(
            self.buffer.states, dim=0)).detach().to(self.device)
        actions = torch.squeeze(torch.stack(
            self.buffer.actions, dim=0)).detach().to(self.device)
        weights = torch.squeeze(torch.stack(
            self.buffer.weights, dim=0)).detach().to(self.device)
        states_ = torch.squeeze(torch.stack(
            self.buffer.states_, dim=0)).detach().to(self.device)
        weights_ = torch.squeeze(torch.stack(
            self.buffer.weights_, dim=0)).detach().to(self.device)
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
        a = 1
        with torch.no_grad():
            action_, logprob_ = self.evaluate(state_, weight_)        
            #logprob_ = logprob_.unsqueeze(1)
            new_q1_value, new_q2_value = self.target_critic(state_, weight_, action_)
            
            min_q_value = a * torch.min(new_q1_value, new_q2_value) - self.alpha * logprob_ 
            next_q_value = reward + (1 - done) * self.gamma * min_q_value
            
            #print(next_q_value.size())
            

        
        q1_value, q2_value = self.critic(state, weight, action)
        
        critic_loss = loss_fn(q1_value, next_q_value) + loss_fn(q2_value, next_q_value)
        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action, logprob = self.evaluate(state, weight)
        q1_value, q2_value = self.critic(state, weight, action)
        
        policy_loss = (self.alpha * logprob - a * torch.min(q1_value, q2_value)).mean()
        #print(self.alpha * logprob.mean(), a * torch.min(q1_value, q2_value).mean())
        #print(policy_loss)
        
        total_loss += policy_loss.sum().item()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        

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
        self.target_net.load_state_dict(model)
    