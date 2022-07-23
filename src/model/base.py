import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import MultivariateNormal
from .buffer import RolloutBuffer
from .models import FNN, CriticFC, create_model


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
STEP = 0.01


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class RLBase(nn.Module):
    def __init__(self, args, action_dim):
        super().__init__()
        setup_seed(args.seed)

        self.args = args
        self.action_dim = action_dim
        self.device = args.device
        self.state_length = args.state_length
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.relu = nn.ReLU()

        self.train_reward = []
        self.train_value = []
        self.val_reward = []
        self.val_value = []
        self.train_loss = []
        self.val_loss = []

    def setup_seed_(self, seed):
        setup_seed(seed)


class StochasticRL(RLBase):
    def __init__(self, args, action_dim):
        super().__init__(args, action_dim)
        self.std = args.action_std_test if self.args.test or self.args.backtest else args.action_std_train
        self.std_train = args.action_std_train
        self.std_decay = args.action_std_decay_rate
        self.step = torch.FloatTensor([STEP]).to(self.device)
        self.buffer = RolloutBuffer()
    
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
    
    def basic_append(self, state, action, weight, reward, done):
        self.buffer.states.append(torch.tensor(state, dtype=torch.float))
        self.buffer.actions.append(torch.tensor(action, dtype=torch.float))
        self.buffer.weights.append(torch.tensor(weight, dtype=torch.float))
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(int(done))  
    
    def change_form(self, a_list):
        return torch.squeeze(torch.stack(a_list, dim=0)).detach().to(self.device)
            

class ActorCritic(nn.Module):
    def __init__(self, args, state_length, action_dim):
        super().__init__()
        self.args = args
        self.hidden_dim1 = 64
        self.hidden_dim2 = 64
        self.output_dim = 1
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)
        
        if self.args.algo == 'PPO':
            self.CNN = create_model(args, state_length, action_dim)
            self.critic = FNN(args, action_dim, self.hidden_dim1,
                              self.hidden_dim2, self.output_dim)

        elif self.args.algo == 'DDPG':
            self.actor = create_model(args, state_length, action_dim)
            self.critic = create_model(args, state_length, action_dim)
            self.critic_fc = CriticFC(action_dim, self.hidden_dim2, self.output_dim)    
        
        
    def forward(self, state, weight):
        if self.args.algo == 'PPO':
            x = self.CNN(state, weight)
            value = self.critic(x)

        elif self.args.algo == 'DDPG':
            x = self.actor(state, weight) 
            s = self.critic(state, weight)
            value = self.critic_fc(s, x)
            
        action = self.softmax(x)
        return value, action


class PolicyBased(nn.Module):
    def __init__(self, args, state_length, action_dim):
        super().__init__()
        self.args = args
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)
    
        if self.args.algo == 'SAC':
            self.actor = create_model(args, state_length, action_dim)
            self.log_std_net = nn.Linear(action_dim, action_dim)

    def forward(self, state, weight):
        x = self.actor(state, weight)
        action_mean = self.softmax(x)     
        action_log_std = self.log_std_net(x)
        action_log_std = torch.clamp(action_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return action_mean, action_log_std


class Qnet(nn.Module):
    def __init__(self, args, state_length, action_dim):
        super().__init__()
        self.args = args
        self.hidden_dim1 = 128
        self.hidden_dim2 = 128
        self.output_dim = 1
        self.relu = nn.ReLU()
        
        self.q1 = create_model(args, state_length, action_dim)
        self.q1_fc = FNN(args, action_dim * 2, self.hidden_dim1,
                            self.hidden_dim2, self.output_dim)
        self.q2 = create_model(args, state_length, action_dim)
        self.q2_fc = FNN(args, action_dim * 2, self.hidden_dim1,
                            self.hidden_dim2, self.output_dim)
    
    def forward(self, state, weight, action):
        x1 = self.q1(state, weight)
        x1 = torch.concat((x1, action), 1)
        q1 = self.q1_fc(x1)
        x2 = self.q2(state, weight)
        x2 = torch.concat((x2, action), 1)
        q2 = self.q2_fc(x2)
        return q1, q2