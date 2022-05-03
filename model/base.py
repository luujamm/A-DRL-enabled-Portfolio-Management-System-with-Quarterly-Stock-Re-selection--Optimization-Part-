import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import MultivariateNormal
from .models import FNN, CriticFC, create_model


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

class ActorCritic(nn.Module):
    def __init__(self, args, day_length, action_dim):
        super(ActorCritic, self).__init__()
        self.args = args
        self.hidden_dim1 = 64
        self.hidden_dim2 = 64
        self.output_dim = 1
        self.relu = nn.ReLU()
        
        if self.args.algo == 'PPO':
            self.CNN = create_model(args, day_length, action_dim)
            self.critic = FNN(args, action_dim, self.hidden_dim1,
                              self.hidden_dim2, self.output_dim)
        elif self.args.algo == 'DDPG':
            self.actor = create_model(args, day_length, action_dim)
            self.critic = create_model(args, day_length, action_dim)
            self.critic_fc = CriticFC(action_dim, self.hidden_dim2, self.output_dim)
        
        self.softmax = nn.Softmax(1)
        
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
    def __init__(self, args, day_length, action_dim):
        super(PolicyBased, self).__init__()
        self.args = args
        self.relu = nn.ReLU()
    
        if self.args.algo == 'SAC':
            self.actor = create_model(args, day_length, action_dim)
            self.log_std_net = nn.Linear(action_dim, action_dim)
        
        self.softmax = nn.Softmax(1)

    def forward(self, state, weight):
        x = self.actor(state, weight)
        action_mean = self.softmax(x)     
        action_log_std = self.log_std_net(x)
        action_log_std = torch.clamp(action_log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return action_mean, action_log_std


class Qnet(nn.Module):
    def __init__(self, args, day_length, action_dim):
        super(Qnet, self).__init__()
        self.args = args
        self.hidden_dim1 = 128
        self.hidden_dim2 = 128
        self.output_dim = 1
        self.relu = nn.ReLU()
        self.q1 = create_model(args, day_length, action_dim)
        self.q1_fc = FNN(args, action_dim * 2, self.hidden_dim1,
                            self.hidden_dim2, self.output_dim)
        self.q2 = create_model(args, day_length, action_dim)
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
