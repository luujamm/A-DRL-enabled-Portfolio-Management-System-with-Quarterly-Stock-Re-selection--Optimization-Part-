import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import MultivariateNormal
from .models import CNN_res, CNN_tcn, FNN, CNN_EIIE, CriticFC, create_model


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
            #self.critic = FNN(args, action_dim, self.hidden_dim1,
            #                  self.hidden_dim2, self.output_dim)
            self.critic = create_model(args, day_length, action_dim)
            self.critic_fc = CriticFC(action_dim, self.hidden_dim2, self.output_dim)
        elif self.args.algo == 'DDPG':
            self.actor = create_model(args, day_length, action_dim)
            self.critic = create_model(args, day_length, action_dim)
            self.critic_fc = CriticFC(action_dim, self.hidden_dim2, self.output_dim)
        
        self.softmax = nn.Softmax(1)
        
    def forward(self, state, weight):
        if self.args.algo == 'PPO':
            x = self.CNN(state, weight)
            s = self.critic(state, weight)
            value = self.critic_fc(s, weight)
            #value = self.critic(x)
        elif self.args.algo == 'DDPG':
            x = self.actor(state, weight) 
            s = self.critic(state, weight)
            value = self.critic_fc(s, weight)
            
        action = self.softmax(x)
        
        return value, action


class PolicyBased(nn.Module):
    def __init__(self, args, day_length, action_dim):
        super(PolicyBased, self).__init__()
        self.args = args
        self.relu = nn.ReLU()
    
        if self.args.algo == 'DPG':
            self.actor = create_model(args, day_length, action_dim)
        
        self.softmax = nn.Softmax(1)

    def forward(self, state, weight):
        x = self.actor(state, weight)
        action = self.softmax(x)     
        return action 