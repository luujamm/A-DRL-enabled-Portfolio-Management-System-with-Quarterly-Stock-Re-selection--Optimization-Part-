import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
from .base import ActorCritic, setup_seed


EPS = 1e-8


class GaussianNoise:
    def __init__(self, args, dim, mu=None, std=None):
        np.random.seed(args.seed)
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)

class OrnsteinUhlenbeckProcess:
    def __init__(self, args, mu, sigma=0.2, theta=.15, dimension=1e-2, x0=None,num_steps=12000):
        np.random.seed(args.seed)
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dimension
        self.x0 = x0
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayMemory:
    # __slots__ = ['buffer','epi_buffer','capacity']
    def __init__(self, args, capacity=252):
        np.random.seed(args.seed)
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (prev_action, state, action, reward, next_state, done)
        self.buffer.append(tuple(transition))

    #def epi_append(self):
    #    self.epi_buffer.append(self.buffer)
    #    self.buffer = []

    #def GDP_epi_idx(self, start, end, bias):  # select episodes based on geometrical distribution
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
    #    ran = np.random.geometric(bias)
    #    while ran > end - start:
    #        ran = np.random.geometric(bias)
    #    result = end - ran
    #    return result
    
    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        #epi_idx = self.GDP_epi_idx(0, self.capacity, bias=0.02) 
        #trajectory = self.epi_buffer[epi_idx]
        #batch_start = np.random.randint(low=0, high=len(trajectory)-batch_size)
        #transitions = np.array(trajectory[batch_start:batch_start+batch_size], dtype='object')
        #return (torch.tensor(x, dtype=torch.float, device=device)
        transitions = random.sample(self.buffer, batch_size)
        return (np.array(x) for x in zip(*transitions))


class DDPG(nn.Module):
    def __init__(self, args, action_dim):
        super(DDPG, self).__init__()
        setup_seed(args.seed)
        self.action_dim = action_dim
        self.args = args
        self.device = args.device
        self.day_length = args.state_length
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.behavior_network = ActorCritic(args, self.day_length, self.action_dim).to(self.device)
        self.actor_opt = optim.Adam(self.behavior_network.actor.parameters(), lr=args.lra)
        self.critic_opt = optim.Adam([{'params': self.behavior_network.critic.parameters()}, 
                                      {'params': self.behavior_network.critic_fc.parameters()}],
                                      lr=self.args.lrv)
        
        self.target_network = ActorCritic(
            args, self.day_length, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.behavior_network.state_dict())
        self.memory = ReplayMemory(args, capacity=args.capacity)
        self.Gau_var = args.Gau_var
        self.Gau_decay = args.Gau_decay
        self.tau = args.tau
        self.relu = nn.ReLU()
        #self.action_noise = OrnsteinUhlenbeckProcess(self.args, np.zeros(self.action_dim))

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

    def setup_seed_(self, seed):
        setup_seed(seed)

    def choose_action(self, state, old_action, noise_inp=True):
        state = torch.FloatTensor(state[np.newaxis, :]).to(self.device)
        old_action = torch.FloatTensor(old_action[np.newaxis, :]).to(self.device) 
        _, action = self.behavior_network(state, old_action)
        if noise_inp == True:
            self.action_noise = GaussianNoise(self.args, dim=len(old_action), mu=None, std=self.Gau_var)
            
            noise = self.action_noise.sample()
            noise = torch.FloatTensor(noise).to(self.device)
            noised_action = action + noise
            noised_action = self.relu(noised_action)
            noised_action /= torch.sum(noised_action)
            self.Gau_var *= self.Gau_decay
            return noised_action.cpu().data.numpy().flatten()
        else:
            return action.cpu().data.numpy().flatten()
    
    def append(self, old_action, state, action, reward, state_, done):
        self.memory.append(old_action, state, action, [reward], state_, [int(done)])
    
    def epi_append(self):
        self.memory.epi_append()
    
    def update(self):
        
        self.update_behavior_network()
        self.update_target_network()
    
    def update_behavior_network(self):
        loss_fn = nn.MSELoss()
        old_action, state, action, reward, state_, done = self.memory.sample(self.batch_size, self.device)
        
        old_action = torch.FloatTensor(old_action).to(self.device) 
        state = torch.FloatTensor(state).to(self.device) 
        action = torch.FloatTensor(action).to(self.device) 
        reward = torch.FloatTensor(reward).to(self.device)
        state_ = torch.FloatTensor(state_).to(self.device) 
        done = torch.FloatTensor(done).to(self.device)
        # update critic
        value_, action_ = self.target_network(state_, action)
        #print(type(reward))#, reward.size())
        #print(type(done))#, done.size())
        #print(type(value_))#, value_.size())
        #exit()
        q_target = reward + (1 - done) * self.gamma * value_.detach()
        value, _ = self.behavior_network(state, old_action)
        critic_loss = loss_fn(q_target, value)
        
        self.train_loss.append(critic_loss.mean().item())

        self.behavior_network.zero_grad()
        critic_loss.mean().backward()
        self.critic_opt.step()

        # update actor
        value, _ = self.behavior_network(state, old_action)
        actor_loss = -torch.mean(value)

        self.behavior_network.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
    
    
    def update_target_network(self):
        for target, behavior in zip(self.target_network.parameters(), self.behavior_network.parameters()):
            target.data.copy_(self.tau * behavior.data + (1-self.tau) * target.data)

    def save(self, model_path):
        torch.save({'behavior_network': self.behavior_network.state_dict(),
                    'target_network': self.target_network.state_dict()} ,model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        self.behavior_network.load_state_dict(model['behavior_network'])
    
        
    
    

