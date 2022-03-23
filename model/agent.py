import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import MultivariateNormal
from .models import CNN_res, CNN_tcn, FNN, CNN_EIIE, CriticFC

EPS = 1e-8
STEP = 0.01


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def create_model(args, day_length, action_dim):
    if args.model == 'tcn':
        model = CNN_tcn(args, day_length, action_dim)
    elif args.model == 'EIIE':
        model = CNN_EIIE(args, day_length, action_dim)
    elif args.model == 'res':
        model = CNN_res(args, day_length, action_dim)
    return model
    

def GDP_idx(self, start, end, bias):  # select episodes based on geometrical distribution
    """
    @:param end: is excluded
    @:param bias: value in (0, 1)
    """
    ran = np.random.geometric(bias)
    while ran > end - start:
        ran = np.random.geometric(bias)
    result = end - ran
    return result    


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_ = []
        self.next_values = []
        self.weights = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.states_[:]
        del self.next_values[:]
        del self.weights[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.actions)
    
class GaussianNoise:
    def __init__(self, args, dim, mu=None, std=None):
        np.random.seed(args.seed)
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)  
    

    
class ReplayMemory:
    # __slots__ = ['buffer','epi_buffer','capacity']
    def __init__(self, args, capacity=252):
        np.random.seed(args.seed)
        # self.buffer = deque(maxlen=capacity)
        self.buffer = []
        self.epi_buffer = deque(maxlen=capacity)
        self.capacity = capacity
    def __len__(self):
        return len(self.buffer)
    def append(self, *transition):
        # (prev_action, state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))
    def epi_append(self):
        self.epi_buffer.append(self.buffer)
        self.buffer = []
    def GDP_epi_idx(self, start, end, bias):  # select episodes based on geometrical distribution
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)
        result = end - ran
        return result
    
    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        # 要改成geometrical distribution的話，似乎可以從這裡
        epi_idx = self.GDP_epi_idx(0, self.capacity, bias=0.02) 
        # print('epi_idx',epi_idx)
        # print(len(self.epi_buffer))
        trajectory = self.epi_buffer[epi_idx]
        batch_start = np.random.randint(low=0, high=len(trajectory)-batch_size)
        # transitions = random.sample(self.buffer, batch_size)
        transitions = trajectory[batch_start:batch_start+batch_size]
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))
    

class ActorCritic(nn.Module):
    def __init__(self, args, day_length, action_dim, agent_name):
        super(ActorCritic, self).__init__()
        self.args = args
        self.hidden_dim1 = 64
        self.hidden_dim2 = 64
        self.output_dim = 1
        self.relu = nn.ReLU()
        
        if args.algo == 'PPO':
            self.CNN = create_model(args, day_length, action_dim)
            self.critic = FNN(args, action_dim, self.hidden_dim1,
                              self.hidden_dim2, self.output_dim)
        elif args.algo == 'DDPG':
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
            value = self.critic_fc(s, weight)
            
        action = self.softmax(x)
           
        return value, action

class PPO:
    def __init__(self, args, action_dim, agent_name):
        self.args = args
        self.device = args.device
        self.action_dim = action_dim
        self.day_length = args.state_length
        self.std = args.action_std_test if args.test or args.backtest else args.action_std_train
        self.std_train = args.action_std_train
        self.std_decay = args.action_std_decay_rate
        self.K_epochs = args.K_epochs
        self.eps_clip = args.eps_clip
        self.gamma = args.gamma
        self.dist_entropy_coef = args.dist_entropy_coef
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.policy_opt = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.args.lra},
            {'params': self.policy.critic.parameters(), 'lr': self.args.lrv}])#, weight_decay=0.01)
        self.policy_old = ActorCritic(
            args, self.day_length, self.action_dim, agent_name).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.relu = nn.ReLU()
        self.step = torch.FloatTensor([STEP]).to(self.device)
        self.buffer = RolloutBuffer()
    
    def get_action_dist(self, action_mean):
        action_std = nn.Parameter(torch.ones(1, self.action_dim) * self.std)
        action_std = action_std.expand_as(action_mean).to(self.device)
        cov_mat = torch.diag_embed(action_std).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        return dist
    
    def choose_action(self, state, weight):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.device)
            weight = torch.FloatTensor(weight[np.newaxis, :]).to(self.device)     
            value, action_mean = self.policy_old(state, weight)
            dist = self.get_action_dist(action_mean)
            action = dist.sample()
            action_log_prob = dist.log_prob(action) + torch.log(self.step) * self.action_dim
            
            use_action = self.relu(action).cpu().numpy().flatten() + EPS
            use_action /= np.sum(use_action)
            
            return use_action, action.cpu().numpy().flatten(), action_log_prob.cpu().numpy().flatten(), value
    
    def evaluate(self, state, action, weight):       
        value, action_mean = self.policy(state, weight)
        dist = self.get_action_dist(action_mean)
        action_logprobs = dist.log_prob(action) + torch.log(self.step) * self.action_dim
        dist_entropy = dist.entropy()
        return action_logprobs, value, dist_entropy

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
        rewards = []
        discounted_reward = 0

        for reward, is_terminal, next_value in zip(reversed(self.buffer.rewards),
                                       reversed(self.buffer.is_terminals), 
                                       reversed(self.buffer.next_values)):
            if is_terminal:
                discounted_reward = next_value

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        old_states = torch.squeeze(torch.stack(
            self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(
            self.buffer.actions, dim=0)).detach().to(self.device)
        old_weights = torch.squeeze(torch.stack(
            self.buffer.weights, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(
            self.buffer.logprobs, dim=0)).detach().to(self.device)
        
        for _ in range(self.K_epochs):
            idx_gen = self.batch_idx_generator(self.buffer.__len__(), self.batch_size)
            
            for i in range(int(self.buffer.__len__() / self.batch_size)):
                idx = next(idx_gen)
                
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.evaluate(
                    old_states[idx], old_actions[idx], old_weights[idx])
                
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs[idx].detach())
                
                # Finding Surrogate Loss
                advantages = rewards[idx] - state_values.detach()
                #advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                    1+self.eps_clip) * advantages
                
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * \
                    loss_fn(rewards[idx], state_values) - 0.01 * dist_entropy
                
                # take gradient step
                self.policy_opt.zero_grad()
                loss.mean().backward()
                self.policy_opt.step()
        
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict()) 
        self.std_train = max(self.std_train * self.std_decay, 1e-4)
        # clear buffer
        self.buffer.clear()  
            
    def save(self, model_path):
        torch.save(self.policy_old.state_dict(), model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        self.policy.load_state_dict(model)
        self.policy_old.load_state_dict(model)


class DDPG:
    def __init__(self, args, action_dim, agent_name):
        #super(DDPG, self).__init__()
        self.args = args
        self.device = args.device
        self.action_dim = action_dim
        self.day_length = args.state_length
        #self.std = args.action_std_test if args.test or args.backtest else args.action_std_train
        #self.std_train = args.action_std_train
        #self.std_decay = args.action_std_decay_rate
        #self.K_epochs = args.K_epochs
        #self.eps_clip = args.eps_clip
        self.gamma = args.gamma
        #self.dist_entropy_coef = args.dist_entropy_coef
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.policy = ActorCritic(args, self.day_length, self.action_dim, agent_name).to(self.device)
        self.target_policy = ActorCritic(
            args, self.day_length, self.action_dim, agent_name).to(self.device)
        
        self.actor_opt = optim.Adam(self.policy.actor.parameters(), self.args.lra)
        self.critic_opt = optim.Adam([{'params': self.policy.critic.parameters()},
                                      {'params': self.policy.critic_fc.parameters()}], self.args.lrv)
        
        self.target_policy.load_state_dict(self.policy.state_dict())
        
        self.relu = nn.ReLU()
        self.Gau_var = args.Gau_var
        self.Gau_decay = args.Gau_decay
        #self.step = torch.FloatTensor([STEP]).to(self.device)
        self.buffer = RolloutBuffer()
    
    def choose_action(self, state, weight):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.device)
            weight = torch.FloatTensor(weight[np.newaxis, :]).to(self.device)     
            value, action = self.policy(state, weight)
            self.action_noise = GaussianNoise(self.args, dim=len(weight), mu=None, std=self.Gau_var)
            noise = self.action_noise.sample()
            noise = torch.FloatTensor(noise).to(self.device)
            use_action = torch.abs(action + noise).cpu().numpy().flatten() + EPS
            use_action /= np.sum(use_action)
            
            return use_action, 0, 0, 0 
    
    def update(self):
        self.update_behavior_network()
        self.update_target_network()
    
    
    def update_behavior_network(self):
        loss_fn = nn.MSELoss()
        
        idx = GDP_idx(0, self.capacity, bias=0.02) 
        
        
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        old_states = torch.squeeze(torch.stack(
            self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(
            self.buffer.actions, dim=0)).detach().to(self.device)
        old_weights = torch.squeeze(torch.stack(
            self.buffer.weights, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(
            self.buffer.logprobs, dim=0)).detach().to(self.device)
        
        for _ in range(self.K_epochs):
            idx_gen = self.batch_idx_generator(self.buffer.__len__(), self.batch_size)
            
            for i in range(int(self.buffer.__len__() / self.batch_size)):
                idx = next(idx_gen)
                
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.evaluate(
                    old_states[idx], old_actions[idx], old_weights[idx])
                
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs[idx].detach())
                
                # Finding Surrogate Loss
                advantages = rewards[idx] - state_values.detach()
                #advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                    1+self.eps_clip) * advantages
                
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * \
                    loss_fn(rewards[idx], state_values) - 0.01 * dist_entropy
                
                # take gradient step
                self.policy_opt.zero_grad()
                loss.mean().backward()
                self.policy_opt.step()
        
        # Copy new weights into old policy
        #self.policy_old.load_state_dict(self.policy.state_dict())
        self.soft_update(self.policy_old, self.policy, self.tau)
        
        #self.std_train = max(self.std_train * self.std_decay, 1e-4)
        # clear buffer
        self.buffer.clear()  
        
    @staticmethod
    def soft_update(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_(tau * behavior.data + (1-tau) * target.data)
            
    def save(self, model_path):
        torch.save(self.policy_old.state_dict(), model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        self.policy.load_state_dict(model)
        self.policy_old.load_state_dict(model)
        
        

class Agent(nn.Module):
    def __init__(self, args, action_dim, agent_name):
        super(Agent, self).__init__()
        setup_seed(args.seed)
        self.args = args
        if args.algo == 'PPO':
            self.algo = PPO(args, action_dim, agent_name)
        elif args.algo == 'DDPG':
            self.algo = DDPG(args, action_dim, agent_name)
        
        #recorder
        self.train_reward = []
        self.train_value = []
        self.val_reward = []
        self.val_value = []
        self.train_acc = []
        self.val_acc = []
        #recorder

    def setup_seed_(self, seed):
        setup_seed(seed)

    def append(self, state, state_, next_value, weight, action, log_prob, reward, done):
        self.algo.buffer.states.append(torch.tensor(state, dtype=torch.float))
        self.algo.buffer.states_.append(torch.tensor(state_, dtype=torch.float))
        self.algo.buffer.next_values.append(torch.tensor(next_value, dtype=torch.float))
        self.algo.buffer.weights.append(torch.tensor(weight, dtype=torch.float))
        self.algo.buffer.actions.append(torch.tensor(action, dtype=torch.float))
        self.algo.buffer.logprobs.append(torch.tensor(log_prob, dtype=torch.float))
        self.algo.buffer.rewards.append(reward)
        self.algo.buffer.is_terminals.append(int(done))
        
    def update(self):
        self.algo.update()
    
    def save(self, model_path):
        self.algo.save(model_path)
    
    def load(self, model_path):
        self.algo.load(model_path)
    