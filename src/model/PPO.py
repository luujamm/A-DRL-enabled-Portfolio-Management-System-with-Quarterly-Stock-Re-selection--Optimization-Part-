import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions import MultivariateNormal
from .base import StochasticRL, ActorCritic


EPS = 1e-8


class PPO(StochasticRL):
    def __init__(self, args, action_dim):
        super().__init__(args, action_dim)
        self.K_epochs = args.K_epochs
        self.eps_clip = args.eps_clip
        self.dist_entropy_coef = args.dist_entropy_coef
        self.buffer = RolloutBuffer()
        
        self.policy = ActorCritic(args, self.state_length, self.action_dim).to(self.device)
        self.policy_opt = optim.Adam([
            {'params': self.policy.CNN.parameters()},
            {'params': self.policy.critic.parameters(), 'lr': self.args.lrv},],
            lr=self.args.lra)
        self.policy_old = ActorCritic(
            args, self.state_length, self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

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

    def append(self, state, next_value, weight, action, log_prob, reward, done):
        self.basic_append(state, action, weight, reward, done)
        self.buffer.next_values.append(torch.tensor(next_value, dtype=torch.float))
        self.buffer.logprobs.append(torch.tensor(log_prob, dtype=torch.float))
        
    def update(self):
        loss_fn = nn.MSELoss()
        rewards = []
        discounted_reward = 0
        total_loss = 0

        for reward, is_terminal, next_value in zip(reversed(self.buffer.rewards),
                                       reversed(self.buffer.is_terminals), 
                                       reversed(self.buffer.next_values)):
            
            if is_terminal:
                discounted_reward = next_value

            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        old_states = self.change_form(self.buffer.states)
        old_actions = self.change_form(self.buffer.actions)
        old_weights = self.change_form(self.buffer.weights)
        old_logprobs = self.change_form(self.buffer.logprobs)
        
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
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                    1 + self.eps_clip) * advantages
                
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * \
                    loss_fn(rewards[idx], state_values) - 0.01 * dist_entropy
                total_loss += loss_fn(rewards[idx], state_values).sum().item()
                
                # take gradient step
                self.policy_opt.zero_grad()
                loss.mean().backward()
                self.policy_opt.step()

        self.train_loss.append(total_loss)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # std decay
        self.std_train = max(self.std_train * self.std_decay, 1e-4)

        # clear buffer
        self.buffer.clear()
        
    def save(self, model_path):
        torch.save(self.policy_old.state_dict(), model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        self.policy.load_state_dict(model)
        self.policy_old.load_state_dict(model)