# -*- coding: utf-8 -*-
# PPO通用代码
import sys
import numpy as np
import torch
# 导入torch的各种模块
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import gym
import warnings
warnings.filterwarnings('ignore')
import random
from collections import deque
import matplotlib.pyplot as plt
import copy
from IPython.display import clear_output
import pandas as pd
import pickle
#%matplotlib inline

# 在PPO算法中使用DQN中的replay buffer（使用(state, action, reward, next_state)训练，而不必等到回合结束）
# advantage_t = q(s_t,a_t) - mean(q(s_t,a_t)) t=1,...,batch_size 作为advantage

# avoid the warning message
gym.logger.set_level(40)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class DQNReplayer:
    def __init__(self, capacity=50000):    # capacity越大，训练得越慢
        self.memory = pd.DataFrame(index=range(capacity),
                columns=['state', 'action', 'reward', 'next_state', 'log_probs', 'terminated'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = np.asarray(args, dtype=object)
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in
                self.memory.columns)
 
        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
                )
        
        # critic
        self.critic_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim)
                )
        
        self.critic_target_layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim)
                )
        
        self.update_target()
        
        
    def forward(self):
        raise NotImplementedError
        
    def update_target(self):
        hard_update(self.critic_target_layer, self.critic_layer)
        
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()         
        return state,action,dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() 
        state_q = self.critic_layer(state)
        return action_logprobs, torch.squeeze(state_q), dist_entropy
        
class PPO:
    def __init__(self,state_dim, action_dim, hidden_dim, lr, gamma, eps_clip):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.policy.action_layer.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic_layer.parameters(), lr=lr)
        #self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MseLoss = nn.MSELoss()
        self.losses = []
        # 用作缓冲区
        self.memory = DQNReplayer()
        self.invalid_ratio = 0.2
        self.update_steps = 0
        self.entropy_weight = self.decay_entropy_weight(0)
        
    def step(self, observation):
        state, action, logprob = self.policy.select_action(observation)
        return action.item(), logprob.item()
    
    def decay_entropy_weight(self, episode, max_episodes=2000, init_entropy_weight=0.01, min_entropy_weight=0.0001):
        self.entropy_weight = init_entropy_weight * episode / max_episodes + min_entropy_weight
        
    def learn(self):   
        
        states, actions, rewards, next_states, logprobs, terminateds = \
                self.memory.sample(1024)
        
        # # convert list to tensor
        state_tensor = torch.as_tensor(states, dtype=torch.float)
        action_tensor = torch.as_tensor(actions, dtype=torch.long)
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float)
        next_state_tensor = torch.as_tensor(next_states, dtype=torch.float)
        logprob_tensor = torch.as_tensor(logprobs, dtype=torch.float)
        terminated_tensor = torch.as_tensor(terminateds, dtype=torch.float)
        
        ## update actor
        new_logprobs, state_qs, dist_entropy = self.policy.evaluate(state_tensor, action_tensor)
        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(new_logprobs - logprob_tensor.detach())
        invalid_data = (ratios < (1 - self.eps_clip)) + (ratios > (1 + self.eps_clip))
        self.invalid_ratio = 0.95 * self.invalid_ratio + 0.05 * invalid_data.sum() / ratios.numel()
        #print('不参与训练的数据的比重：{}'.format(invalid_data.sum() / ratios.numel()))
        # Finding Surrogate Loss:
        #advantages = rewards - state_qs.detach()
        q_tensor = state_qs.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        qs = q_tensor.detach()
        
        #########################################################################
        ############### 减均值这一步很关键 #######################################
        qs = qs - qs.mean()
        #########################################################################
        
        surr1 = ratios * qs
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * qs
        #loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_qs, rewards) - 0.01*dist_entropy
        loss = -torch.min(surr1, surr2)  - self.entropy_weight*dist_entropy
        self.losses.append(loss.mean())
        # take gradient step
        self.actor_optimizer.zero_grad()
        loss.mean().backward()
        self.actor_optimizer.step()
        
        # update value net
        with torch.no_grad():
            next_q_tensor = self.policy.critic_target_layer(next_state_tensor)
        next_max_q_tensor, _ = next_q_tensor.max(axis=-1)
        target_tensor = reward_tensor + self.gamma * \
                (1. - terminated_tensor) * next_max_q_tensor
        loss_tensor = self.MseLoss(target_tensor, q_tensor)
        self.critic_optimizer.zero_grad()
        loss_tensor.backward()
        self.critic_optimizer.step()
        
        self.update_steps += 1
        if self.update_steps % 100 == 0:
            self.policy.update_target()
        
            
    # 下面是训练网络   
    def train_network(self, env, env_name, epsiodes=500):
        epsiode_rewards = []
        mean_rewards = []
        for epsiode in range(1,epsiodes+1):
            state, _ = env.reset()
            ep_reward = 0
            done = False
            elapsed_steps = 1
            while True:
                action,log_prob = self.step(state)
                                
                next_state, reward, done, _, _ = env.step(action)
                
                self.memory.store(state, action, reward, next_state, log_prob, done)

                ep_reward += reward
                
                if (epsiode < 10):
                    #if (self.memory.count >= self.memory.capacity * 0.95) and (elapsed_steps % 50 == 0):
                    if (self.memory.count >= 50000) and (elapsed_steps % 50 == 0):
                        self.learn()
                else:
                    if elapsed_steps % 10 == 0:
                        self.learn()

                    
                state = next_state
                
                if done:
                    break
                
                # if elapsed_steps > 1000:
                #     break
                
                elapsed_steps += 1
    
            #self.decay_entropy_weight(epsiode)
            # logging
            epsiode_rewards.append(ep_reward)
            mean_rewards.append(torch.mean(torch.Tensor(epsiode_rewards[-30:])))
            print("第{}回合的奖励值: {:.2f}, 平均奖励: {:.2f}, 不参与训练的数据的比重：{}".format(
                epsiode,ep_reward, mean_rewards[-1], self.invalid_ratio))
            
            if epsiode % 50 == 0:
                with open('ppo_dqn_{}.pkl'.format(env_name), 'wb') as f:
                    pickle.dump(self.policy, f)
        return epsiode_rewards,mean_rewards
    
    def demo(self, env, env_name, epsiodes=100):
        with open('ppo_dqn_{}.pkl'.format(env_name), 'rb') as f:
            self.policy = pickle.load(f)
        epsiode_rewards = []
        mean_rewards = []
        for epsiode in range(1,epsiodes+1):
            state, _ = env.reset()
            ep_reward = 0
            elapsed_steps = 1
            while True:
                # Running policy_old:
                action, _ = self.step(state)
                                
                next_state, reward, done, _, _ = env.step(action)
                ep_reward += reward   
                state = next_state
                
                if done:
                    break
                
                elapsed_steps += 1
        
            # logging
            epsiode_rewards.append(ep_reward)
            mean_rewards.append(torch.mean(torch.Tensor(epsiode_rewards)))
            print("第{}回合的奖励值: {:.2f},平均奖励: {:.2f}".format(epsiode,ep_reward,mean_rewards[-1]))
            
        return epsiode_rewards,mean_rewards
        
        
if __name__ == '__main__': 
    ############## Hyperparameters ##############
    # creating environment
    env_name = "MountainCar-v0"    #"Acrobot-v1"    
    env = gym.make(env_name)
    env = env.unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episodes = 2000        # max training episodes
    max_timesteps = 5000         # max timesteps in one episode
    hidden_dim = 64           # number of variables in hidden layer
    update_timestep = 1000      # update policy every n timesteps
    lr = 0.002
    gamma = 0.99                # discount factor
    eps_clip = 0.2              # clip parameter for PPO
    #############################################            
    torch.manual_seed(2)
    #env.seed(2)
    ppo = PPO(state_dim, action_dim, hidden_dim, lr, gamma, eps_clip)
    epsiode_rewards,mean_rewards = ppo.train_network(env, env_name, epsiodes=2000)
    plt.plot(epsiode_rewards)
    plt.plot(mean_rewards)
    plt.xlabel("epsiode")
    plt.ylabel("rewards")
    plt.show()
    
    env = gym.make(env_name, env_name, render_mode='human')
    ppo.demo(env, epsiodes=20)

