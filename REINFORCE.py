from torch.distributions import Categorical
import gym
from network import mlp
from torch.optim import Adam
import torch.nn as nn
import torch
import numpy as np

import time

def train(env_name='CartPole-v0', gamma=0.9, hidden_node=[32], activations=[nn.Tanh, nn.Identity], batch_size=5000, epochs=1000, optimizer=Adam, lr=1e-2, render=False):
    
    env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    logits_net = mlp([obs_dim]+hidden_node+[act_dim], [nn.Tanh, nn.Identity])

    optimizer = optimizer(logits_net.parameters(), lr=lr)

    def policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    def get_action(obs):
        return policy(obs).sample().item()

    def get_loss(obs, act, weight):
        logp = policy(obs).log_prob(act)

        # simplest policy gradient
        # - for gradient ascent        
        return -(logp * weight).mean()

    def train_one_epoch():
        batch_ret = []
        batch_len = []
        batch_obs = []
        batch_act = []
       
        # for caulculating policy gradient 
        batch_weights = []  # Gt

        eps_rwd = []
        obs = env.reset()

        while True:
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            batch_obs.append(obs)
            
            obs, _, done, _ = env.step(act)
            if not done: rwd = 1

            batch_act.append(act) 
            eps_rwd.append(rwd)

            # if done:
            if not (abs(obs[2]) < np.pi / 3 and abs(obs[0]) <= 2.4):
                eps_len = len(eps_rwd)
                eps_goal = eps_rwd.copy()
                
                for i in range(1, eps_len):
                    eps_goal[-(i+1)] += eps_goal[-i] * gamma

                eps_ret = eps_goal[0]

                gamma_weight = 1
                for i in eps_goal:
                    i *= gamma_weight
                    gamma_weight *= gamma

                batch_weights += eps_goal

                obs = env.reset()
                eps_rwd = []

                batch_ret.append(eps_ret)
                batch_len.append(eps_len)

                if len(batch_obs) > batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = get_loss(torch.as_tensor(batch_obs, dtype=torch.float32),
                               torch.as_tensor(batch_act, dtype=torch.int32),
                               torch.as_tensor(batch_weights, dtype=torch.float32)
                               )

        batch_loss.backward()
        optimizer.step()

        return batch_ret, batch_len, batch_loss

    for i in range(epochs):
        batch_ret, batch_len, batch_loss = train_one_epoch()
        print(f'epoch: {i: 3d} \t loss: {batch_loss: .3f} \
             \t ret: {np.mean(batch_ret): .3f} {np.std(batch_ret): .3f} {np.min(batch_ret): .3f} {np.max(batch_ret): .3f} \
             \t len: {np.mean(batch_len): .3f} {np.std(batch_len): .3f} {np.min(batch_len): .3f} {np.max(batch_len): .3f}')

        if render:

            obs = env.reset()
            done = False


            try:
                while True:
                    env.render()
                    act = get_action(torch.as_tensor(obs, dtype=torch.float32))
                    obs, rwd, done, _ = env.step(act)

                    # if done:
                    if not (abs(obs[2]) < np.pi / 3 and abs(obs[0]) <= 2.4):
                        break

            except KeyboardInterrupt:
                print('stop')

if __name__=='__main__':
    env_name = 'CartPole-v0'
    train(env_name=env_name, gamma=1)