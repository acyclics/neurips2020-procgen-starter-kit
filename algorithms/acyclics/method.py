import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import deepcopy
import time

from algorithms.acyclics.mpo_single import MPO
from algorithms.acyclics.traj_buffer import TrajBuffer
from algorithms.acyclics.vqvae_single import Vqvae
from algorithms.acyclics.colorbin import visualize_color_bin


class CategoricalActor(torch.nn.Module):

    def __init__(self, in_units):
        super(CategoricalActor, self).__init__()

        self.in_units = in_units

        self.layers = torch.nn.Sequential(
            nn.Linear(in_units, 512),
            torch.nn.Tanh(),
            nn.Linear(512, 256),
            torch.nn.Tanh(),
            nn.Linear(256, 128),
            torch.nn.Tanh(),
            nn.Linear(128, 128),
            torch.nn.Tanh(),
            nn.Linear(128, 128),
            torch.nn.Tanh()
        )

        self.logits = nn.Linear(128, 16)

    def forward(self, states):
        x = self.layers(states)
        x = self.logits(x)
        x = torch.softmax(x, dim=-1)
        return x
    
    def action(self, state):
        with torch.no_grad():
            probs = self.forward(state)
            action_distribution = Categorical(probs=probs)
            action = action_distribution.sample()
            prob = action_distribution.probs
        return action, prob
    
    def evaluate_action(self, state, action):
        probs = self.forward(state)
        action_distribution = Categorical(probs=probs)
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy().mean()
        return action_distribution.probs, log_prob, entropy
    

class Critic(torch.nn.Module):
    def __init__(self, in_units):
        super(Critic, self).__init__()

        self.in_units = in_units

        self.layers = torch.nn.Sequential(
            nn.Linear(in_units, 512),
            torch.nn.Tanh(),
            nn.Linear(512, 256),
            torch.nn.Tanh(),
            nn.Linear(256, 128),
            torch.nn.Tanh(),
            nn.Linear(128, 128),
            torch.nn.Tanh(),
            nn.Linear(128, 128),
            torch.nn.Tanh()
        )

        self.value = nn.Linear(128, 16)

    def forward(self, state):
        x = self.layers(state)
        x = self.value(x)
        return x


class Method(nn.Module):

    def __init__(self):
        self.vqvae = Vqvae().cuda()

        in_units = 256
        self.actor = CategoricalActor(in_units).cuda()
        self.critic = Critic(in_units).cuda()

        self.mpo = MPO(self.actor, self.critic, self.vqvae)

        self.VAE_ITRS = 100
        self.MPO_ITRS = 50
    
    def train(self, buffer):
        for _ in range(self.VAE_ITRS):
            state_batch, action_batch, reward_batch, policies_batch, dones_batch = buffer.get()
            state_batch = state_batch.reshape(-1, 3, 64, 64)
            data_variance = np.var(state_batch)
            for idx in range(0, state_batch.shape[0], 256):
                state_mb = state_batch[idx:idx+256]
                state_mb = torch.from_numpy(state_mb).float().to('cuda')
                vae_loss = self.vqvae.train(state_mb, data_variance)

        for _ in range(self.MPO_ITRS):
            state_batch, action_batch, reward_batch, policies_batch, dones_batch = buffer.get()

            state_batch = torch.from_numpy(state_batch).float().to('cuda')
            action_batch = torch.from_numpy(action_batch).float().to('cuda')
            reward_batch = torch.from_numpy(reward_batch).float().to('cuda')
            policies_batch = torch.from_numpy(policies_batch).float().to('cuda')
            dones_batch = torch.from_numpy(dones_batch).float().to('cuda')

            q_loss, loss_policy, η, η_kl = self.mpo.train(state_batch, action_batch, reward_batch, policies_batch, dones_batch)
        
        self.mpo._update_param()        
