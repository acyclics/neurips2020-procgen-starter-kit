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


class Method():

    def __init__(self, env, n_envs):
        self.env = env
        self.n_envs = n_envs
        self.total_steps = 0
    
        episode_length = 1000

        MAX_BUFFER_SIZE = 100000
        self.buffer = TrajBuffer(env, episode_length, n_envs, MAX_BUFFER_SIZE)

        self.vqvae = Vqvae().cuda()

        in_units = 256
        actor = CategoricalActor(in_units).cuda()
        critic = Critic(in_units).cuda()

        self.mpo = MPO(actor, critic, self.vqvae, episode_length=episode_length, n_envs=n_envs)

        self.VAE_ITRS = 100
        self.MPO_ITRS = 50
    
    def augment_og_obs(self, obs):
        empty_obs = np.zeros([self.n_envs, 64, 64, 3])
        for idx in range(n_envs):
            empty_obs[idx] = visualize_color_bin(obs[idx])
        empty_obs = np.transpose(empty_obs, [0, 3, 1, 2])
        empty_obs = empty_obs / 255.0
        return empty_obs
    
    def preprocess_states(self, states):
        states = self.vqvae.encode(states)
        #states = states.view(states.size(0), 256)
        states = torch.reshape(states, (states.size(0), 256))
        return states
    
    def sample_trajectory(self, episodes, episode_length, obs_function):
        mean_rewards = []

        for _ in range(episodes):
            _, observation, _ = self.env.observe()
            observation = observation['rgb']
            observation = obs_function(observation)

            obs_b = np.zeros([episode_length, self.n_envs, 3, 64, 64])
            action_b = np.zeros([episode_length, self.n_envs])
            reward_b = np.zeros([episode_length, self.n_envs])
            prob_b = np.zeros([episode_length, self.n_envs, 16])
            done_b = np.zeros([episode_length, self.n_envs])

            with torch.no_grad():
                agent_observation = torch.from_numpy(observation).float().to('cuda')
                agent_observation = self.preprocess_states(agent_observation)

            mean_reward = 0
            for steps in range(episode_length):
                with torch.no_grad():
                    action, prob = self.mpo.target_actor.action(agent_observation)
                    action = np.reshape(action.cpu().numpy(), -1)
                    prob = prob.cpu().numpy()

                self.env.act(action)
                self.total_steps += 1
                reward, new_observation, done = self.env.observe()

                new_observation = new_observation['rgb']
                new_observation = obs_function(new_observation)

                with torch.no_grad():
                    agent_observation = torch.from_numpy(new_observation).float().to('cuda')
                    agent_observation = self.preprocess_states(agent_observation)

                mean_reward += reward
                
                obs_b[steps] = observation
                action_b[steps] = action
                reward_b[steps] = reward
                prob_b[steps] = prob
                done_b[steps] = done

                observation = new_observation
            
            mean_rewards.append(mean_reward)
            self.buffer.put(obs_b, action_b, reward_b, prob_b, done_b)

        mean_rewards = np.mean(mean_rewards)
        return mean_rewards
    
    def train(self):
        
        mean_rewards = self.sample_trajectory(1, episode_length, augment_og_obs)

        for _ in range(self.VAE_ITRS):
            state_batch, action_batch, reward_batch, policies_batch, dones_batch = self.buffer.get()
            state_batch = state_batch.reshape(-1, 3, 64, 64)
            data_variance = np.var(state_batch)
            for idx in range(0, state_batch.shape[0], 256):
                state_mb = state_batch[idx:idx+256]
                state_mb = torch.from_numpy(state_mb).float().to('cuda')
                vae_loss = self.vqvae.train(state_mb, data_variance)

        for _ in range(MPO_ITRS):
            state_batch, action_batch, reward_batch, policies_batch, dones_batch = self.buffer.get()

            state_batch = torch.from_numpy(state_batch).float().to('cuda')
            action_batch = torch.from_numpy(action_batch).float().to('cuda')
            reward_batch = torch.from_numpy(reward_batch).float().to('cuda')
            policies_batch = torch.from_numpy(policies_batch).float().to('cuda')
            dones_batch = torch.from_numpy(dones_batch).float().to('cuda')

            q_loss, loss_policy, η, η_kl = self.mpo.train(state_batch, action_batch, reward_batch, policies_batch, dones_batch)
        
        self.mpo._update_param()
        
        return mean_rewards, self.total_steps
