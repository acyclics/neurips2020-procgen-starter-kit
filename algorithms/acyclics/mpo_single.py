import time
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from scipy.optimize import minimize


class MPO(object):
    def __init__(self, actor, critic, vae, dual_constraint=0.1, kl_constraint=0.01,
                 learning_rate=0.99, alpha=10.0, episode_length=3000,
                 lagrange_it=5, device='cuda',
                 save_path="./mpo/model/mpo"):
        # initialize some hyperparameters
        self.α = alpha  # scaling factor for the update step of η_μ
        self.ε = dual_constraint  # hard constraint for the KL
        self.ε_kl = kl_constraint
        self.γ = learning_rate  # learning rate
        self.episode_length = episode_length
        self.lagrange_it = lagrange_it

        self.device = device
        
        # initialize networks and optimizer
        self.obs_shape = 256
        self.action_shape = 16

        self.critic = critic
        self.target_critic = deepcopy(critic)
        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.actor = actor
        self.target_actor = deepcopy(actor)
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.vae = vae
        self.target_vae = deepcopy(vae)
        for target_param, param in zip(self.target_vae.parameters(),
                                       self.vae.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False

        #self.mse_loss = nn.MSELoss()
        self.norm_loss_q = nn.SmoothL1Loss()

        # initialize Lagrange Multiplier
        self.η = np.random.rand()
        self.η_kl = 0.0

        # control/log variables
        self.save_path = save_path
    
    def _update_critic_retrace(self, state_batch, action_batch, policies_batch, reward_batch, done_batch):
        action_size = policies_batch.shape[-1]
        nsteps = state_batch.shape[0]
        n_envs = state_batch.shape[1]

        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            policies, a_log_prob, entropy = self.actor.evaluate_action(state_batch.view(-1, self.obs_shape), action_batch.view(-1, 1))
            target_policies, _, _ = self.target_actor.evaluate_action(state_batch.view(-1, self.obs_shape), action_batch.view(-1, 1))

        qval = self.critic(state_batch.view(-1, self.obs_shape))
        val = (qval * policies).sum(1, keepdim=True)

        old_policies = policies_batch.view(-1, action_size)
        policies = policies.view(-1, action_size)
        target_policies = target_policies.view(-1, action_size)

        val = val.view(-1, 1)
        qval = qval.view(-1, action_size)
        a_log_prob = a_log_prob.view(-1, 1)
        actions = action_batch.view(-1, 1)

        q_i = qval.gather(1, actions.long())
        rho = policies / (old_policies + 1e-10)
        rho_i = rho.gather(1, actions.long())

        with torch.no_grad():
            next_qval = self.critic(state_batch[-1]).detach()
            policies, a_log_prob, entropy = self.actor.evaluate_action(state_batch[-1], action_batch[-1])
            next_val = (next_qval * policies).sum(1, keepdim=True)
        
        q_retraces = reward_batch.new(nsteps + 1, n_envs, 1).zero_()
        q_retraces[-1] = next_val

        for step in reversed(range(nsteps)):
            q_ret = reward_batch[step] + self.γ * q_retraces[step + 1] * (1 - done_batch[step + 1])
            q_retraces[step] = q_ret
            q_ret = (rho_i[step] * (q_retraces[step] - q_i[step])) + val[step]
        
        q_retraces = q_retraces[:-1]
        q_retraces = q_retraces.view(-1, 1)

        q_loss = (q_i - q_retraces.detach()).pow(2).mean() * 0.5
        q_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        return q_loss.detach()

    def _categorical_kl(self, p1, p2):
        """
        calculates KL between two Categorical distributions
        :param p1: (B, D)
        :param p2: (B, D)
        """
        p1 = torch.clamp_min(p1, 0.0001)
        p2 = torch.clamp_min(p2, 0.0001)
        return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))

    def _update_param(self):
        """
        Sets target parameters to trained parameter
        """
        # Update policy parameters
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # Update critic parameters
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Update vae parameters
        for target_param, param in zip(self.target_vae.parameters(), self.vae.parameters()):
            target_param.data.copy_(param.data)

    def train(self, state_batch, action_batch, reward_batch, policies_batch, done_batch):
        episode_length = state_batch.shape[0]
        n_envs = state_batch.shape[1]
        mb_size = (episode_length-1) * n_envs

        state_batch = state_batch[0:-1]
        action_batch = action_batch[0:-1]
        reward_batch = reward_batch[0:-1]
        policies_batch = policies_batch[0:-1]
        done_batch = done_batch[0:]

        reward_batch = torch.unsqueeze(reward_batch, dim=-1)
        done_batch = torch.unsqueeze(done_batch, dim=-1)
        
        # Update Q-function
        with torch.no_grad():
            qstate_batch = state_batch.reshape(-1, 3, 64, 64)
            qstate_batch = self.vae.encode(qstate_batch)
            qstate_batch = torch.reshape(qstate_batch, (state_batch.size(0), state_batch.size(1), 256))

        q_loss = self._update_critic_retrace(qstate_batch, action_batch, policies_batch, reward_batch, done_batch)

        # Sample values
        state_batch = state_batch.view(mb_size, *tuple(state_batch.shape[2:]))
        action_batch = action_batch.view(mb_size, *tuple(action_batch.shape[2:]))
        
        with torch.no_grad():
            actions = torch.arange(self.action_shape)[..., None].expand(self.action_shape, mb_size).to(self.device)
            target_state_batch = self.target_vae.encode(state_batch)
            target_state_batch = torch.reshape(target_state_batch, (target_state_batch.size(0), self.obs_shape))
            b_p = self.target_actor.forward(target_state_batch)
            b = Categorical(probs=b_p)
            b_prob = b.expand((self.action_shape, mb_size)).log_prob(actions).exp()
            target_q = self.target_critic.forward(target_state_batch)
            target_q = target_q.transpose(0, 1)
            b_prob_np = b_prob.cpu().numpy() 
            target_q_np = target_q.cpu().numpy()
        
        with torch.no_grad():
            state_batch = self.vae.encode(state_batch)
            state_batch = torch.reshape(state_batch, (state_batch.size(0), 256))

        # E-step
        # Update Dual-function
        def dual(η):
            """
            dual function of the non-parametric variational
            g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
            """
            max_q = np.max(target_q_np, 0)
            return η * self.ε + np.mean(max_q) \
                + η * np.mean(np.log(np.sum(b_prob_np * np.exp((target_q_np - max_q) / η), axis=0)))

        bounds = [(1e-6, None)]
        res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
        self.η = res.x[0]

        # calculate the new q values
        qij = torch.softmax(target_q / self.η, dim=0)

        # M-step
        # update policy based on lagrangian
        for _ in range(self.lagrange_it):
            π_p = self.actor.forward(state_batch)
            π = Categorical(probs=π_p)
            loss_p = torch.mean(
                qij * π.expand((self.action_shape, mb_size)).log_prob(actions)
            )
        
            kl = self._categorical_kl(p1=π_p, p2=b_p)

            # Update lagrange multipliers by gradient descent
            self.η_kl -= self.α * (self.ε_kl - kl).detach().item()

            if self.η_kl < 0.0:
                self.η_kl = 0.0

            self.actor_optimizer.zero_grad()
            loss_policy = -(loss_p + self.η_kl * (self.ε_kl - kl))
            loss_policy.backward()
            clip_grad_norm_(self.actor.parameters(), 5.0)
            self.actor_optimizer.step()
        
        return q_loss.item(), loss_policy.item(), self.η, self.η_kl
            
    def load_weights(self, data):
        self.critic.load_state_dict(data['critic_state_dict'])
        self.target_critic.load_state_dict(data['target_critic_state_dict'])
        self.actor.load_state_dict(data['actor_state_dict'])
        self.target_actor.load_state_dict(data['target_actor_state_dict'])
        self.critic_optimizer.load_state_dict(data['critic_optim_state_dict'])
        self.actor_optimizer.load_state_dict(data['actor_optim_state_dict'])
        self.η = data['lagrange_η']
        self.η_kl = data['lagrange_η_kl']
        #self.critic.train()
        #self.target_critic.train()
        #self.actor.train()
        #self.target_actor.train()

    def get_weights(self):
        data = {
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'lagrange_η': self.η,
            'lagrange_η_kl': self.η_kl
        }
        return data
