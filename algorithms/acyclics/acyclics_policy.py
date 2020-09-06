import numpy as np
import torch

from ray.rllib.policy import Policy
import numpy as np

from algorithms.acyclics.method import Method
from algorithms.acyclics.traj_buffer import TrajBuffer
from algorithms.acyclics.colorbin import visualize_color_bin


class AcyclicsPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.method = Method()

        self.episode_length = episode_length = config['rollout_fragment_length']
        self.n_envs = n_envs = config['num_workers'] * config['num_envs_per_worker']
        MAX_BUFFER_SIZE = 100000

        self.buffer = TrajBuffer(episode_length, n_envs, MAX_BUFFER_SIZE)
    
    def augment_og_obs(self, obs):
        empty_obs = np.zeros([self.n_envs, 64, 64, 3])
        for idx in range(self.n_envs):
            empty_obs[idx] = visualize_color_bin(obs[idx])
        empty_obs = np.transpose(empty_obs, [0, 3, 1, 2])
        empty_obs = empty_obs / 255.0
        return empty_obs
    
    def preprocess_states(self, states):
        states = self.method.vqvae.encode(states)
        states = torch.reshape(states, (states.size(0), 256))
        return states

    def compute_actions(
        self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        """Return the action for a batch
        Returns:
            action_batch: List of actions for the batch
            rnn_states: List of RNN states if any
            info: Additional info
        """
        action_batch = []
        rnn_states = []
        info = {}
        obs_batch = self.augment_og_obs(obs_batch)
        with torch.no_grad():
            agent_observation = torch.from_numpy(obs_batch).float().to('cuda')
            agent_observation = self.preprocess_states(agent_observation)
            action_batch, prob = self.method.mpo.target_actor.action(agent_observation)
            action_batch = np.reshape(action_batch.cpu().numpy(), -1)
            prob_batch = prob.cpu().numpy()
        info['probs'] = prob_batch
        return action_batch, rnn_states, info

    def learn_on_batch(self, samples):
        """Fused compute gradients and apply gradients call.
        Either this or the combination of compute/apply grads must be
        implemented by subclasses.
        Returns:
            grad_info: dictionary of extra metadata from compute_gradients().
        Examples:
            >>> batch = ev.sample()
            >>> ev.learn_on_batch(samples)
        Reference: https://github.com/ray-project/ray/blob/master/rllib/policy/policy.py#L279-L316
        """
        # implement your learning code here
        obs_b = samples['obs']
        action_b = samples['actions']
        reward_b = samples['rewards']
        prob_b = samples['probs']
        done_b = samples['dones']
        self.buffer.put(obs_b, action_b, reward_b, prob_b, done_b)
        self.method.train(self.buffer)
        return {}
    
    def get_weights(self):
        """Returns model weights.
        Returns:
            weights (obj): Serializable copy or view of model weights
        """
        data = self.method.mpo.get_weights()
        data['vae_state_dict'] = self.method.vqvae.state_dict()
        return {"w": data}

    def set_weights(self, weights):
        """Returns the current exploration information of this policy.
        This information depends on the policy's Exploration object.
        
        Returns:
            any: Serializable information on the `self.exploration` object.
        """
        data = weights["w"]
        self.method.mpo.load_weights(data)
        self.method.vqvae.load_state_dict(data['vae_state_dict'])
