import numpy as np
import random


class TrajBuffer(object):
    def __init__(self, env, n_steps, n_env, obs_shape, size=50000, raw_pixels=False):
        """
        A buffer for observations, actions, rewards, mu's, states, masks and dones values
        :param env: (Gym environment) The environment to learn from
        :param n_steps: (int) The number of steps to run for each environment
        :param size: (int) The buffer size in number of steps
        """
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = self.n_env * self.n_steps
        # Each loc contains n_env * n_steps frames, thus total buffer is n_env * size frames
        self.size = size // self.n_steps
       
        # Memory
        self.enc_obs = None
        self.actions = None
        self.rewards = None
        self.mus = None
        self.dones = None

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0

    def has_atleast(self, frames):
        """
        Check to see if the buffer has at least the asked number of frames
        :param frames: (int) The number of frames checked
        :return: (bool) number of frames in buffer >= number asked
        """
        # Frames per env, so total (n_env * frames) Frames needed
        # Each buffer loc has n_env * n_steps frames
        return self.num_in_buffer >= (frames // self.n_steps)

    def can_sample(self):
        """
        Check if the buffer has at least one frame
        :return: (bool) if the buffer has at least one frame
        """
        return self.num_in_buffer > 0

    def put(self, enc_obs, actions, rewards, mus, dones):
        """
        Adds a frame to the buffer
        :param enc_obs: ([float]) the encoded observation
        :param actions: ([float]) the actions
        :param rewards: ([float]) the rewards
        :param mus: ([float]) the policy probability for the actions
        :param dones: ([bool])
        :param masks: ([bool])
        """
        # enc_obs [n_env, (n_steps + n_stack), nh, nw, nc]
        # actions, rewards, dones [n_env, n_steps]
        # mus [n_env, n_steps, n_act]

        if self.enc_obs is None:
            self.enc_obs = np.empty([self.size] + list(enc_obs.shape), dtype=np.float32)
            self.actions = np.empty([self.size] + list(actions.shape), dtype=np.int32)
            self.rewards = np.empty([self.size] + list(rewards.shape), dtype=np.float32)
            self.mus = np.empty([self.size] + list(mus.shape), dtype=np.float32)
            self.dones = np.empty([self.size] + list(dones.shape), dtype=np.bool)

        self.enc_obs[self.next_idx] = enc_obs
        self.actions[self.next_idx] = actions
        self.rewards[self.next_idx] = rewards
        self.mus[self.next_idx] = mus
        self.dones[self.next_idx] = dones

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def take(self, arr, idx, envx):
        """
        Reads a frame from a list and index for the asked environment ids
        :param arr: (np.ndarray) the array that is read
        :param idx: ([int]) the idx that are read
        :param envx: ([int]) the idx for the environments
        :return: ([float]) the askes frames from the list
        """
        n_env = self.n_env
        out = np.empty(arr.shape[1:], dtype=arr.dtype)
        for i in range(n_env):
            out[:, i] = arr[idx[i], :, envx[i]]
        return out

    def get(self):
        """
        randomly read a frame from the buffer
        :return: ([float], [float], [float], [float], [bool], [float])
                 observations, actions, rewards, mus, dones, maskes
        """
        # returns
        # obs [n_env, (n_steps + 1), nh, nw, n_stack*nc]
        # actions, rewards, dones [n_env, n_steps]
        # mus [n_env, n_steps, n_act]
        n_env = self.n_env
        assert self.can_sample()
        
        # Sample exactly one id per env. If you sample across envs, then higher correlation in samples from same env.
        idx = np.random.randint(0, self.num_in_buffer, n_env)
        envx = np.arange(n_env)

        dones = self.take(self.dones, idx, envx)
        obs = self.take(self.enc_obs, idx, envx)
        actions = self.take(self.actions, idx, envx)
        rewards = self.take(self.rewards, idx, envx)
        mus = self.take(self.mus, idx, envx)
        return obs, actions, rewards, mus, dones
