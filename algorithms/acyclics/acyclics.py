import numpy as np
import torch

from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.utils.annotations import override

from algorithms.acyclics.method import Method


class Acyclics(Trainer):
    _name = "Acyclics"
    _default_config = with_common_config({})

    @override(Trainer)
    def _init(self, config, env_creator):
        self.env = env_creator(config["env_config"])
        self.method = Method(self.env, )

    @override(Trainer)
    def _train(self):
        episode_reward_mean, steps = self.method.train()
        return {
            "episode_reward_mean": episode_reward_mean,
            "timesteps_this_iter": steps,
        }


if __name__ == "__main__":
    trainer = Acyclics()
    result = trainer.train()
