import numpy as np
import torch

from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.utils.annotations import override

from algorithms.acyclics.method import Method


# yapf: disable
# __sphinx_doc_begin__
class Acyclics(Trainer):
    _name = "Acyclics"
    _default_config = with_common_config({})

    @override(Trainer)
    def _init(self, config, env_creator):
        self.env = env_creator(config["env_config"])
        self.method = Method(self.env, config["num_workers"] * config["num_envs_per_worker"])

    @override(Trainer)
    def _train(self):
        for _ in range(self.config["rollouts_per_iteration"]):
            episode_reward_mean, steps = self.method.train()
        return {
            "episode_reward_mean": episode_reward_mean,
            "timesteps_this_iter": steps,
        }
# __sphinx_doc_end__
# don't enable yapf after, it's buggy here


if __name__ == "__main__":
    trainer = Acyclics()
    result = trainer.train()
