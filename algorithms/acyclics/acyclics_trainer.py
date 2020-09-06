from ray.rllib.agents.trainer_template import build_trainer
from algorithms.acyclics.acyclics_policy import AcyclicsPolicy

DEFAULT_CONFIG = (
    {}
)  # Default config parameters that can be overriden by experiments YAML.

AcyclicsPolicyTrainer = build_trainer(
    name="AcyclicsPolicyTrainer",
    default_policy=AcyclicsPolicy,
    default_config=DEFAULT_CONFIG,
)
