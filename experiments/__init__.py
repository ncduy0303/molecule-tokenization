from typing import Optional, Union
from omegaconf import DictConfig
import pathlib

from .exp_base import BaseExperiment
from importlib import import_module

exp_registry = {
    "tokenizer_benchmark": "TokenizerBenchmarkExperiment",
    "tokenizer_training": "TokenizerTrainingExperiment",
    "downstream_classification": "DownstreamClassificationExperiment",
}


def build_experiment(
    cfg: DictConfig, output_dir: Union[str, pathlib.Path], ckpt_path: Optional[Union[str, pathlib.Path]] = None
) -> BaseExperiment:
    """
    Build an experiment instance based on registry
    :param cfg: configuration file
    :param output_dir: Hydra output directory for this run
    :param ckpt_path: optional checkpoint path for saving and loading
    :return:
    """
    if cfg.experiment._name not in exp_registry:
        raise ValueError(
            f"Experiment {cfg.experiment._name} not found in registry {list(exp_registry.keys())}. "
            "Make sure you register it correctly in 'experiments/__init__.py' under the same name as yaml file."
        )

    name = cfg.experiment._name
    experiment = getattr(import_module(f".{name}", __name__), exp_registry[name])
    return experiment(cfg, output_dir, ckpt_path)
