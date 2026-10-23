"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.
"""

from abc import ABC
from typing import Optional, Union, Dict
from pathlib import Path

import torch
from omegaconf import DictConfig


from utils.print_utils import cyan
from utils.distributed_utils import is_rank_zero

torch.set_float32_matmul_precision("high")


class BaseExperiment(ABC):
    """
    Shared configuration, algorithm construction, and task dispatch for molecular experiments.
    """

    # each key has to be a yaml file under '[project_root]/configurations/algorithm' without .yaml suffix
    compatible_algorithms: Dict = NotImplementedError

    def __init__(
        self, root_cfg: DictConfig, output_dir: Optional[Union[str, Path]], ckpt_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Constructor

        Args:
            root_cfg: configuration file that contains root configuration from project_root/configurations/config.yaml
            output_dir: a directory to save outputs
            ckpt_path: an optional path to saved checkpoint
        """
        super().__init__()
        self.root_cfg = root_cfg
        self.output_dir = Path(output_dir)
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None

        self.cfg = root_cfg.experiment
        self.debug = root_cfg.debug

        # some tasks doesn't need logger or algo (e.g. download dataset) so leave for None for now
        self.logger = None
        self.algo = None

    def _build_algo(self):
        """
        Build the configured molecular algorithm
        :return: algorithm wrapper selected by the configuration
        """
        algo_name = self.root_cfg.algorithm._name
        if algo_name not in self.compatible_algorithms:
            raise ValueError(
                f"Algorithm {algo_name} not found in compatible_algorithms for this Experiment class. "
                "Make sure you define compatible_algorithms correctly and make sure that each key has "
                "same name as yaml file under '[project_root]/configurations/algorithm' without .yaml suffix"
            )
        self.algo = self.compatible_algorithms[algo_name](self.root_cfg.algorithm)
        return self.algo

    def exec_task(self, task: str) -> None:
        """
        Executing a certain task specified by string. Each task should be a stage of experiment.
        Molecular experiments implement training and, where available, evaluation.

        Args:
            task: a string specifying a task implemented for this experiment
        """

        if hasattr(self, task) and callable(getattr(self, task)):
            if is_rank_zero:
                print(cyan("Executing task:"), f"{task} out of {self.cfg.tasks}")
            getattr(self, task)()
        else:
            raise ValueError(
                f"Specified task '{task}' not defined for class {self.__class__.__name__} or is not callable."
            )
