"""Tiny CPU training through the actual experiment classes. Not scientific results."""

import argparse
import json
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from experiments.tokenizer_benchmark import TokenizerBenchmarkExperiment
from experiments.downstream_classification import DownstreamClassificationExperiment
from datamodules.molecule_datasets.prepared import load_prepared


def smoke(data_root, tokenizer_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Smoke output exists; choose a fresh directory")
    torch.set_num_threads(1)
    config_dir = str(Path("configurations").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        mlm_cfg = compose(
            config_name="config",
            overrides=[
                "experiment=tokenizer_benchmark",
                "dataset=external_tsdy_mlm",
                "algorithm=external_tsdy_roberta",
                "wandb.mode=disabled",
            ],
        )
    with open_dict(mlm_cfg):
        mlm_cfg.name = "ENHANCED-TSDY-SMOKE-NOT-A-RESULT"
        mlm_cfg.algorithm._name = "external_tsdy_roberta"
        mlm_cfg.dataset._name = "external_tsdy_mlm"
        mlm_cfg.algorithm.tokenizer.pretrained = str(tokenizer_path)
        mlm_cfg.dataset.prepared_path = str(Path(data_root) / "enhanced_mlm")
        mlm_cfg.dataset.allow_smoke = True
        for key, value in dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2
        ).items():
            mlm_cfg.algorithm.model[key] = value
        t = mlm_cfg.experiment.training
        t.use_cpu = True
        t.num_train_epochs = 1
        t.per_device_train_batch_size = 4
        t.per_device_eval_batch_size = 4
        t.dataloader_num_workers = 0
        t.logging_steps = 1
        t.mlm_probability = 0.3

    class TinyMLM(TokenizerBenchmarkExperiment):
        def _load_dataset(self):
            data = super()._load_dataset()
            return type(data)({s: ds.select(range(min(8, len(ds)))) for s, ds in data.items()})

    (output / "mlm").mkdir(parents=True)
    TinyMLM(mlm_cfg, output / "mlm").training()
    final = output / "mlm/final_checkpoint"
    # The classifier uses the real load_prepared route. Limit it by a temporary
    # wrapper at the experiment import boundary, not by changing any saved data.
    import experiments.downstream_classification as classification
    from unittest.mock import patch

    def small_loader(cfg, tok):
        data = load_prepared(cfg, tok)
        return type(data)({s: ds.select(range(min(8, len(ds)))) for s, ds in data.items()})

    for task in ("hiv", "bbbp"):
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "experiment=downstream_classification",
                    f"dataset=external_tsdy_{task}",
                    "algorithm=external_tsdy_classifier",
                    "wandb.mode=disabled",
                ],
            )
        with open_dict(cfg):
            cfg.name = "ENHANCED-TSDY-SMOKE-NOT-A-RESULT"
            cfg.algorithm._name = "external_tsdy_classifier"
            cfg.algorithm.tokenizer.pretrained = str(tokenizer_path)
            cfg.algorithm.model.pretrained = str(final)
            cfg.dataset.prepared_path = str(Path(data_root) / f"enhanced_{task}")
            cfg.dataset.allow_smoke = True
            t = cfg.experiment.training
            t.use_cpu = True
            t.num_train_epochs = 1
            t.dataloader_num_workers = 0
            t.per_device_train_batch_size = 4
            t.per_device_eval_batch_size = 4
            t.logging_steps = 1
        (output / task).mkdir()
        with patch("datamodules.molecule_datasets.prepared.load_prepared", side_effect=small_loader):
            classification.DownstreamClassificationExperiment(cfg, output / task).training()
    checks = {}
    import math

    for name in ("mlm", "hiv", "bbbp"):
        path = output / name / "test_results.json"
        results = json.loads(path.read_text())
        if not math.isfinite(results.get("test_loss", results.get("eval_loss", float("nan")))):
            raise ValueError("Nonfinite smoke loss")
        checks[name] = dict(metrics_file=str(path), passed=True)
    (output / "smoke_passed.json").write_text(
        json.dumps(dict(passed=True, diagnostic_only=True, checks=checks), indent=2)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data/enhanced_tsdy_v3_1"))
    p.add_argument("--tokenizer", type=Path, default=Path("outputs/enhanced_tsdy_v3_1/tokenizer"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    smoke(a.data_root, a.tokenizer, a.output)


if __name__ == "__main__":
    main()
