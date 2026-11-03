"""Prepare and run the 25 historical CA baselines using their archived MLMs."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path("outputs/ca_seed_trials")
DATA = Path("data/ca_seed_trials")
METHODS = {
    "BPE": "bpe",
    "SBPE": "bpe",
    "SPCATT": "pcatt",
    "SMIRK": "smirk",
    "SMIRK-GPE": "smirk",
    "SMIRK-SGPE": "smirk",
    "SMIRK-SPCATT": "smirk_pcatt",
    "APE": "ape",
    "SPE": "spe",
}


def conditions():
    for family in ("SMILY", "SAFE", "TSMILES"):
        methods = (
            METHODS
            if family != "TSMILES"
            else [v + suffix for v in ("TSSA", "TSDY", "TSID", "TSIS") for suffix in ("", "-M")]
        )
        for method in methods:
            if (family, method) == ("SMILY", "SMIRK-GPE"):
                continue
            key = f"{family}-{method}" + ("" if method in ("SMIRK", "APE") else "-5K")
            model = Path(f"data/pretrained_models/mlm/CANONICAL-{key}-1M-20Epoch")
            tok = Path("data/pretrained_tokenizers") / ("CANONICAL_" + key.replace("-", "_"))
            algo = METHODS.get(method, "tsmiles")
            if method == "APE":
                tok = Path("data/ape_tokenizer/mikemayuare/vocab.json")
            elif method == "SMIRK":
                tok = Path("data/pretrained_tokenizers/SMILY_SMIRK/tokenizer.json")
            elif algo == "smirk":
                tok /= "tokenizer.json"
            for task in ("hiv", "bbbp"):
                args = [
                    "experiment=downstream_classification",
                    f"dataset=molnet_{task}",
                    f"algorithm={algo}_classifier",
                    f"algorithm.model.pretrained={model}",
                    f"algorithm.tokenizer.pretrained={tok}",
                    "dataset.seed=42",
                    f"dataset.use_safe={str(family == 'SAFE').lower()}",
                    f"dataset.use_tsmiles={str(family == 'TSMILES').lower()}",
                ]
                if family == "TSMILES":
                    args += [
                        f"dataset.tsmiles_variant={method.split('-')[0]}",
                        "dataset.tsmiles_slicer=" + ("mmpa" if method.endswith("-M") else "brics"),
                    ]
                yield key, task, args, model, tok


def trials():
    return [(row, seed) for row in conditions() for seed in (42, 43, 44)]


def prepare():
    from hydra import compose, initialize_config_dir
    from algorithms.downstream_finetune.algo import MoleculeClassificationAlgo
    from datamodules.molecule_datasets.molnet import load_molnet
    from datamodules.molecule_datasets.prepared import load_prepared
    from utils.external_tsdy_tokenizer import tokenizer_identity
    from utils.cache_utils import file_digest
    from omegaconf import OmegaConf

    DATA.mkdir(parents=True, exist_ok=True)
    for key, task, args, model, tok in conditions():
        dest = DATA / f"{key}_{task}"
        cache = DATA / "corpus" / task
        cache.mkdir(parents=True, exist_ok=True)
        src = Path(f"data/shared_study_2m_v2/base_TSDY_{task}")
        for file in (
            f"{task}.csv",
            f"{task}_split_indices.json",
            f"{task}_split_indices.json.meta.json",
        ):
            if not (cache / file).exists():
                shutil.copyfile(src / file, cache / file)
            if file_digest(src / file) != file_digest(cache / file):
                raise ValueError(f"Frozen inputs differ: {file}")
        with initialize_config_dir(config_dir=str(Path("configurations").resolve()), version_base=None):
            cfg = compose(config_name="config", overrides=args + [f"dataset.data_dir={cache}"])
        tokenizer = MoleculeClassificationAlgo(cfg.algorithm).tokenizer
        if len(tokenizer) != json.loads((model / "config.json").read_text())["vocab_size"]:
            raise ValueError(f"Tokenizer and MLM vocabulary sizes differ for {key}")
        if dest.exists():
            load_prepared(
                OmegaConf.create(dict(prepared_path=str(dest), prepared_task=task, max_length=512)),
                tokenizer,
            )
            continue
        if not (model / "model.safetensors").is_file():
            raise FileNotFoundError(model)
        splits = load_molnet(cfg.dataset, tokenizer)
        with tempfile.TemporaryDirectory(dir=DATA) as tmp:
            stage = Path(tmp) / "export"
            splits.save_to_disk(str(stage / "dataset"))
            manifest = dict(
                key=key,
                task=task,
                overrides=args,
                model=str(model),
                tokenizer=str(tok),
                split_hash=file_digest(src / f"{task}_split_indices.json"),
            )
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
            ready = dict(
                passed=True,
                tokenizer=tokenizer_identity(tokenizer),
                task=task,
                max_length=512,
                smoke=False,
                split_sizes={k: len(v) for k, v in splits.items()},
                files={str(p.relative_to(stage)): file_digest(p) for p in stage.rglob("*") if p.is_file()},
            )
            (stage / "ready.json").write_text(json.dumps(ready, indent=2))
            stage.rename(dest)


def run(index):
    (key, task, args, _, _), seed = trials()[index]
    dest = DATA / f"{key}_{task}"
    name = f"CA-V2-{key}-{task}-SEED{seed}"
    root = ROOT / "trials" / f"{key}_{task}" / f"seed{seed}"
    if (root / "complete.json").exists():
        return
    output = root / os.environ["SLURM_JOB_ID"] / "run"
    output.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        WANDB_TAGS=f"ca_seed_trials,{key.split('-')[0]},{task},seed{seed}",
        WANDB_LOG_MODEL="false",
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "main",
            f"+name={name}",
            *args,
            f"+dataset.prepared_path={dest}",
            f"+dataset.prepared_task={task}",
            f"experiment.training.seed={seed}",
            f"experiment.training.data_seed={seed}",
            "wandb.mode=online",
            "wandb.entity=ncduy0303",
            "wandb.project=molecule-tokenization",
            f"hydra.run.dir={output}",
        ],
        check=True,
        env=env,
    )
    metric = json.loads((output / "test_results.json").read_text())["test_roc_auc"]
    if not math.isfinite(metric) or not 0 <= metric <= 1:
        raise ValueError(metric)
    receipt = root / "complete.tmp"
    receipt.write_text(
        json.dumps(
            dict(
                name=name,
                seed=seed,
                task=task,
                key=key,
                test_roc_auc=metric,
                output=str(output),
            )
        )
    )
    receipt.replace(root / "complete.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "prepare", "run"])
    parser.add_argument("index", type=int, nargs="?")
    opt = parser.parse_args()
    if opt.action == "prepare":
        prepare()
    elif opt.action == "run":
        if opt.index not in range(len(trials())):
            parser.error("Invalid trial index")
        run(opt.index)
    else:
        for i, ((key, task, _, model, tok), seed) in enumerate(trials()):
            if not tok.exists() or not (model / "model.safetensors").is_file():
                sys.exit(f"Missing inputs: {model}, {tok}")
            print(i, key, task, seed)
