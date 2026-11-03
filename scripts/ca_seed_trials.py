"""Prepare and run the 25 historical CA baselines using their archived MLMs."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import runpy
import sys
import tempfile
from datetime import datetime

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
    sys.argv = [
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
    ]
    os.environ.update(env)
    print(datetime.now().isoformat(timespec="seconds"), "Starting Hydra", flush=True)
    runpy.run_module("main", run_name="__main__", alter_sys=True)
    record_completion(index, output)


def record_completion(index, output):
    (key, task, _, _, _), seed = trials()[index]
    name = f"CA-V2-{key}-{task}-SEED{seed}"
    root = output.parents[1]
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
    print(
        datetime.now().isoformat(timespec="seconds"),
        "Local results complete:",
        root,
        flush=True,
    )


def recover(index):
    """Recover only after full local export, confirmed in the matching Slurm log."""
    import re

    (key, task, _, _, _), seed = trials()[index]
    root = ROOT / "trials" / f"{key}_{task}" / f"seed{seed}"
    if (root / "complete.json").exists():
        print("Completion already recorded")
        return
    matches = []
    for log in Path("slurm_logs").glob(f"ca_trial_*_{index}.out"):
        text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text(errors="replace"))
        for path in re.findall(r"^Saved best model to: (.+)$", text, re.M):
            output = Path(path.strip()).parent
            if output.parents[1].resolve() != root.resolve():
                continue
            if all(
                (output / f).is_file()
                for f in (
                    "test_results.json",
                    "validation_results.json",
                    "prepared_input.json",
                    "best_model/model.safetensors",
                    "best_model/config.json",
                )
            ):
                matches.append(output)
    if len(matches) != 1:
        raise SystemExit(f"Expected one verified export, found {len(matches)}. No receipt created.")
    record_completion(index, matches[0])
    print("Recovered", matches[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "prepare", "run", "recover"])
    parser.add_argument("index", type=int, nargs="?")
    opt = parser.parse_args()
    if opt.action == "prepare":
        prepare()
    elif opt.action in ("run", "recover"):
        if opt.index not in range(len(trials())):
            parser.error("Invalid trial index")
        (run if opt.action == "run" else recover)(opt.index)
    else:
        for i, ((key, task, _, model, tok), seed) in enumerate(trials()):
            if not tok.exists() or not (model / "model.safetensors").is_file():
                sys.exit(f"Missing inputs: {model}, {tok}")
            print(i, key, task, seed)
