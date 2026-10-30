"""Isolated full-domain HIV loader diagnostic, with identical initialization checks."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf, open_dict

ROOT = Path("outputs/diagnostics/fragsmiles_hiv_full_seed42")
SOURCE = Path("data/shared_study_2m_v2/seed_trials/base_fragsmiles_hiv")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def config():
    cfg = OmegaConf.load("outputs/shared_study_2m_v2/seed_trials/base_fragsmiles_hiv/seed42/.hydra/config.yaml")
    with open_dict(cfg):
        cfg.algorithm._name = "fragsmiles_classifier"
        cfg.dataset._name = "molnet_hiv"
        cfg.wandb.mode = "disabled"
        cfg.resume = None
    return cfg


def prepare():
    from scripts.prepare_external_tsdy_training import prepare_task, load_tokenizer
    from datamodules.molecule_datasets.molnet import load_molnet
    from datasets import load_from_disk
    from utils.cache_utils import file_digest

    cfg = config()
    indices = json.loads((SOURCE / "hiv_split_indices.json").read_text())
    groups = {s: indices[s] for s in ("train", "validation", "test")}
    rows = list(csv.DictReader((SOURCE / "hiv.csv").open()))
    texts = {s: (SOURCE / f"hiv_canon_fragsmiles_{s}_seed42.txt").read_text().splitlines() for s in groups}

    def provider(split, start, stop):
        return [
            dict(source_row=i, text=texts[split][j], labels=[float(rows[i]["HIV_active"])])
            for j, i in enumerate(groups[split][start:stop], start)
        ]

    identity = dict(
        diagnostic="full_original_HIV_no_exclusion",
        files={str(p): file_digest(p) for p in SOURCE.iterdir() if p.is_file()},
    )
    prepare_task(
        ROOT / "prepared",
        groups,
        provider,
        identity,
        cfg.algorithm.tokenizer.pretrained,
        "fragsmiles",
        "hiv",
        workers=8,
        length_policy="truncate",
        resume=(ROOT / "prepared").exists(),
    )
    legacy = load_molnet(cfg.dataset, load_tokenizer(cfg.algorithm.tokenizer.pretrained, "fragsmiles"))
    prepared = load_from_disk(str(ROOT / "prepared/dataset"))
    for s in groups:
        assert len(legacy[s]) == len(prepared[s]) == len(groups[s])
        for i in range(len(groups[s])):
            a, b = legacy[s][i], prepared[s][i]
            for key in ("input_ids", "attention_mask", "labels"):
                assert a[key] == b[key], (s, i, key)
            assert not any(a.get("token_type_ids", [])), (s, i, "nonzero token types")
    write(
        ROOT / "inputs_verified.json",
        dict(passed=True, split_sizes={s: len(v) for s, v in groups.items()}, source=identity),
    )


def train(mode):
    import torch
    from experiments.downstream_classification import DownstreamClassificationExperiment

    assert json.loads((ROOT / "inputs_verified.json").read_text())["passed"]
    output = ROOT / mode
    if output.exists():
        raise ValueError(f"{output} exists; preserve it and inspect before retrying")
    cfg = config()
    with open_dict(cfg):
        cfg.name = f"DIAGNOSTIC-FRAGSMILES-FULL-HIV-{mode}-SEED42"
        if mode == "prepared_loader":
            cfg.dataset.prepared_path = str(ROOT / "prepared")
            cfg.dataset.prepared_task = "hiv"
            cfg.dataset.allow_smoke = False
    output.mkdir()
    OmegaConf.save(cfg, output / "config.yaml")
    experiment = DownstreamClassificationExperiment(cfg, output)
    experiment._build_algo()
    original = experiment.algo.build_model

    def checked_model(*args, **kwargs):
        model = original(*args, **kwargs)
        digest = hashlib.sha256()
        for name, t in sorted(model.state_dict().items()):
            digest.update(name.encode())
            digest.update(t.detach().cpu().contiguous().numpy().tobytes())
        info = dict(
            sha256=digest.hexdigest(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        )
        write(output / "initialization.json", info)
        if mode == "prepared_loader":
            assert info == json.loads(
                (ROOT / "legacy_loader/initialization.json").read_text()
            ), "Initialization/environment differs"
        return model

    experiment.algo.build_model = checked_model
    experiment.training()


def summarize():
    result = {m: json.loads((ROOT / m / "test_results.json").read_text()) for m in ("legacy_loader", "prepared_loader")}
    result["test_auc_difference"] = result["prepared_loader"]["test_roc_auc"] - result["legacy_loader"]["test_roc_auc"]
    write(ROOT / "comparison.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "legacy_loader", "prepared_loader", "summarize"])
    args = p.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "summarize":
        summarize()
    else:
        train(args.stage)
