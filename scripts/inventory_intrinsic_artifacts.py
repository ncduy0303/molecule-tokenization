"""Build a read-only artifact registry for the approved intrinsic study."""

import argparse
import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf


def inventory(root):
    entries = []
    candidates = list((root / "data/pretrained_tokenizers").glob("*/tokenizer_config.json"))
    candidates += list((root / "outputs/shared_study_2m_v2").glob("*/tokenizer/tokenizer_config.json"))
    candidates += [root / "outputs/enhanced_tsdy_v3_1/tokenizer/tokenizer_config.json"]
    for config in sorted(set(candidates)):
        if not config.exists():
            continue
        folder = config.parent
        cfg = json.loads(config.read_text())
        sources = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.iterdir()
            if p.is_file() and (p.suffix in (".json", ".txt")) and not p.name.startswith("corpus")
        }
        hydra = folder / ".hydra/config.yaml"
        if not hydra.exists():
            hydra = folder.parent / ".hydra/config.yaml"
        source_cfg = OmegaConf.to_container(OmegaConf.load(hydra), resolve=False) if hydra.exists() else None
        entries.append(
            dict(
                path=str(folder.relative_to(root)),
                tokenizer_class=cfg.get("tokenizer_class"),
                molecular_representation=cfg.get("molecular_representation"),
                artifact_sha256=sources,
                training_config=source_cfg,
                status="needs_adapter_and_run_mapping",
            )
        )
    return entries


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("outputs/intrinsic_inventory/registry.json"))
    a = p.parse_args()
    rows = inventory(Path.cwd())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Registered {len(rows)} tokenizer artifacts: {a.output}")
