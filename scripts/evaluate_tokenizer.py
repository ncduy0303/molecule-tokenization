"""Measure one saved tokenizer on a SMILES file, with the project's intrinsic metrics."""

import argparse
import json
from pathlib import Path

from utils.intrinsic_evaluation import evaluate
from utils.cache_utils import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="One SMILES per line")
    p.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="Tokenizer directory, or an APE vocab file",
    )
    p.add_argument(
        "--representation",
        required=True,
        choices=["smiles", "safe", "tsmiles", "fragsmiles", "shared", "enhanced"],
    )
    p.add_argument("--variant", choices=["fragsmiles", "TSSA", "TSID", "TSDY", "TSIS"])
    p.add_argument("--slicer", choices=["brics", "mmpa"])
    p.add_argument("--ps-merges", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    rep = a.representation
    if (rep in ("shared", "tsmiles")) != bool(a.variant):
        p.error("--variant is required exactly for shared and tsmiles")
    if (rep == "shared" and a.variant == "TSSA") or (
        rep == "tsmiles" and a.variant == "fragsmiles"
    ):
        p.error("Unsupported representation and variant combination")
    if a.ps_merges and rep != "shared" or a.slicer and rep not in ("tsmiles", "safe"):
        p.error("Invalid fragmentation options for representation")
    item = dict(
        name=a.tokenizer.name,
        path=str(a.tokenizer),
        representation=rep,
        variant=a.variant,
        slicer=a.slicer or "brics",
        ps_path=str(a.ps_merges) if a.ps_merges else None,
    )
    if a.tokenizer.is_file():
        if rep not in ("smiles", "safe"):
            p.error("APE vocabulary files require smiles or safe")
        item["kind"] = "ape"
    elif not a.tokenizer.is_dir():
        p.error("Tokenizer does not exist")
    if a.tokenizer.is_dir():
        config = json.loads((a.tokenizer / "tokenizer_config.json").read_text())
        cls = config["tokenizer_class"]
        base = a.variant if rep == "shared" else rep
        expected = (
            "ExternalTSDYTokenizer"
            if base == "enhanced"
            else "FragSMILESTokenizer"
            if base == "fragsmiles"
            else "TSMILESTokenizer"
            if rep in ("tsmiles", "shared")
            else None
        )
        fragment_classes = {
            "ExternalTSDYTokenizer",
            "FragSMILESTokenizer",
            "TSMILESTokenizer",
        }
        if (expected and cls != expected) or (not expected and cls in fragment_classes):
            p.error(f"Representation {rep} does not match saved tokenizer class {cls}")
        recorded = config.get("molecular_representation")
        if recorded or rep == "shared":
            from utils.cache_utils import representation_identity

            settings = dict(
                use_fragsmiles=base == "fragsmiles",
                use_tsmiles=expected == "TSMILESTokenizer",
                fragmentation="exocyclic" if rep == "shared" else None,
                ps_merges_path=item["ps_path"],
                tsmiles_variant=a.variant,
            )
            if representation_identity(settings) != recorded:
                p.error(
                    "Representation or merge rules differ from the fitted tokenizer"
                )
    with a.input.open() as f:
        rows = [
            dict(id=i, group="input", smiles=line.strip()) for i, line in enumerate(f)
        ]
    if not rows:
        p.error("Input is empty")
    identity = dict(
        input_sha256=file_digest(a.input),
        settings=item,
        ps_sha256=file_digest(a.ps_merges) if a.ps_merges else None,
    )
    files = (
        [a.tokenizer]
        if a.tokenizer.is_file()
        else [
            path
            for path in a.tokenizer.iterdir()
            if path.is_file()
            and path.suffix in (".txt", ".json")
            and not path.name.startswith("corpus")
        ]
    )
    identity["tokenizer_files"] = {path.name: file_digest(path) for path in files}
    a.output.mkdir(parents=True, exist_ok=False)
    evaluate((item, rows, a.output, identity))
    print(a.output / "summary.json")


if __name__ == "__main__":
    main()
