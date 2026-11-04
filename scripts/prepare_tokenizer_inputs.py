"""Prepare one enhanced-TSDY MLM or binary classification dataset from three CSV splits."""

import argparse
from pathlib import Path

from datasets import Dataset

from scripts.prepare_external_tsdy_training import SPLITS, prepare_task
from utils.cache_utils import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument(f"--{split}", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--label-column", help="Binary 0/1 column. Omit for MLM.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument(
        "--length-policy", choices=["error", "truncate"], default="error"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.shard_size) < 1:
        parser.error("Workers and shard size must be positive")

    sources = {s: getattr(args, s) for s in SPLITS}
    hashes = {s: file_digest(path) for s, path in sources.items()}
    if len(set(hashes.values())) != len(SPLITS):
        parser.error("Train, validation and test must be distinct split files")
    datasets = {s: Dataset.from_csv(str(path)) for s, path in sources.items()}
    for s, dataset in datasets.items():
        required = {args.smiles_column}
        if args.label_column:
            required.add(args.label_column)
        if not len(dataset) or required.difference(dataset.column_names):
            parser.error(
                f"Empty split or missing columns in {s}: expected {sorted(required)}"
            )

    def provide(split, start, stop):
        records = []
        batch = datasets[split][start:stop]
        for offset, text in enumerate(batch[args.smiles_column]):
            row = start + offset
            record = dict(source_row=row, text=text)
            if args.label_column:
                label = batch[args.label_column][offset]
                if label not in (0, 1):
                    raise ValueError(
                        f"{split} row {row}: expected a binary label, got {label!r}"
                    )
                record["labels"] = [float(label)]
            records.append(record)
        return records

    prepare_task(
        args.output,
        {s: list(range(len(ds))) for s, ds in datasets.items()},
        provide,
        dict(
            split_sha256=hashes,
            smiles_column=args.smiles_column,
            label_column=args.label_column,
        ),
        args.tokenizer,
        "enhanced",
        "classification" if args.label_column else "mlm",
        workers=args.workers,
        shard_size=args.shard_size,
        length_policy=args.length_policy,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
