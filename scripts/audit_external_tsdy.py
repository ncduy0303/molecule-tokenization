"""Offline pilot audit, with disjoint prefix training/validation slices.

The validation slice is held out from pilot vocabulary fitting, not a new
benchmark holdout. The input may belong entirely to tokenizer-training data.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from rdkit import Chem, rdBase
from utils.external_tsdy import encode, model_tokens, decode_tokens, COMPACT_GRAMMAR, molecules_equal, TOKEN_FORMAT
from utils.shared_fragmentation import partition_molecule, serialize_tsmiles
from utils.tsmiles_tokenizer import split_tsmiles_tokens
from utils.cache_utils import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-limit", type=int, default=2000)
    p.add_argument("--validation-limit", type=int, default=1000)
    a = p.parse_args()
    if a.output.exists() or min(a.train_limit, a.validation_limit) < 1:
        p.error("Use a new output path and positive sample sizes")
    samples = []
    failures = []
    seen = set()
    # De-duplicate by canonical molecular identity across the pilot slices.
    with a.input.open() as f:
        for line in f:
            if len(samples) == a.train_limit + a.validation_limit:
                break
            s = line.strip()
            try:
                expected = Chem.MolToSmiles(partition_molecule(s).mol)
                if expected in seen:
                    continue
                seen.add(expected)
                doc = encode(s)
                ts = model_tokens(doc)
                if not molecules_equal(decode_tokens(ts), expected):
                    raise ValueError("Token stream reconstruction mismatch")
                base = split_tsmiles_tokens(serialize_tsmiles(partition_molecule(s), "TSDY"), True)
                cores = [x for c in json.loads(doc)["components"] for x in c["cores"]]
                samples.append((ts, cores, base))
            except Exception as exc:
                failures.append(dict(smiles=s, error=str(exc)))
    if len(samples) < a.train_limit + a.validation_limit:
        raise ValueError("Not enough supported unique molecules")
    train, valid = samples[: a.train_limit], samples[a.train_limit :]
    # Reserve grammar even if absent in this sample. Seven model special tokens
    # are budgeted separately, matching the project convention.
    grammar = COMPACT_GRAMMAR
    tc = Counter(t for row in train for t in row[1])
    bc = Counter(t for row in train for t in row[2] if t not in {"&", "^", "."})
    vocab = set(t for t, _ in tc.most_common(5000 - 7 - len(grammar))) | grammar
    bvocab = set(t for t, _ in bc.most_common(5000 - 7 - 3)) | {"&", "^", "."}

    def metrics(rows, col, vocab):
        seqs = [r[col] for r in rows]
        return dict(
            mean_tokens=mean(map(len, seqs)),
            max_tokens=max(map(len, seqs)),
            over_510=sum(len(s) > 510 for s in seqs),
            unk_token_pct=100 * sum(t not in vocab for s in seqs for t in s) / sum(map(len, seqs)),
        )

    result = dict(
        format=TOKEN_FORMAT,
        rdkit=rdBase.rdkitVersion,
        input=str(a.input),
        input_sha256=file_digest(a.input),
        pilot_train=len(train),
        pilot_validation=len(valid),
        token_roundtrips=len(samples),
        excluded_count=len(failures),
        failures=failures[:30],
        unique_training_cores=len(tc),
        unique_training_tsdy_fragments=len(bc),
        reserved_grammar=len(grammar),
        enhanced_validation=metrics(valid, 0, vocab),
        base_validation=metrics(valid, 2, bvocab),
        base_fragment_unk_pct=100
        * sum(t not in bvocab for r in valid for t in r[2] if t not in {"&", "^", "."})
        / sum(sum(t not in {"&", "^", "."} for t in r[2]) for r in valid),
        enhanced_core_unk_pct=100 * sum(t not in vocab for r in valid for t in r[1]) / sum(len(r[1]) for r in valid),
        note="Matched pilot vocabularies, not full 2M trained vocabularies. Excluded molecules are reported, not silently substituted.",
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
