"""Pair shared TSDY fragments with enhanced clean cores using identical atom partitions."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time
from scripts.evaluate_intrinsic import inputs, write, digest
from utils.shared_fragmentation import partition_molecule, component_partitions, decorated_fragments
from utils.external_tsdy import _component, encode


def pairs(smiles):
    encode(smiles)  # Require full molecular reconstruction before interpreting sharing.
    result = []
    for part in component_partitions(partition_molecule(smiles)):
        doc = _component(part)
        sites = defaultdict(list)
        for a, b, _ in doc["links"]:
            sites[a[0]].append(a[1])
            sites[b[0]].append(b[1])
        decorated = decorated_fragments(part)
        for i, (core, frag) in enumerate(zip(doc["cores"], decorated, strict=True)):
            result.append(dict(core=core, tsdy=re.sub(r"\[\d+\*\]", "*", frag), sites=sorted(sites[i])))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--downstream-limit", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("outputs/vocabulary_sharing/full"))
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Output exists; choose a new directory")
    print("Loading fixed input samples...", flush=True)
    rows, source = inputs(a.limit, a.downstream_limit)
    print(f"Loaded {len(rows):,} molecules: {dict(Counter(r['group'] for r in rows))}", flush=True)
    # PubChem is the primary descriptive corpus; downstream groups remain separate.
    base = json.loads(Path("outputs/shared_study_2m_v2/base_TSDY/tokenizer/vocab.json").read_text())
    enhanced = json.loads(Path("outputs/enhanced_tsdy_v3_1/tokenizer/vocab.json").read_text())
    grouped = defaultdict(
        lambda: defaultdict(lambda: dict(variants=Counter(), counts=Counter(), sites=Counter(), occurrences=0))
    )
    failures = []
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        if index == 1 or index % 100 == 0:
            elapsed = time.monotonic() - started
            print(
                f"Starting molecule {index:,}/{len(rows):,}; group={row['group']} "
                f"id={row['id']}; elapsed={elapsed:.0f}s; failures={len(failures)}",
                flush=True,
            )
        try:
            records = pairs(row["smiles"])
        except Exception as e:
            failures.append(dict(row, error=str(e)))
            continue
        for r in records:
            d = grouped[row["group"]][r["core"]]
            d["occurrences"] += 1
            d["variants"][r["tsdy"]] += 1
            d["counts"][len(r["sites"])] += 1
            d["sites"][",".join(map(str, r["sites"]))] += 1
    print("Writing vocabulary-sharing results...", flush=True)
    result = {}
    for group, cores in grouped.items():
        result[group] = [
            dict(
                core=c,
                occurrences=d["occurrences"],
                distinct_variants=len(d["variants"]),
                core_in_enhanced_vocab=c in enhanced,
                variants=[dict(text=v, count=n, in_base_vocab=v in base) for v, n in d["variants"].most_common()],
                attachment_counts=dict(d["counts"]),
                attachment_site_patterns=dict(d["sites"]),
            )
            for c, d in cores.items()
        ]
        result[group].sort(key=lambda d: (-d["distinct_variants"], -d["occurrences"]))
    write(a.output / "sharing.json", result)
    write(a.output / "failures.json", failures)
    write(
        a.output / "manifest.json",
        dict(
            source=source,
            downstream_limit=a.downstream_limit,
            code={
                str(x): digest(x)
                for x in [Path(__file__), Path("utils/external_tsdy.py"), Path("utils/shared_fragmentation.py")]
            },
            vocabulary_hashes={
                str(x): digest(x)
                for x in [
                    Path("outputs/shared_study_2m_v2/base_TSDY/tokenizer/vocab.json"),
                    Path("outputs/enhanced_tsdy_v3_1/tokenizer/vocab.json"),
                ]
            },
            caveat="Site indices use canonical enhanced core atom order; symmetry can make distinct index patterns chemically equivalent. Sharing can also remove stereo distinctions; variants are not attributed solely to attachment sites.",
        ),
    )
    summary = {
        g: dict(
            clean_cores=len(cs),
            distinct_decorated_forms=len({v["text"] for c in cs for v in c["variants"]}),
            cores_with_multiple_forms=sum(c["distinct_variants"] > 1 for c in cs),
            rescued_occurrences=sum(
                v["count"] for c in cs if c["core_in_enhanced_vocab"] for v in c["variants"] if not v["in_base_vocab"]
            ),
            lost_occurrences=sum(
                v["count"] for c in cs if not c["core_in_enhanced_vocab"] for v in c["variants"] if v["in_base_vocab"]
            ),
        )
        for g, cs in result.items()
    }
    write(a.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
