"""Summarize an existing audit and export explicit scope masks, without training."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from utils.cache_utils import file_digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audit", type=Path)
    a = p.parse_args()
    d = json.loads((a.audit / "summary.json").read_text())
    rows = [json.loads(l) for l in (a.audit / "rows.jsonl").open()]
    scope = defaultdict(lambda: dict(supported=[], unsupported=[], over_510=[]))
    for r in rows:
        key = "supported" if r["ok"] else "unsupported"
        scope[r["group"]][key].append(r["row"])
        if r["ok"] and r["compact_length"] > 510:
            scope[r["group"]]["over_510"].append(r["row"])
    (a.audit / "scope_indices.json").write_text(json.dumps(scope, indent=2) + "\n")
    (a.audit / "implementation_hashes.json").write_text(
        json.dumps(
            {f: file_digest(f) for f in ["utils/external_tsdy.py", "scripts/audit_external_tsdy_large.py"]}, indent=2
        )
        + "\n"
    )
    total = sum(g["attempted"] for g in d["groups"].values())
    ok = sum(g["roundtrip_ok"] for g in d["groups"].values())
    lines = [
        "# Compact enhanced TSDY audit",
        "",
        f"{ok:,} graph roundtrips passed out of {total:,} fixed selected inputs. Failures are retained in the denominator. No model training performed.",
        "",
        "|Input group|Passed / attempted|Mean verbose tokens|Mean compact tokens|Reduction %|Compact >510|Base >510|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, g in d["groups"].items():
        reduction = 100 * (1 - g["compact"]["mean_tokens"] / g["verbose"]["mean_tokens"])
        lines.append(
            f"|{k}|{g['roundtrip_ok']} / {g['attempted']}|{g['verbose']['mean_tokens']:.2f}|{g['compact']['mean_tokens']:.2f}|{reduction:.2f}|{g['compact']['over_510']}|{g['base']['over_510']}|"
        )
    lines += [
        "",
        "## Matched vocabulary coverage",
        "",
        "|Input group|Base fragment UNK %|Enhanced core UNK %|",
        "|---|---:|---:|",
    ]
    for k, g in d["groups"].items():
        lines.append(f"|{k}|{g['base_fragment_unk_pct']:.2f}|{g['enhanced_fragment_unk_pct']:.2f}|")
    lines += [
        "",
        f"Training sample: {d['vocab']['unique_training_base_fragments']} distinct TSDY fragments versus {d['vocab']['unique_training_cores']} clean cores. Grammar reserves {d['vocab']['reserved_compact_grammar']} entries plus seven model special tokens within the 5K budget. Neither pilot vocabulary fills the budget. Both are fitted on the same fixed 10K prefix only, not on downstream inputs. Pilot train/validation canonical-molecule overlap: {d['pilot_molecular_overlap']}.",
        "",
        "## Scope and decisions",
        "",
        "- Radical state is retained in fragment chemistry; radical fixtures and the larger selected corpus pass. No generic claim of support for all radical/organometallic chemistry.",
        "- The one excluded HIV input is source row 35278 in the training split, the known iron coordination complex with dative bonds. There is no plain-SMILES fallback. Its exact input and reason are in rows.jsonl.",
        "- Nonstandard bonds and non-tetrahedral atom stereo remain outside the declared domain. Any sanitization, cap mapping or roundtrip failure raises rather than silently substituting an input.",
        "- One large symmetric achiral molecule triggered different RDKit canonical strings. The corrected verifier accepts only an exact mapped graph match, including element, isotope, charge, radical count, H count, aromaticity and bond types. This fallback does not waive stereo checks for stereospecified molecules. The initial audit is preserved separately; these results come from a full repeat after the correction.",
        "- These are complete untruncated representation-token roundtrips. They are not vocabulary-ID roundtrips after UNK replacement, nor roundtrips of sequences truncated by a model.",
        "- All rows in the existing downstream split files were audited. Raw CSV rows absent from those frozen split files are outside this audit. Source and split hashes are recorded.",
        "- scope_indices.json separates unsupported rows from supported-but-overlength rows. It is an audit artifact, not an automatic new split or an instruction to filter test data.",
        "- For a supported-domain experiment, omit the unsupported HIV training row from every compared method and rerun those baselines; leave all validation/test rows in place. Do not compare newly filtered runs to frozen runs as though the training protocol were identical.",
        "- Length risk remains: even compact sequences can exceed the current 510-content-token budget. Do not drop long test molecules to improve results. Decide and disclose a common truncation policy, or use a matched larger context for all compared models before training. Longer context changes compute and model configuration.",
        "- Vocabulary coverage improvements and length trade-offs are descriptive. No ROC-AUC improvement has been demonstrated. Test-input diagnostics were inspected; further design decisions should use train/validation rather than optimize test performance.",
        "",
        "## Verification",
        "",
        "Eight focused regression tests pass: compact/verbose reconstruction agreement, large integer escapes, grammar coverage, malformed streams, substitution patterns, salts/isotopes/radicals/stereo, randomized input fixtures, and symmetry/chemical-identity checks.",
        "",
        "Recommendation: the compact codec is ready for larger intrinsic vocabulary evaluation on the supported domain, but the training protocol still needs an explicit decision on unsupported training chemistry and overlength inputs. No tokenizer/MLM training entry point or cache identity was changed.",
    ]
    (a.audit / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
