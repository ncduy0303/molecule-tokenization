# Shared exocyclic fragmentation study

## What is implemented

The shared mode separates molecular partitioning from sequence serialization:

1. Parse and canonicalize the molecular graph once. Preserve salts, isotopes,
   formal charges and stereochemistry; do not run the legacy t-SMILES standardizer.
2. Select chemicalgof's default cut SMARTS:
   `[!$([+1,-1]~[-1,+1])]-&!@[*]`. This means non-ring single bonds with the
   library's exception for oppositely charged atoms. Shared version 2 also
   requires each cut to be a graph bridge (removing it disconnects its endpoints).
   RDKit 2025.09.4 can mark a cycle edge as non-ring in a highly bridged cage;
   the bridge check keeps that edge inside the fragment instead of failing.
3. Optionally apply one frozen model of learned PS-style merges to these blocks.
4. Serialize the resulting atom partition as fragSMILES, TSID or TSDY.

TSID and TSDY use the same first-child/next-sibling full binary tree and
breadth-first traversal, including the upstream extra root dummy. The new adapter
accepts the partition directly. It does not call BRICS/MMPA or construct a
spanning tree by dropping molecular edges. Root and child order are deterministic
from canonical atom order. Fragment strings use aromatic, isomeric RDKit SMILES.

TSDY differs from TSID only by removing the attachment dummy isotope IDs. It can
be compared for prediction, but exact unique reconstruction is not promised.
The TSID audit decoder pairs attachment IDs; it is not a validator for arbitrary
generated binary-tree strings. Input dummy atoms and atom-mapped input are rejected.

The shared path raises encoding errors instead of falling back to plain SMILES.
Existing BRICS/MMPA and legacy PS-fragSMILES paths remain available with
`dataset.fragmentation=null`. The installed chemicalgof dependency is unchanged.

## Experiment matrix

| Fragmentation                     | fragSMILES    | TSID     | TSDY     |
| --------------------------------- | ------------- | -------- | -------- |
| Exocyclic base blocks             | Baseline      | Baseline | Baseline |
| Same blocks plus frozen PS merges | PS-fragSMILES | PS-TSID  | PS-TSDY  |

The same row means exactly the same original atom groups and cut bonds, not the
same decorated fragment strings. Dummies and connector indices are notation.
Keep source rows, preprocessing, MLM splits and downstream splits fixed across
all cells. This compares complete representation/tokenization pipelines; native
token boundaries and masking do not represent identical amounts of chemistry.

PS mining learns recurring connected unions of base blocks. Its key retains
bond order, charges and isotopes but ignores stereochemistry; the partition keeps
the original molecular graph and stereo. Overlapping occurrences merge in
canonical bond-index order, without overlap within one rule. Rules are replayed
in training order, including any repeated rule. This is constrained PS-style
graph BPE, not the original atom-initialized PS-VAE miner or a neural VAE.

`num_merges` controls mining. `vocab_size` controls the later model-token
vocabulary. They are separate: adding merges need not increase the final model
vocabulary or improve prediction. It cannot increase the number of fragments
for an individual molecule, but serialized token lengths need measurement.

The new miner updates candidate counts only in affected molecules. It retains
states in memory and is currently single-process. There is no demonstrated 2M
scaling result or mining-resume support. Start with a bounded pilot and measure
time/memory before increasing the corpus. Legacy `ps_vocab.txt` files cannot be
used as shared merge models.

## Cluster setup

### Bridge-safe correction (shared version 2)

The 2M TSID/TSDY jobs exposed `Cut bonds disagree with fragment membership`.
The failure is reproducible on pinned PubChem source row 2832359:
`CC1C2C3CC45C6C7C38CC1(C)C8(C)C7(C)C4(C)C1(C)C6C3C4CC26CC35C1(C)C4(C)C6C`.
The old SMARTS proposes a cut inside this cage, although its endpoints remain
connected. Version 2 intersects SMARTS matches with true graph bridges and
preserves the existing validation. It does not discard the molecule.

The same fix applies to all representations and PS mining. New representation
metadata uses `shared_version: 2`; new merge models use
`shared-exocyclic-ps-v2`. Old models must be re-mined, not relabelled. Keep old
artifacts as pilot results and use fresh cache/output directories, for example
`data/shared_study_2m_v2/` and `outputs/shared_study_2m_v2/`. Restart all three
base jobs with the corrected code. An already-running process still uses the old
code. Failed tokenizer encoding does not save its partial in-memory corpus, so
it restarts encoding from the beginning.

The completed 200K mining pilot remains useful for resource planning: supplied
Slurm accounting reports 00:22:03 elapsed and 9.21G MaxRSS, and its log confirms
1,000 merges. These are whole-job measurements, not a mining-only benchmark.
They do not guarantee linear scaling to 2M.

Run all commands from the repository root with the updated code on the cluster.
No new dependencies are needed. Run `source .venv/bin/activate` and `wandb login`
when online logging is wanted. Each block below submits jobs and returns before
they finish. Wait for the required preceding stage to succeed.

Resolve and save one dataset revision. Run the assignment and guard again in
each new shell before submission. Do not replace the revision file during a study.

```bash
mkdir -p outputs/shared_study
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
path = Path('outputs/shared_study/pubchem_revision.txt')
if not path.exists():
    revision = HfApi().dataset_info('mikemayuare/PubChem10M_SMILES_SELFIES').sha
    with path.open('x') as stream:
        stream.write(revision + '\n')
PY
FYP_HF_REV=$(cat outputs/shared_study/pubchem_revision.txt)
: "${FYP_HF_REV:?Dataset revision must not be empty}"
```

Examples use a **20,000-molecule tokenizer/mining pilot** and a 5,000-token
model vocabulary. This is not a 2M-tokenizer-corpus experiment. Use identical
corpus size and seed in every tokenizer/miner job when increasing the scale.

### Base tokenizers, without learned merging

```bash
bash scripts/submit_long.sh SHARED-EXO-FRAGSMILES-5K \
  experiment=tokenizer_training dataset=pubchem10m_tokenizer_train \
  algorithm=train_fragsmiles dataset.use_fragsmiles=true \
  dataset.fragmentation=exocyclic dataset.hf_revision="$FYP_HF_REV" \
  dataset.corpus_size=20000 dataset.seed=42 \
  dataset.data_dir=data/shared_study/base_fragsmiles \
  algorithm.tokenizer.vocab_size=5000 \
  hydra.run.dir=outputs/shared_study/base_fragsmiles

for variant in TSID TSDY; do
  bash scripts/submit_long.sh "SHARED-EXO-${variant}-5K" \
    experiment=tokenizer_training dataset=pubchem10m_tokenizer_train \
    algorithm=train_tsmiles dataset.use_tsmiles=true \
    dataset.fragmentation=exocyclic dataset.tsmiles_variant="$variant" \
    dataset.hf_revision="$FYP_HF_REV" dataset.corpus_size=20000 dataset.seed=42 \
    dataset.data_dir="data/shared_study/base_${variant}" \
    algorithm.tokenizer.vocab_size=5000 \
    hydra.run.dir="outputs/shared_study/base_${variant}"
done
```

In shared mode `tsmiles_slicer` is ignored: fragmentation comes from the common
partition, even if that legacy config field still says `brics`. Cache filenames
and logs label the effective slicer `exocyclic`.

### Optional: mine one PS model for all representations

The miner takes raw canonical SMILES. Do not enable `use_fragsmiles`,
`use_tsmiles`, or `fragmentation` on this mining job.

```bash
bash scripts/submit_long.sh SHARED-PS-MINING-20K \
  experiment=tokenizer_training dataset=pubchem10m_tokenizer_train \
  algorithm=train_shared_ps \
  dataset.hf_revision="$FYP_HF_REV" dataset.corpus_size=20000 dataset.seed=42 \
  dataset.data_dir=data/shared_study/miner \
  algorithm.tokenizer.num_merges=1000 algorithm.tokenizer.min_frequency=10 \
  hydra.run.dir=outputs/shared_study/miner
```

After it finishes, the frozen artifact is
`outputs/shared_study/miner/ps_merges.json`. It records corpus provenance and
ordered merge rules. The miner may stop before the requested merge count when
no candidate meets the minimum frequency. It refuses to overwrite a model.

Repeat the base tokenizer commands with:

```text
dataset.ps_merges_path=outputs/shared_study/miner/ps_merges.json
```

Use new `ps_fragsmiles`, `ps_TSID`, and `ps_TSDY` cache/output directories and
distinct run names. Keep the same `train_fragsmiles`/`train_tsmiles` algorithms.
**Do not use `train_ps_fragsmiles` for this study:** that is the older pipeline.

The PS artifact must have been trained on exactly the same tokenizer source-row
selection. The loader verifies this for tokenizer training and MLM exclusion.
All rows used to learn merges are therefore excluded from MLM splits too.

### Audit before expensive jobs

The optional miner creates a verified raw `corpus.txt`, usable for this check:

```bash
python -m scripts.audit_shared_fragmentation \
  --input data/shared_study/miner/corpus.txt \
  --limit 1000 \
  --output outputs/shared_study/audit_base.json

python -m scripts.audit_shared_fragmentation \
  --input data/shared_study/miner/corpus.txt \
  --limit 1000 \
  --ps-merges-path outputs/shared_study/miner/ps_merges.json \
  --output outputs/shared_study/audit_ps.json
```

Without a mining run, provide any SMILES text file to the audit. This diagnostic
does not change training selections. It reports parse/partition failures,
encoding failures, exact isomeric reconstruction, and fragment counts. Add
`--require-roundtrip` to return a nonzero status on any mismatch/error. TSDY is
not assigned a reconstruction score. Existing output files are not overwritten.

### MLM example: base TSID

Wait until its tokenizer succeeds. This trains the configured full MLM schedule,
not a one-epoch pilot. Reduce training settings explicitly for a smoke run.

```bash
export WANDB_TAGS="shared-exocyclic,tsid,mlm"
bash scripts/submit_gpu_long.sh SHARED-EXO-TSID-5K-1M-20Epoch \
  experiment=tokenizer_benchmark dataset=pubchem10m_mlm_train \
  algorithm=tsmiles_roberta dataset.use_tsmiles=true \
  dataset.fragmentation=exocyclic dataset.tsmiles_variant=TSID \
  dataset.hf_revision="$FYP_HF_REV" dataset.seed=42 \
  dataset.data_dir=data/shared_study/base_TSID_mlm \
  dataset.tokenizer_indices_path=data/shared_study/base_TSID/subset_indices.json \
  algorithm.tokenizer.pretrained=outputs/shared_study/base_TSID/tokenizer \
  experiment.training.save_strategy=epoch experiment.training.save_total_limit=1 \
  wandb.mode=online wandb.entity=ncduy0303 wandb.project=molecule-tokenization \
  hydra.run.dir=outputs/shared_study/base_TSID_mlm
```

This retains only the latest resumable intermediate checkpoint, plus the final
export. Use `experiment.training.save_strategy=no` for final-export-only saving.

For TSDY, change the variant, tokenizer and all paths consistently. For
fragSMILES, use `algorithm=fragsmiles_roberta` and `dataset.use_fragsmiles=true`
instead of `use_tsmiles`. For a PS condition, also pass the same frozen
`dataset.ps_merges_path` used to train that tokenizer.

### Downstream example: base TSID

After MLM succeeds:

```bash
for task in hiv bbbp; do
  WANDB_TAGS="shared-exocyclic,tsid,${task}" \
  bash scripts/submit_gpu_long.sh "SHARED-EXO-TSID-${task}" \
    experiment=downstream_classification dataset="molnet_${task}" \
    algorithm=tsmiles_classifier dataset.use_tsmiles=true \
    dataset.fragmentation=exocyclic dataset.tsmiles_variant=TSID \
    dataset.data_dir=data/shared_study/base_TSID_molnet \
    algorithm.tokenizer.pretrained=outputs/shared_study/base_TSID/tokenizer \
    algorithm.model.pretrained=outputs/shared_study/base_TSID_mlm/final_checkpoint \
    wandb.mode=online wandb.entity=ncduy0303 wandb.project=molecule-tokenization \
    hydra.run.dir="outputs/shared_study/base_TSID_${task}"
done
```

Use `fragsmiles_classifier` for fragSMILES. Keep the same variant and optional PS
model at every stage. Tokenizer artifacts record the representation identity;
MLM and downstream loading reject mismatched settings/model checksums.

## Limits and interpretation

- The old chemicalgof library can still lose stereochemistry or fail decoding.
  In a local diagnostic on the first 100 BBBP rows, 98 parsed; all 98 TSID
  encodings reconstructed exactly. Standard shared fragSMILES reconstructed
  93 exactly, had 4 mismatches, and 1 decode error. This is a non-random diagnostic,
  not a dataset-wide estimate. Source data and package versions matter.
- The no-merge fragSMILES serializer intentionally matches the pinned library.
  It does not claim to fix all chemicalgof stereochemistry behavior. Review audit
  failures before claiming the representations contain identical information.
- Shared tokenizers preserve disconnected-component delimiters as tokens.
  Historical tokenizers may not. Train new tokenizers for this study instead
  of mixing old checkpoints into the matched comparison.
- `5K` is a vocabulary cap, not a guarantee that every condition produces exactly
  5,000 distinct tokens. Record actual vocabulary size, UNK rate, sequence
  lengths, truncation, and downstream results. Do not infer performance from
  compression alone or compare raw token-level perplexity as though token units
  were identical.
- No SAFE adapter or chemicalgof 0.3.0 migration is included in this implementation.

## Validation

```bash
MPLCONFIGDIR=/tmp/fyp-matplotlib \
  python -m unittest discover -s tests
```

New tests check cut-rule parity, atom coverage, rings and multiple bonds,
isomeric TSID reconstruction, exact TSDY ID removal, agreement with upstream
binary-tree serialization, deterministic incremental mining against full recount,
cache multiplicity, artifact reload, and all three representations through
tokenizer training, MLM dataset preparation, and HIV/BBBP dataset preparation.
These are offline fixtures, not completed cluster experiments.
