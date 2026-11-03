#!/bin/bash
# Run from repo root; include failed/incomplete attempts for diagnosis.
set -euo pipefail
root=outputs/ca_seed_trials
bundle=ca_seed_trials_review.tar.gz
[[ -d "$root" ]]
find "$root" data/ca_seed_trials slurm_logs -type f \( \
  -name 'submissions.json' -o -name 'submission_pending.json' -o -name 'complete.json' \
  -o -name 'test_results.json' -o -name 'validation_results.json' -o -name 'prepared_input.json' \
  -o -path '*/.hydra/config.yaml' -o -path '*/.hydra/overrides.yaml' \
  -o -name 'manifest.json' -o -name 'ready.json' -o -name '*split_indices.json.meta.json' \
  -o -name 'ca_prepare_*.out' -o -name 'ca_prepare_*.err' \
  -o -name 'ca_trial_*.out' -o -name 'ca_trial_*.err' \) -print0 |
  tar -czf "$bundle" --null -T -
echo "$bundle"
