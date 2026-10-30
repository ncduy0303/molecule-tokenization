"""Read-only W&B metadata export. Uses existing login; never accepts a key argument."""

import argparse
import json
from pathlib import Path
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="ncduy0303/molecule-tokenization")
    parser.add_argument("--output", type=Path, default=Path("outputs/intrinsic_inventory/wandb_runs.json"))
    args = parser.parse_args()
    rows = []
    for run in wandb.Api(timeout=30).runs(args.project, per_page=100, include_sweeps=False, lazy=False):
        summary = run._attrs.get("summaryMetrics", {})
        if isinstance(summary, str):
            summary = json.loads(summary)
        rows.append(
            dict(
                id=run.id,
                name=run.name,
                state=run.state,
                url=run.url,
                config=run.config,
                summary=summary,
                tags=run.tags,
            )
        )
        if len(rows) % 100 == 0:
            print(f"Fetched {len(rows)} runs", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, default=str) + "\n")
    print(f"Exported {len(rows)} runs to {args.output}")
