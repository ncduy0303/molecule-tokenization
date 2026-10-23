"""Local Hugging Face Trainer checkpoint validation."""

from pathlib import Path
import json


def resolve_resume_checkpoint(cfg):
    if cfg.get("load") is not None:
        raise ValueError(
            "The legacy load option is unsupported. Use algorithm.model.pretrained for weights, "
            "algorithm.model.finetuned for a classifier, or resume=/path/to/checkpoint-N for training state."
        )
    resume = cfg.get("resume")
    if resume is None:
        return None
    if not isinstance(resume, str) or not resume.strip():
        raise ValueError("resume must be a local Hugging Face Trainer checkpoint directory")
    if "model" not in cfg.algorithm or list(cfg.experiment.tasks) != ["training"]:
        raise ValueError("resume is only supported for MLM or classification with experiment.tasks=[training]")
    path = Path(resume).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Resume checkpoint directory does not exist: {path}. W&B run IDs are not supported.")
    required = ("config.json", "trainer_state.json", "optimizer.pt", "scheduler.pt")
    missing = [name for name in required if not (path / name).is_file()]
    weights = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    if not any((path / name).is_file() for name in weights):
        missing.append("model weights")
    if not list(path.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    if missing:
        raise ValueError(
            f"{path} is not a full Trainer checkpoint; missing {missing}. "
            "Final model exports only support weight loading, not exact training resumption."
        )
    try:
        state = json.loads((path / "trainer_state.json").read_text())
        if not isinstance(state["global_step"], int) or state["global_step"] < 0:
            raise ValueError("invalid global_step")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid trainer_state.json in {path}") from exc
    return path
