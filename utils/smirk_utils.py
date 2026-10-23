"""Load the same SMIRK vocabulary for pretraining and downstream use."""

from pathlib import Path


def load_smirk_tokenizer(pretrained=None):
    from smirk import SmirkTokenizerFast

    if not pretrained:
        return SmirkTokenizerFast()
    path = Path(pretrained)
    if path.is_dir():
        return SmirkTokenizerFast.from_pretrained(str(path))
    if path.is_file():
        return SmirkTokenizerFast(tokenizer_file=str(path))
    raise FileNotFoundError(f"SMIRK tokenizer not found: {path}. Supply a saved directory or tokenizer.json file.")
