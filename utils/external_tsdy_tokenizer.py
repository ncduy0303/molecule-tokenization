"""HF adapter for v3.1: SMILES input, indivisible cores and ordinary grammar words."""

import json
from pathlib import Path
from utils.tsmiles_tokenizer import TSMILESTokenizer
from utils.external_tsdy import TOKEN_FORMAT, COMPACT_GRAMMAR, encode, model_tokens, decode_tokens
from utils.cache_utils import file_digest, value_digest

SPECIALS = ["[BOS]", "[PAD]", "[EOS]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def compatible_codec(source, expected):
    actual = file_digest(Path(__file__).resolve().parents[1] / source)
    if actual == expected:
        return True
    # This exact patch changes equality validation only. Core generation,
    # model tokens, vocabulary IDs and decoding are byte-for-byte unchanged.
    return (source, expected, actual) == (
        "utils/external_tsdy.py",
        "ae2bcde9344b02126f5142153936e58e1d94e432bf5ba7f4cfa92eee53122366",
        "cec2e01fbce4fc159fb6cc255da1cd823b53f8aef3bec571c1437e4e1fd7e19b",
    )


def tokenizer_identity(tokenizer):
    return dict(
        tokenizer_class=type(tokenizer).__name__,
        vocab_sha256=value_digest(tokenizer.get_vocab()),
        specials={k: str(v) for k, v in tokenizer.special_tokens_map.items()},
        token_format=tokenizer.init_kwargs.get("token_format"),
        representation=tokenizer.init_kwargs.get("molecular_representation"),
    )


class ExternalTSDYTokenizer(TSMILESTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, token_format=None, vocabulary_identity=None, **kwargs):
        if token_format != TOKEN_FORMAT:
            raise ValueError(f"Expected enhanced tokenizer format {TOKEN_FORMAT}, got {token_format}")
        fitting = kwargs.get("fitting_manifest")
        if fitting:
            for source in ("utils/external_tsdy.py", "utils/shared_fragmentation.py"):
                if not compatible_codec(source, fitting["code"][source]):
                    raise ValueError("Tokenizer codec differs from its fitting manifest")
        vocab = json.loads(Path(vocab_file).read_text())
        if vocabulary_identity != value_digest(vocab):
            raise ValueError("Enhanced vocabulary identity mismatch")
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError("Vocabulary IDs must be unique and contiguous")
        if any(vocab.get(t) != i for i, t in enumerate(SPECIALS)) or not COMPACT_GRAMMAR <= vocab.keys():
            raise ValueError("Missing or changed special/grammar tokens")
        super().__init__(
            vocab_file=vocab_file, token_format=token_format, vocabulary_identity=vocabulary_identity, **kwargs
        )
        if len(self) != len(vocab) or set(map(str, self.all_special_tokens)) != set(SPECIALS):
            raise ValueError("Unexpected added tokens or model specials")

    def _tokenize(self, text, **kwargs):
        return model_tokens(encode(text))

    def convert_tokens_to_string(self, tokens):
        # Unambiguous inspection, not concatenated SMILES. Use decode_molecule for chemistry.
        return json.dumps(tokens, separators=(",", ":"))

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError("Unexpected second sequence")
            return [int(i in self.all_special_ids) for i in token_ids_0]
        return super().get_special_tokens_mask(token_ids_0, token_ids_1, False)

    def decode_molecule(self, ids):
        tokens = self.convert_ids_to_tokens(ids)
        if "[UNK]" in tokens or "[MASK]" in tokens:
            raise ValueError("Cannot reconstruct unknown or masked cores")
        return decode_tokens([t for t in tokens if t not in SPECIALS])


def export_tokenizer(preflight, output):
    """Export the already fitted word vocabulary, never refit or overwrite a mismatch."""
    preflight, output = Path(preflight), Path(output)
    summary = json.loads((preflight / "summary.json").read_text())
    manifest = json.loads((preflight / "manifest.json").read_text())
    meta = json.loads((preflight / "vocabulary/metadata.json").read_text())
    source = preflight / "vocabulary/vocab.json"
    if not summary["complete"] or not summary["vocabulary_gate_passed"] or summary["unsupported"]:
        raise ValueError("Preflight reconstruction gate has not passed")
    if meta["source"] != manifest or manifest["token_format"] != TOKEN_FORMAT:
        raise ValueError("Preflight/vocabulary provenance mismatch")
    for file, expected in [
        (source, meta["vocab_sha256"]),
        (preflight / "core_counts.json", meta["core_counts_sha256"]),
    ]:
        if file_digest(file) != expected:
            raise ValueError(f"Preflight checksum mismatch: {file}")
    for file in ("utils/external_tsdy.py", "utils/shared_fragmentation.py"):
        if not compatible_codec(file, manifest["code"][file]):
            raise ValueError(f"Codec changed since preflight: {file}")
    tok = ExternalTSDYTokenizer(
        str(source),
        token_format=TOKEN_FORMAT,
        vocabulary_identity=value_digest(json.loads(source.read_text())),
        model_max_length=512,
        fitting_manifest=manifest,
    )
    if output.exists():
        existing = ExternalTSDYTokenizer.from_pretrained(str(output), local_files_only=True)
        if (
            tokenizer_identity(existing) != tokenizer_identity(tok)
            or existing.init_kwargs.get("fitting_manifest") != manifest
        ):
            raise ValueError("Existing HF tokenizer differs; choose a new directory")
        return existing
    output.mkdir(parents=True)
    tok.save_pretrained(str(output))
    return ExternalTSDYTokenizer.from_pretrained(str(output), local_files_only=True)
