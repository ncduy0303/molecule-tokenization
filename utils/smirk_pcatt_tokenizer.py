"""
HuggingFace-compatible wrapper for the Smirk-PCATT tokenizer pipeline.

Composes three components in sequence:
  1. SmirkTokenizerFast  — splits a SMILES string into Smirk glyphs
  2. SmirkPCATTAdapter   — maps glyphs to single-byte values
  3. GreedTok (PCATT)    — byte-level subword tokenization

The wrapper transparently converts SMILES strings to bytes before
delegating to the underlying GreedTok, so it can be used as a drop-in
replacement in HuggingFace Trainer, DataCollator, etc.
"""

from __future__ import annotations

import json
from pathlib import Path

from utils.smirk_pcatt_adapter import SmirkPCATTAdapter

# Metadata file saved alongside the GreedTok tokenizer
_ADAPTER_META_FILE = "smirk_pcatt_meta.json"


class SmirkPCATTTokenizer:
    """
    Wrapper that adds Smirk pre-tokenization + byte encoding on top of
    a PCATT GreedTok tokenizer.

    Transparently converts SMILES string inputs to byte-encoded form
    before calling the underlying GreedTok, so userland code can pass
    plain SMILES strings and get proper token IDs back.
    """

    def __init__(self, greedtok, smirk_tokenizer, adapter: SmirkPCATTAdapter):
        self._greedtok = greedtok
        self._smirk = smirk_tokenizer
        self._adapter = adapter

    # ── SMILES → bytes conversion ───────────────────────────────────────

    def _smiles_to_bytes(self, smiles: str) -> bytes:
        """Convert a single SMILES string to adapter-encoded bytes."""
        tokens = self._smirk.tokenize(smiles)
        return self._adapter.encode_for_pcatt(tokens)

    def _convert_input(self, text):
        """Convert string or list-of-strings to bytes for GreedTok."""
        if isinstance(text, str):
            return self._smiles_to_bytes(text)
        elif isinstance(text, bytes):
            return text
        elif isinstance(text, list):
            return [self._convert_input(t) for t in text]
        return text

    # ── HuggingFace tokenizer interface ─────────────────────────────────

    def __call__(self, text, text_pair=None, **kwargs):
        text = self._convert_input(text)
        if text_pair is not None:
            text_pair = self._convert_input(text_pair)
        return self._greedtok(text, text_pair=text_pair, **kwargs)

    def tokenize(self, text, **kwargs):
        text = self._convert_input(text)
        return self._greedtok.tokenize(text, **kwargs)

    def encode(self, text, text_pair=None, **kwargs):
        text = self._convert_input(text)
        if text_pair is not None:
            text_pair = self._convert_input(text_pair)
        return self._greedtok.encode(text, text_pair=text_pair, **kwargs)

    def decode(self, token_ids, **kwargs):
        return self._greedtok.decode(token_ids, **kwargs)

    def batch_decode(self, sequences, **kwargs):
        return self._greedtok.batch_decode(sequences, **kwargs)

    # ── Persistence ─────────────────────────────────────────────────────

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save the GreedTok tokenizer and adapter metadata."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the underlying GreedTok
        self._greedtok.save_pretrained(str(save_dir), **kwargs)

        # Save adapter metadata (smirk_vocab list) so we can reconstruct
        meta = {"smirk_vocab": self._adapter.smirk_vocab}
        with open(save_dir / _ADAPTER_META_FILE, "w") as f:
            json.dump(meta, f)

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> "SmirkPCATTTokenizer":
        """Load a saved SmirkPCATTTokenizer from a directory."""
        from pcatt.hf.greedtok import GreedTok
        from smirk import SmirkTokenizerFast

        pretrained_dir = Path(pretrained_path)

        # Load GreedTok
        greedtok = GreedTok.from_pretrained(str(pretrained_dir), **kwargs)

        # Load adapter metadata
        meta_path = pretrained_dir / _ADAPTER_META_FILE
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            smirk_vocab = meta["smirk_vocab"]
        else:
            # Fallback: reconstruct from Smirk tokenizer (same logic as training)
            smirk_tok = SmirkTokenizerFast()
            smirk_vocab = [token for token, _ in sorted(smirk_tok.get_vocab().items(), key=lambda item: item[1])]

        adapter = SmirkPCATTAdapter(smirk_vocab)
        smirk_tokenizer = SmirkTokenizerFast()

        return cls(greedtok, smirk_tokenizer, adapter)

    # ── Delegate everything else to the underlying GreedTok ─────────────

    def __len__(self):
        return len(self._greedtok)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying GreedTok tokenizer."""
        # This is only called when normal attribute lookup fails,
        # so self._greedtok etc. are found normally via __init__.
        return getattr(self._greedtok, name)
