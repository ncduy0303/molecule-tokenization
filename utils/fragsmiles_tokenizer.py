"""
FragSMILES tokenizer implementation.

Uses chemicalgof to split SMILES strings into fragment tokens.
Maps each unique fragment to a unique ID based on frequency in the training corpus.
Unseen fragments are seamlessly mapped to the configured [UNK] token.
"""

import collections
import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


def load_vocab(vocab_file: str) -> collections.OrderedDict:
    """Load vocabulary from JSON file."""
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f, object_pairs_hook=collections.OrderedDict)
    return vocab


class FragSMILESTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a FragSMILES tokenizer based on fragment decomposition.

    This tokenizer uses chemicalgof to decompose SMILES strings into fragment tokens.
    Each unique fragment is mapped to a unique token ID based on frequency in the
    training corpus. Unseen fragments are mapped to the UNK token.
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        **kwargs,
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")

        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize a SMILES string using fragSMILES decomposition.
        """
        import chemicalgof

        try:
            # Encode and split the SMILES string
            encoded = chemicalgof.encode(text, canonical=True)
            fragments = chemicalgof.split(encoded)

            # Convert fragments to tokens
            tokens = [frag for frag in fragments if frag]
            return tokens if tokens else [self.unk_token]  # type: ignore
        except Exception:
            # Fallback to UNK token if strictly invalid text is passed
            return [self.unk_token]  # type: ignore

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token, -1))

    def _convert_id_to_token(self, index: int) -> str:
        """Convert a vocabulary ID back to its token string."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of string tokens back into a single string."""
        import chemicalgof

        # Because fragments are from an *encoded* string, we must attempt to reverse the process.
        joined_encoded = "".join(tokens)
        try:
            return chemicalgof.decode(joined_encoded)
        except Exception:
            # Fallback (useful if special tokens are passed into this method)
            return joined_encoded

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Saves the vocabulary to a directory."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)

        vocab_file = VOCAB_FILES_NAMES["vocab_file"]
        if filename_prefix is not None:
            vocab_file = filename_prefix + "-" + vocab_file

        output_vocab_file = os.path.join(save_directory, vocab_file)

        with open(output_vocab_file, "w", encoding="utf-8") as writer:
            json.dump(self.vocab, writer, ensure_ascii=False, indent=2)

        return (output_vocab_file,)
