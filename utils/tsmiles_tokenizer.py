"""
t-SMILES tokenizer implementation.

Consumes precomputed t-SMILES strings and tokenizes them using the
structural regex r'&{1}|[\\^]{1}|[^&\\^]+'. Tokens outside the vocabulary
are mapped to the configured [UNK] token.

Reference: fragsmiles_tokenizer.py
"""

import collections
import json
import logging
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}

# Regex that splits a t-SMILES string into atomic tokens:
#   &   – represents a non-fragment node in the FBT
#   ^   – represents a separator between two fragments
#   everything else – a maximal run of non-separator characters
_TSMILES_RE = re.compile(r"&{1}|\^{1}|[^&\^]+")


def load_vocab(vocab_file: str) -> collections.OrderedDict:
    """Load vocabulary from JSON file."""
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f, object_pairs_hook=collections.OrderedDict)
    return vocab


class TSMILESTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a t-SMILES tokenizer based on structural regex splitting.

    This tokenizer expects t-SMILES strings produced by tsmiles_utils and
    uses the regex r'&{1}|\^{1}|[^&\^]+' to split them into tokens.
    Unseen tokens are mapped to UNK.
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
        Tokenize a t-SMILES string using the structural separator regex.

        `^` and `&` are kept as individual tokens; everything between
        separators is a single fragment token.
        """
        try:
            tokens = [m for m in _TSMILES_RE.findall(text) if m]
            return tokens if tokens else [self.unk_token]  # type: ignore
        except Exception:
            return [self.unk_token]  # type: ignore

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token, -1))

    def _convert_id_to_token(self, index: int) -> str:
        """Converts a vocabulary ID back to its token string."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens back into a single string."""
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs with special tokens.

        Format:
        - single sequence: [CLS] X [SEP]
        - pair sequence:   [CLS] A [SEP] B [SEP]
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]  # type: ignore
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep  # type: ignore

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Return a mask with 1 for special tokens and 0 for regular tokens."""
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence already has special tokens."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Create token type IDs with the same length as built inputs."""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

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
