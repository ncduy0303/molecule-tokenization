"""
Adapts the PCATT GreedyTokenizer to work with Smirk tokens by mapping
Smirk glyphs to single-byte values.

This version uses raw `bytes` to avoid UTF-8 encoding issues.
"""

from typing import List


class SmirkPCATTAdapter:
    """
    Adapts the PCATT GreedyTokenizer to work with Smirk tokens by mapping
    Smirk glyphs to single-byte values.

    This version uses raw `bytes` to avoid UTF-8 encoding issues.
    """

    def __init__(self, smirk_vocab: List[str]):
        """
        Args:
            smirk_vocab: A list of all unique glyphs produced by Smirk.
        """
        self.smirk_vocab = smirk_vocab
        self.vocab_size = len(smirk_vocab)

        # We map glyphs to integer byte values (1 to 255).
        # We skip 0 (\x00) to avoid C-string null termination issues.
        if self.vocab_size > 255:
            raise ValueError(
                f"Smirk vocab size ({self.vocab_size}) exceeds 255. "
                "Cannot use single-byte mapping strategy."
            )

        # Bi-directional maps
        # Glyph -> Integer Byte Value (int)
        self.glyph_to_int = {glyph: i + 1 for i, glyph in enumerate(smirk_vocab)}
        # Integer Byte Value (int) -> Glyph
        self.int_to_glyph = {i + 1: glyph for i, glyph in enumerate(smirk_vocab)}

    def encode_for_pcatt(self, glyphs: List[str]) -> bytes:
        """
        Converts a list of glyphs into an encoded raw byte string ready for PCATT.

        Returns:
            bytes: A byte object.
        """
        try:
            byte_vals = [self.glyph_to_int[t] for t in glyphs]
            encoded_bytes = bytes(byte_vals)
            return encoded_bytes
        except KeyError as e:
            print(f"Error: Glyph {e} found in tokens but not in provided vocab.")
            raise e

    def decode_from_pcatt(self, raw_bytes: bytes) -> List[str]:
        """
        Converts a raw byte string from PCATT back into a list of Smirk glyphs.
        """
        byte_indices = list(raw_bytes)
        try:
            reconstructed_glyph_seq = [self.int_to_glyph[idx] for idx in byte_indices]
            return reconstructed_glyph_seq
        except KeyError as e:
            print(f"Warning: Byte value {e} not found in map. Skipping.")
            raise e
