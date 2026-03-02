"""
Utility functions for SAFE (Sequential Attachment-based Fragment Embedding) encoding.

SAFE strings are a rearrangement of SMILES that groups by molecular fragments.
Every SAFE string is still a valid SMILES string, so downstream tokenizers
and models require no modifications.

Usage:
    from utils.safe_utils import encode_safe, encode_safe_batch

    safe_str = encode_safe("CC(Cc1ccc(cc1)C(C(=O)O)C)C")
    # -> "c12ccc3cc1.C3(C)C(=O)O.CC(C)C2"  (still valid SMILES)
"""

import safe
from typing import List
from rdkit import Chem


def encode_safe(smiles: str, slicer: str = "brics") -> str:
    """
    Convert a SMILES string to its SAFE representation.

    If encoding fails (e.g. the molecule cannot be fragmented),
    the original SMILES is canonicalized and returned instead;
    if that also fails, the original SMILES is returned as-is.

    Args:
        smiles: Canonical SMILES string.
        slicer: Fragmentation method ("brics" or "recap").

    Returns:
        SAFE-encoded string, or the original SMILES on failure.
    """
    try:
        return safe.encode(smiles, canonical=True, slicer=slicer, ignore_stereo=True)
    except (safe.SAFEEncodeError, safe.SAFEFragmentationError):
        try:
            # If SAFE encoding fails, at least return a canonical SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception:
            pass  # If canonicalization also fails, return original SMILES
        return smiles


def encode_safe_batch(smiles_list: List[str], slicer: str = "brics") -> List[str]:
    """
    Convert a list of SMILES strings to SAFE representations.

    Molecules that fail SAFE encoding are kept as their original SMILES.

    Args:
        smiles_list: List of canonical SMILES strings.
        slicer: Fragmentation method ("brics" or "recap").

    Returns:
        List of SAFE-encoded strings (same length as input).
    """
    return [encode_safe(smi, slicer=slicer) for smi in smiles_list]
