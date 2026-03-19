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

import logging
import safe
from typing import List
from rdkit import Chem


logger = logging.getLogger(__name__)


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
    except (safe.SAFEEncodeError, safe.SAFEFragmentationError) as exc:
        logger.warning(
            "SAFE encoding failed for SMILES=%r (slicer=%s). Falling back to canonical SMILES. Error: %s",
            smiles,
            slicer,
            exc,
        )
        try:
            # If SAFE encoding fails, at least return a canonical SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except Exception as canon_exc:
            logger.warning(
                "RDKit canonicalization fallback also failed for SMILES=%r. Returning original SMILES. Error: %s",
                smiles,
                canon_exc,
            )
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
