"""
Utility functions for fragSMILES encoding.

fragSMILES strings are generated from SMILES using chemicalgof.encode.
These helpers are used by dataset loaders to precompute and cache corpora,
so tokenizer calls only need chemicalgof.split at runtime.
"""

import logging
from typing import List

from rdkit import Chem


logger = logging.getLogger(__name__)


def encode_fragsmiles(smiles: str) -> str:
    """
    Convert a SMILES string to canonical fragSMILES.

    If conversion fails, returns canonical SMILES when possible,
    otherwise returns the original input string.
    """
    import chemicalgof

    try:
        return chemicalgof.encode(smiles, canonical=True)
    except Exception as exc:
        logger.warning(
            "fragSMILES encoding failed for SMILES=%r. Falling back to canonical SMILES. Error: %s", smiles, exc
        )
        try:
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


def encode_fragsmiles_batch(smiles_list: List[str]) -> List[str]:
    """Convert a list of SMILES strings to fragSMILES strings."""
    return [encode_fragsmiles(smi) for smi in smiles_list]
