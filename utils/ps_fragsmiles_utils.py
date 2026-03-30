"""
Utility functions for PS-fragSMILES encoding.

PS-fragSMILES strings are generated from SMILES by:
1) PS-VAE principal-subgraph decomposition
2) reduction to chemicalgof graph
3) conversion to fragSMILES
"""

import logging
from typing import List

from rdkit import Chem

from utils.ps.ps_encode import PSTokenizer


logger = logging.getLogger(__name__)


def encode_ps_fragsmiles(smiles: str, vocab_path: str) -> str:
    """
    Convert a SMILES string to canonical PS-fragSMILES.

    If conversion fails, returns canonical SMILES when possible,
    otherwise returns the original input string.
    """
    try:
        tokenizer = PSTokenizer(vocab_path)
        return tokenizer.encode(smiles, canonize=True, random=False, capitalize_legacy=True)
    except Exception as exc:
        logger.warning(
            "PS-fragSMILES encoding failed for SMILES=%r. Falling back to canonical SMILES. Error: %s",
            smiles,
            exc,
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


def encode_ps_fragsmiles_batch(smiles_list: List[str], vocab_path: str) -> List[str]:
    """Convert a list of SMILES strings to PS-fragSMILES strings."""
    return [encode_ps_fragsmiles(smi, vocab_path) for smi in smiles_list]
