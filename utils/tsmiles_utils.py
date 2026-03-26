"""
Utility functions for t-SMILES encoding.

Seven variants are supported:
    TSSA  – atom-mapped BFS SMILES (non-dummy); uses BRICS or MMPA slicer.
    TSDY  – dummy-atom BFS SMILES; uses BRICS_DY or MMPA_DY slicer.
    TSID  – dummy-atom BFS SMARTS; uses BRICS_DY or MMPA_DY slicer.
    TSIS  – attachment-point BFS SMARTS (amt, BFS order); uses BRICS_DY or MMPA_DY slicer.
    TSISD – attachment-point DFS SMARTS (amt, DFS order); uses BRICS_DY or MMPA_DY slicer.
    TSISO – TSIS fragments sorted by length (longest first); uses BRICS_DY or MMPA_DY slicer.
    TSISR – TSIS fragments in random order; uses BRICS_DY or MMPA_DY slicer.
"""

import logging
import random
from typing import List, Literal

import rdkit.Chem as Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from t_smiles.dataset.std_tokens import CTokens, STDTokens_Frag_File
from t_smiles.dataset.graph.cnjt_mol import CNJTMolTree
from t_smiles.dataset.graph.cnj_mol_util import CNJMolUtil
from t_smiles.mol_utils.rdk_utils.frag.rdk_frag_util import Fragment_Alg


logger = logging.getLogger(__name__)

Slicer = Literal["brics", "mmpa"]
Variant = Literal["TSSA", "TSDY", "TSID", "TSIS", "TSISD", "TSISO", "TSISR"]

# Slicer + variant → Fragment_Alg mapping.
# TSSA needs non-DY algs; TSDY/TSID/TSIS need DY algs.
_ALG_MAP: dict[tuple[str, str], Fragment_Alg] = {
    ("brics", "TSSA"): Fragment_Alg.BRICS_Base,
    ("brics", "TSDY"): Fragment_Alg.BRICS_DY,
    ("brics", "TSID"): Fragment_Alg.BRICS_DY,
    ("brics", "TSIS"): Fragment_Alg.BRICS_DY,
    ("brics", "TSISD"): Fragment_Alg.BRICS_DY,
    ("brics", "TSISO"): Fragment_Alg.BRICS_DY,
    ("brics", "TSISR"): Fragment_Alg.BRICS_DY,
    ("mmpa", "TSSA"): Fragment_Alg.MMPA,
    ("mmpa", "TSDY"): Fragment_Alg.MMPA_DY,
    ("mmpa", "TSID"): Fragment_Alg.MMPA_DY,
    ("mmpa", "TSIS"): Fragment_Alg.MMPA_DY,
    ("mmpa", "TSISD"): Fragment_Alg.MMPA_DY,
    ("mmpa", "TSISO"): Fragment_Alg.MMPA_DY,
    ("mmpa", "TSISR"): Fragment_Alg.MMPA_DY,
}

_CTOKEN = CTokens(STDTokens_Frag_File(None))


def _standardize(smiles: str) -> str:
    """Return RDKit-standardised canonical SMILES, or raise on failure."""
    return rdMolStandardize.StandardizeSmiles(smiles)


def _encode_single_component(smiles: str, dec_alg: Fragment_Alg) -> CNJTMolTree:
    """Build a CNJTMolTree for one component (no '.' in input)."""
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"RDKit cannot parse SMILES component: {smiles!r}")
    return CNJTMolTree(smiles, ctoken=_CTOKEN, dec_alg=dec_alg)


def _extract_variant(cnjtmol: CNJTMolTree, variant: Variant) -> str:
    """Extract the requested variant string from a successfully-built CNJTMolTree."""
    if variant == "TSSA":
        # TSDY BFS SMILES combined with combine_ex_smiles
        joined, _ = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smiles)
        return joined
    elif variant == "TSDY":
        # Same as TSSA but built with a DY alg – combine_ex_smiles over bfs_ex_smiles
        joined, _ = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smiles)
        return joined
    elif variant == "TSID":
        # BFS SMARTS combined
        joined, _ = CNJMolUtil.combine_ex_smiles(cnjtmol.bfs_ex_smarts)
        return joined
    elif variant == "TSIS":
        # Attachment-point BFS SMARTS (amt_bfs_smarts), fragments joined by '^'
        return cnjtmol.amt_bfs_smarts
    elif variant == "TSISD":
        # Attachment-point DFS SMARTS (amt_dfs_smarts), fragments joined by '^'
        return cnjtmol.amt_dfs_smarts
    elif variant == "TSISO":
        # TSIS fragments sorted by length descending, joined by '^'
        frags = cnjtmol.amt_bfs_smarts.split("^")
        frags_sorted = sorted(frags, key=len, reverse=True)
        return "^".join(frags_sorted)
    elif variant == "TSISR":
        # TSIS fragments in random order, joined by '^'
        frags = cnjtmol.amt_bfs_smarts.split("^")
        random.shuffle(frags)
        return "^".join(frags)
    else:
        raise ValueError(
            f"Unknown variant: {variant!r}. " f"Must be one of TSSA, TSDY, TSID, TSIS, TSISD, TSISO, TSISR."
        )


def encode_tsmiles(smiles: str, slicer: Slicer = "brics", variant: Variant = "TSDY") -> str:
    """
    Convert a SMILES string to a t-SMILES representation.

    The input SMILES is first standardised with RDKit before encoding.
    Multi-component SMILES (containing '.') are split and each component
    is encoded independently; results are joined with '.'.

    If any step fails the original (unstandardised) SMILES is returned.

    Args:
        smiles:  Input SMILES string.
        slicer:  Fragmentation algorithm: "brics" or "mmpa".
        variant: t-SMILES variant: "TSSA", "TSDY", "TSID", "TSIS", "TSISD", "TSISO", or "TSISR".

    Returns:
        t-SMILES string, or the original SMILES on failure.
    """
    key = (slicer, variant)
    if key not in _ALG_MAP:
        raise ValueError(
            f"Unsupported slicer={slicer!r} / variant={variant!r}. "
            f"Valid slicers: 'brics', 'mmpa'. "
            f"Valid variants: 'TSSA', 'TSDY', 'TSID', 'TSIS', 'TSISD', 'TSISO', 'TSISR'."
        )
    dec_alg = _ALG_MAP[key]

    try:
        std_smiles = _standardize(smiles)
    except Exception as exc:
        logger.warning(
            "t-SMILES standardization failed for SMILES=%r. Returning original. Error: %s",
            smiles,
            exc,
        )
        return smiles

    components = std_smiles.split(".")
    encoded_parts: List[str] = []

    for component in components:
        if not component:
            continue
        try:
            cnjtmol = _encode_single_component(component, dec_alg)

            if cnjtmol.mol is None:
                raise RuntimeError(f"CNJTMolTree returned mol=None for component {component!r}")

            part = _extract_variant(cnjtmol, variant)
            if not part:
                raise RuntimeError(f"Empty {variant} result for component {component!r}")

            encoded_parts.append(part.strip())

        except Exception as exc:
            logger.warning(
                "t-SMILES encoding failed for component=%r (slicer=%s, variant=%s). "
                "Returning original SMILES. Error: %s",
                component,
                slicer,
                variant,
                exc,
            )
            return smiles

    if not encoded_parts:
        logger.warning(
            "t-SMILES produced no output for SMILES=%r (slicer=%s, variant=%s). " "Returning original SMILES.",
            smiles,
            slicer,
            variant,
        )
        return smiles

    return ".".join(encoded_parts)


def encode_tsmiles_batch(
    smiles_list: List[str],
    slicer: Slicer = "brics",
    variant: Variant = "TSDY",
) -> List[str]:
    """Convert a list of SMILES strings to t-SMILES representations."""
    return [encode_tsmiles(smi, slicer=slicer, variant=variant) for smi in smiles_list]
