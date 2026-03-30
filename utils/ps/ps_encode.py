#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PS-VAE-based SMILES -> fragSMILES conversion helpers."""

from __future__ import annotations

from typing import Optional

from rdkit import Chem

from chemicalgof.gof import DiGraphFrags, FragNode
from chemicalgof.utils import GetPotAtomLinkers
from chemicalgof.write import GoF2fragSMILES

from utils.ps.utils.chem_utils import MAX_VALENCE, mol2smi
from utils.ps.molecule import SubgraphNode, SubgraphEdge, Molecule
from utils.ps.mol_bpe_ring import MolInSubgraph, Tokenizer


def _get_subgraph_node(ps_molecule: Molecule, node_idx: int) -> SubgraphNode:
    return ps_molecule.nodes[node_idx]["subgraph"]


def _get_subgraph_edge(ps_molecule: Molecule, src_idx: int, dst_idx: int) -> SubgraphEdge:
    return ps_molecule.edges[src_idx, dst_idx]["connects"]


def _safe_potential_linkers(subgraph: SubgraphNode) -> list[int]:
    """Best-effort list of potential atom-linker indices for one subgraph."""
    try:
        linkers = GetPotAtomLinkers(subgraph.smiles)
        if linkers:
            return linkers
    except Exception:
        pass

    mol = getattr(subgraph, "mol", None)
    if mol is None:
        try:
            mol = Chem.MolFromSmiles(subgraph.smiles, sanitize=False)
        except Exception:
            mol = None

    if mol is None:
        return []

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    try:
        linkers = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetTotalNumHs() > 0]
    except Exception:
        linkers = []

    return linkers


def PSReduce2GoF(
    ps_molecule: Molecule,
    smiles: Optional[str] = None,
    capitalize_legacy: bool = False,
) -> DiGraphFrags:
    """Convert a PS-VAE tokenized molecule into a chemicalgof reduced graph."""
    if isinstance(ps_molecule, list):
        raise ValueError(
            "PSReduce2GoF expects one PS-VAE Molecule. "
            "For disconnected inputs (dot-separated SMILES), tokenize each fragment separately."
        )

    mol_smiles = None
    if hasattr(ps_molecule, "graph"):
        mol_smiles = ps_molecule.graph.get("smiles")
    if not mol_smiles:
        mol_smiles = smiles
    if not mol_smiles:
        raise ValueError("Input Error: a PS-VAE Molecule with graph['smiles'] or smiles is required.")

    mol = Chem.MolFromSmiles(Chem.CanonSmiles(mol_smiles))
    if mol is None:
        raise ValueError("Input Error: invalid SMILES provided for chirality extraction.")

    all_chiral_atoms = Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)
    if not capitalize_legacy:
        all_chiral_atoms = dict(all_chiral_atoms)
    else:
        all_chiral_atoms = {idx: cip.upper() for idx, cip in all_chiral_atoms}

    local2orig: dict[int, dict[int, int]] = {}
    for node_idx in ps_molecule.nodes:
        subgraph = _get_subgraph_node(ps_molecule, node_idx)
        local2orig[node_idx] = {local: orig for orig, local in subgraph.atom_mapping.items()}

    inter_atoms_orig: dict[int, set[int]] = {node_idx: set() for node_idx in ps_molecule.nodes}
    for src_idx, dst_idx in ps_molecule.edges:
        edge = _get_subgraph_edge(ps_molecule, src_idx, dst_idx)
        if edge.dummy:
            continue
        for begin_local, end_local, _ in edge.edges:
            inter_atoms_orig[src_idx].add(local2orig[src_idx][begin_local])
            inter_atoms_orig[dst_idx].add(local2orig[dst_idx][end_local])

    diG = DiGraphFrags()
    node_by_idx: dict[int, FragNode] = {}

    for node_idx in ps_molecule.nodes:
        subgraph = _get_subgraph_node(ps_molecule, node_idx)
        pot_linkers = _safe_potential_linkers(subgraph)
        n_atom_linkers = len(pot_linkers)
        chirality: dict[int, str] = {}

        for orig_idx, local_idx in sorted(subgraph.atom_mapping.items(), key=lambda x: x[1]):
            if orig_idx in all_chiral_atoms and (orig_idx not in inter_atoms_orig[node_idx] or n_atom_linkers == 1):
                chirality[local_idx] = all_chiral_atoms[orig_idx]

        frag_mol = subgraph.get_mol()
        Chem.RemoveStereochemistry(frag_mol)
        frag = FragNode(smiles=mol2smi(frag_mol, kekulize=ps_molecule.kekulize), chirality=chirality)
        frag.PotAtomLinkers = pot_linkers
        node_by_idx[node_idx] = frag

    diG.add_nodes_from(node_by_idx.values())

    for src_idx, dst_idx in ps_molecule.edges:
        edge = _get_subgraph_edge(ps_molecule, src_idx, dst_idx)
        if edge.dummy or not edge.edges:
            continue

        src_node = node_by_idx[src_idx]
        dst_node = node_by_idx[dst_idx]

        begin_local, end_local, _ = edge.edges[0]

        src_orig = local2orig[src_idx][begin_local]
        src_stereo = all_chiral_atoms.get(src_orig) if src_node.numPotAtomLinkers > 1 else None
        diG.add_edge(src_node, dst_node, aB=begin_local, stereo=src_stereo)

        dst_orig = local2orig[dst_idx][end_local]
        dst_stereo = all_chiral_atoms.get(dst_orig) if dst_node.numPotAtomLinkers > 1 else None
        diG.add_edge(dst_node, src_node, aB=end_local, stereo=dst_stereo)

    return diG


class PSTokenizer:
    """Reusable encoder that keeps one loaded PS-VAE vocabulary in memory."""

    def __init__(self, vocab_path: str):
        self.vocab_path = vocab_path
        self._tokenizer = Tokenizer(vocab_path)

    def tokenize(self, smiles: str):
        return self._tokenizer.tokenize(smiles)

    def reduce_to_gof(self, smiles: str, capitalize_legacy: bool = True) -> DiGraphFrags:
        ps_molecule = self._tokenizer.tokenize(smiles)
        if isinstance(ps_molecule, list):
            raise ValueError(
                "Disconnected molecules are not supported in one call. "
                "Please provide one connected SMILES at a time."
            )
        return PSReduce2GoF(ps_molecule, smiles=smiles, capitalize_legacy=capitalize_legacy)

    def encode(
        self,
        smiles: str,
        canonize: bool = True,
        random: bool = False,
        capitalize_legacy: bool = True,
    ) -> str:
        # PSReduce2GoF only supports one connected molecule at a time.
        # For disconnected inputs, encode each component independently and
        # join with ';' (same convention used in fragSMILES utilities).
        parts = [part for part in smiles.split(".") if part]
        encoded_parts = []
        for part in parts:
            dig = self.reduce_to_gof(part, capitalize_legacy=capitalize_legacy)
            encoded_parts.append(GoF2fragSMILES(dig, canonize=canonize, random=random))
        return ";".join(encoded_parts)

    def __call__(
        self,
        smiles: str,
        canonize: bool = True,
        random: bool = False,
        capitalize_legacy: bool = True,
    ) -> str:
        return self.encode(
            smiles,
            canonize=canonize,
            random=random,
            capitalize_legacy=capitalize_legacy,
        )


def ps_encode(
    smiles: str,
    vocab_path: str,
    canonize: bool = True,
    random: bool = False,
    capitalize_legacy: bool = True,
) -> str:
    """One-shot helper: SMILES -> PS-VAE reduced graph -> fragSMILES."""
    tokenizer = PSTokenizer(vocab_path)
    return tokenizer.encode(
        smiles,
        canonize=canonize,
        random=random,
        capitalize_legacy=capitalize_legacy,
    )


__all__ = [
    "MAX_VALENCE",
    "SubgraphNode",
    "SubgraphEdge",
    "Molecule",
    "MolInSubgraph",
    "Tokenizer",
    "PSReduce2GoF",
    "PSTokenizer",
    "ps_encode",
]
