"""One molecular partition, independently serialized as fragSMILES/TSID/TSDY.

The cut SMARTS is chemicalgof's default, including its charge-pair exception.
No salt stripping, tautomer standardization, or representation-specific re-cutting
is performed. Errors are raised, never replaced by plain SMILES.
"""

from collections import deque
from dataclasses import dataclass
import re

import networkx as nx
from rdkit import Chem

EXOCYCLIC_SMARTS = "[!$([+1,-1]~[-1,+1])]-&!@[*]"
SHARED_VERSION = 1
_CUT_PATTERN = Chem.MolFromSmarts(EXOCYCLIC_SMARTS)


@dataclass
class FragmentPartition:
    mol: Chem.Mol
    groups: tuple[tuple[int, ...], ...]
    cut_bonds: tuple[int, ...]

    def graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.groups)))
        owner = {atom: i for i, group in enumerate(self.groups) for atom in group}
        for idx in self.cut_bonds:
            bond = self.mol.GetBondWithIdx(idx)
            left, right = owner[bond.GetBeginAtomIdx()], owner[bond.GetEndAtomIdx()]
            if left == right or graph.has_edge(left, right):
                raise ValueError("Partition contains an internal cut or multiple inter-fragment bonds")
            graph.add_edge(left, right, bond_index=idx)
        return graph

    def validate(self):
        atoms = [a for group in self.groups for a in group]
        if sorted(atoms) != list(range(self.mol.GetNumAtoms())):
            raise ValueError("Fragments must partition the original atoms exactly once")
        allowed = eligible_cuts(self.mol)
        if not set(self.cut_bonds) <= allowed:
            raise ValueError("A cut is not a supported exocyclic single bond")
        owner = {a: i for i, group in enumerate(self.groups) for a in group}
        boundary = {b.GetIdx() for b in self.mol.GetBonds() if owner[b.GetBeginAtomIdx()] != owner[b.GetEndAtomIdx()]}
        if boundary != set(self.cut_bonds):
            raise ValueError("Cut bonds disagree with fragment membership")
        if not nx.is_forest(self.graph()):
            raise ValueError("Shared fragment graph must be a forest")


def eligible_cuts(mol):
    return {mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in mol.GetSubstructMatches(_CUT_PATTERN)}


def partition_from_cuts(mol, cuts):
    graph = nx.Graph()
    graph.add_nodes_from(range(mol.GetNumAtoms()))
    cuts = set(cuts)
    graph.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if b.GetIdx() not in cuts)
    groups = tuple(sorted(tuple(sorted(c)) for c in nx.connected_components(graph)))
    result = FragmentPartition(mol, groups, tuple(sorted(cuts)))
    result.validate()
    return result


def partition_molecule(smiles, merge_model=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")
    if any(a.GetAtomicNum() == 0 or a.GetAtomMapNum() for a in mol.GetAtoms()):
        raise ValueError("Shared fragmentation expects molecules without dummy atoms or atom maps")
    # Canonicalize once for identical atom order across every serializer.
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))
    partition = partition_from_cuts(mol, eligible_cuts(mol))
    return merge_model.apply(partition) if merge_model is not None else partition


def component_partitions(partition):
    mappings = []
    mols = Chem.GetMolFrags(partition.mol, asMols=True, fragsMolAtomMapping=mappings)
    for mol, mapping in zip(mols, mappings):
        old_to_new = {old: new for new, old in enumerate(mapping)}
        cuts = []
        for idx in partition.cut_bonds:
            bond = partition.mol.GetBondWithIdx(idx)
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in old_to_new:
                cuts.append(mol.GetBondBetweenAtoms(old_to_new[a], old_to_new[b]).GetIdx())
        yield partition_from_cuts(mol, cuts)


def serialize_fragsmiles(partition):
    """Use chemicalgof 0.2's assembly machinery with explicit shared cuts.

    Mirrors Reduce2GoF after bond selection, without re-fragmenting the molecule.
    The existing dependency is intentionally retained; this is not a 0.3 adapter.
    """
    from chemicalgof import Reduce2GoF
    from chemicalgof.reduce import Decompositer
    from chemicalgof.gof import DiGraphFrags, FragNode
    from chemicalgof.write import GoF2fragSMILES

    encoded = []
    for part in component_partitions(partition):
        mol = part.mol
        if not part.cut_bonds:
            graph = Reduce2GoF(mol=mol, capitalize_legacy=True)
        else:
            pairs = [
                (mol.GetBondWithIdx(i).GetBeginAtomIdx(), mol.GetBondWithIdx(i).GetEndAtomIdx()) for i in part.cut_bonds
            ]
            decomposer = Decompositer(EXOCYCLIC_SMARTS)
            fragments = decomposer.fragment(mol, pairs)
            neighbors = decomposer.frag_bonds(fragments, pairs)
            strings = decomposer.ultimate_smiles(fragments)
            chiral = {a: label.upper() for a, label in Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)}
            attributes = decomposer.setup_nodes_attributes(strings, chiral, {a for p in pairs for a in p})
            graph = DiGraphFrags()
            nodes = [FragNode(smiles=s, chirality=c) for s, c in zip(strings, attributes)]
            graph.add_nodes_from(nodes)
            for mapping, node, links in zip(decomposer.mapsMol2Frag, nodes, neighbors):
                for atom, neighbor in links:
                    graph.add_edge(
                        node,
                        nodes[decomposer.fragsIdxs[neighbor]],
                        aB=mapping[atom],
                        stereo=chiral.get(atom) if node.numPotAtomLinkers > 1 else None,
                    )
        result = GoF2fragSMILES(graph, canonize=True)
        if not isinstance(result, str) or not result:
            raise ValueError("chemicalgof failed to serialize the shared fragment tree")
        encoded.append(result)
    return ";".join(encoded)


def decorated_fragments(partition):
    """Create TSID fragments with a paired isotope ID per cut bond.

    Returns fragments in partition order. RDKit keeps attachment stereochemistry
    when inserting the dummy atoms. Original molecular isotope labels are retained.
    """
    if partition.cut_bonds:
        labels = [(i, i) for i in range(1, len(partition.cut_bonds) + 1)]
        fragmented = Chem.FragmentOnBonds(partition.mol, list(partition.cut_bonds), dummyLabels=labels)
    else:
        fragmented = Chem.Mol(partition.mol)
    mappings = []
    fragments = Chem.GetMolFrags(fragmented, asMols=True, fragsMolAtomMapping=mappings)
    by_atoms = {
        tuple(sorted(a for a in ids if a < partition.mol.GetNumAtoms())): mol for ids, mol in zip(mappings, fragments)
    }
    return [Chem.MolToSmiles(by_atoms[group], isomericSmiles=True) for group in partition.groups]


def binary_tree_tokens(graph, fragments):
    """t-SMILES first-child/next-sibling FBT, BFS, with upstream extra root dummy.

    Canonical atom order chooses the root; children are sorted by fragment index.
    This implements CNJTMolTree.create_bfs_binary_tree_ex/get_bfs_ex without
    invoking MolTree's BRICS/MMPA fragmentation or vocabulary-dependent repairs.
    """
    if not nx.is_tree(graph):
        raise ValueError("A connected tree is required for TSID/TSDY serialization")
    root = min(graph.nodes)
    children = {node: [] for node in graph}
    for parent, child in nx.bfs_edges(graph, root, sort_neighbors=sorted):
        children[parent].append(child)
    left, right = {}, {}
    for parent, nodes in children.items():
        if nodes:
            left[parent] = nodes[0]
            right.update(zip(nodes, nodes[1:]))
    queue = deque([root])
    tokens = []
    while queue:
        node = queue.popleft()
        tokens.append("&" if node is None else fragments[node])
        if node is not None:
            queue.extend((left.get(node), right.get(node)))
    tokens.insert(1, "&")
    return tokens


def serialize_tsmiles(partition, variant="TSID"):
    if variant not in ("TSID", "TSDY"):
        raise ValueError("Shared exocyclic fragmentation supports TSID and TSDY only")
    components = []
    for part in component_partitions(partition):
        fragments = decorated_fragments(part)
        if variant == "TSDY":
            # Keep identical fragment order and stereochemistry; remove only dummy IDs.
            fragments = [re.sub(r"\[\d+\*\]", "*", s) for s in fragments]
        tokens = binary_tree_tokens(part.graph(), fragments)
        result = tokens[0]
        for previous, token in zip(tokens, tokens[1:]):
            result += ("^" if previous != "&" and token != "&" else "") + token
        components.append(result)
    return ".".join(components)


def decode_tsid(text):
    """Reassemble our TSID strings by paired dummy IDs for round-trip audits.

    Attachment IDs recover connectivity independently of the FBT traversal.
    This is not a TSDY decoder and does not validate arbitrary generated FBTs.
    """
    molecules = []
    for component in text.split("."):
        fragments = [s for s in re.split(r"[&^]", component) if s]
        mol = Chem.MolFromSmiles(".".join(fragments))
        if mol is None:
            raise ValueError("Invalid TSID fragment string")
        ids = [a.GetIsotope() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
        from collections import Counter

        if any(i == 0 or count != 2 for i, count in Counter(ids).items()):
            raise ValueError("TSID attachment IDs must occur exactly twice")
        params = Chem.MolzipParams()
        params.label = Chem.MolzipLabel.Isotope
        result = Chem.molzip(mol, params)
        Chem.SanitizeMol(result)
        molecules.append(Chem.MolToSmiles(result, isomericSmiles=True))
    return Chem.CanonSmiles(".".join(molecules))


def encode_shared(smiles, representation, merge_model=None):
    partition = partition_molecule(smiles, merge_model)
    if representation == "fragsmiles":
        return serialize_fragsmiles(partition)
    return serialize_tsmiles(partition, representation)
