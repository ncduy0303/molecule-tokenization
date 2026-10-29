"""Experimental clean-core TSDY codec. Independent of frozen training formats.

JSON is a transport format; model_tokens emits indivisible cores and separate
compositional metadata. No original SMILES or hidden atom mapping is needed by
the decoder. Stereo is relative to explicitly recorded neighbour identities.
"""

import json
from collections import Counter
import networkx as nx
from rdkit import Chem
from utils.shared_fragmentation import partition_molecule, component_partitions, binary_tree_tokens

FORMAT = "external-tsdy-v1"


def _sanitize_core(mol):
    """Validate caps without normalizing away source bond/charge information.

    Capping can make hypervalent iodine eligible for cleanup, or make a
    previously nonaromatic ring aromatic. Those normalizations are not
    reversible when the caps are replaced with the original attachments.
    Run normal valence, radical and kekulization validation, then restore the
    source bond representation. Skipping kekulization entirely would change
    the computed state of ordinary aromatic sulfur cores.
    """
    bonds = [(b.GetBondType(), b.GetIsAromatic()) for b in mol.GetBonds()]
    atoms = [a.GetIsAromatic() for a in mol.GetAtoms()]
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_CLEANUP,
    )
    for bond, (kind, aromatic) in zip(mol.GetBonds(), bonds):
        bond.SetBondType(kind)
        bond.SetIsAromatic(aromatic)
    for atom, aromatic in zip(mol.GetAtoms(), atoms):
        atom.SetIsAromatic(aromatic)
    mol.UpdatePropertyCache(strict=True)


def parse_core(text):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mol = Chem.MolFromSmiles(text, params)
    if mol is None or any(a.GetAtomicNum() == 0 or a.GetAtomMapNum() for a in mol.GetAtoms()):
        raise ValueError("Invalid clean fragment")
    _sanitize_core(mol)
    return mol


def _component(part):
    mol = part.mol
    if any(
        b.GetBondType()
        not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
        for b in mol.GetBonds()
    ):
        raise ValueError("Nonstandard bond chemistry is outside prototype scope")
    cores, addresses = [], {}
    cut_degree = Counter()
    for i in part.cut_bonds:
        b = mol.GetBondWithIdx(i)
        cut_degree.update((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    for fi, group in enumerate(part.groups):
        rw = Chem.RWMol(mol)
        Chem.RemoveStereochemistry(rw)
        for a in rw.GetAtoms():
            a.SetIntProp("_source", a.GetIdx())
            a.SetNumExplicitHs(a.GetTotalNumHs() + cut_degree[a.GetIdx()])
            a.SetNoImplicit(True)
        for i in reversed(range(mol.GetNumAtoms())):
            if i not in group:
                rw.RemoveAtom(i)
        core = rw.GetMol()
        _sanitize_core(core)
        text = Chem.MolToSmiles(core, canonical=True, isomericSmiles=True)
        order = list(core.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"])
        parsed = parse_core(text)
        if parsed.GetNumAtoms() != len(group):
            raise ValueError("Capping changed source atom count")
        for local, i in enumerate(order):
            addresses[core.GetAtomWithIdx(i).GetIntProp("_source")] = [fi, local]
        cores.append(text)
    links = []
    for parent, child in nx.bfs_edges(part.graph(), 0, sort_neighbors=sorted):
        b = mol.GetBondWithIdx(part.graph()[parent][child]["bond_index"])
        a, z = addresses[b.GetBeginAtomIdx()], addresses[b.GetEndAtomIdx()]
        if a[0] != parent:
            a, z = z, a
        links.append([a, z, "SINGLE"])
    stereo_atoms, stereo_bonds = [], []
    for a in mol.GetAtoms():
        tag = a.GetChiralTag()
        if tag == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        if tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            raise ValueError("Only tetrahedral atom stereo supported")
        stereo_atoms.append([addresses[a.GetIdx()], int(tag), [addresses[n.GetIdx()] for n in a.GetNeighbors()]])
    for b in mol.GetBonds():
        if b.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        if b.GetStereo() not in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            raise ValueError("Only specified E/Z bond stereo supported")
        stereo_bonds.append(
            [
                addresses[b.GetBeginAtomIdx()],
                addresses[b.GetEndAtomIdx()],
                int(b.GetStereo()),
                [addresses[i] for i in b.GetStereoAtoms()],
            ]
        )
    tree = [None if x == "&" else int(x) for x in binary_tree_tokens(part.graph(), [str(i) for i in range(len(cores))])]
    return dict(cores=cores, tree=tree, links=links, atom_stereo=stereo_atoms, bond_stereo=stereo_bonds)


def molecules_equal(left, right):
    """Canonical isomeric equality, with an exact achiral graph fallback.

    Large symmetric graphs can have differing RDKit canonical strings. Never
    waive stereo checks: the fallback is restricted to unspecified stereo.
    """
    a, b = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
    if a is None or b is None:
        return False
    if Chem.MolToSmiles(a) == Chem.MolToSmiles(b):
        return True
    if a.GetNumAtoms() != b.GetNumAtoms() or a.GetNumBonds() != b.GetNumBonds():
        return False
    for mol in (a, b):
        if any(x.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for x in mol.GetAtoms()):
            return False
        if any(x.GetStereo() != Chem.BondStereo.STEREONONE for x in mol.GetBonds()):
            return False
    # RDKit can assign different aromatic flags to the same bridged-ring
    # graph depending on atom order. Accept only an identical explicit-bond
    # graph after kekulization, and only within the achiral scope above.
    ka, kb = Chem.Mol(a), Chem.Mol(b)
    try:
        Chem.Kekulize(ka, clearAromaticFlags=True)
        Chem.Kekulize(kb, clearAromaticFlags=True)
        if Chem.MolToSmiles(ka, isomericSmiles=True) == Chem.MolToSmiles(kb, isomericSmiles=True):
            return True
    except (ValueError, RuntimeError):
        pass
    mapping = b.GetSubstructMatch(a, useChirality=True)
    if not mapping:
        return False

    def properties(x):
        return (
            x.GetAtomicNum(),
            x.GetIsotope(),
            x.GetFormalCharge(),
            x.GetNumRadicalElectrons(),
            x.GetTotalNumHs(),
            x.GetIsAromatic(),
        )

    return all(properties(x) == properties(b.GetAtomWithIdx(mapping[x.GetIdx()])) for x in a.GetAtoms()) and all(
        x.GetBondType() == b.GetBondBetweenAtoms(mapping[x.GetBeginAtomIdx()], mapping[x.GetEndAtomIdx()]).GetBondType()
        for x in a.GetBonds()
    )


def encode(smiles):
    part = partition_molecule(smiles)
    doc = dict(format=FORMAT, components=[_component(p) for p in component_partitions(part)])
    text = json.dumps(doc, separators=(",", ":"))
    # Fail closed even on unusual supported-looking chemistry.
    if not molecules_equal(decode(text), Chem.MolToSmiles(part.mol, isomericSmiles=True)):
        raise ValueError("Reconstruction mismatch; molecule not supported by prototype")
    return text


def _decode_component(c):
    cores = [parse_core(s) for s in c["cores"]]
    if not cores:
        raise ValueError("Empty component")
    offsets, combined = [], Chem.Mol()
    for core in cores:
        offsets.append(combined.GetNumAtoms())
        combined = Chem.CombineMols(combined, core)

    def address(pair):
        if not isinstance(pair, list) or len(pair) != 2 or any(type(i) is not int for i in pair):
            raise ValueError("Invalid atom address")
        f, a = pair
        if not 0 <= f < len(cores) or not 0 <= a < cores[f].GetNumAtoms():
            raise ValueError("Atom address out of range")
        return offsets[f] + a

    graph = nx.Graph()
    graph.add_nodes_from(range(len(cores)))
    rw = Chem.RWMol(combined)
    consumed = Counter()
    for left, right, kind in c["links"]:
        a, b = address(left), address(right)
        if kind != "SINGLE" or left[0] == right[0] or graph.has_edge(left[0], right[0]):
            raise ValueError("Invalid or duplicate fragment edge")
        graph.add_edge(left[0], right[0])
        rw.AddBond(a, b, Chem.BondType.SINGLE)
        consumed.update((a, b))
    if not nx.is_tree(graph):
        raise ValueError("Component must be a tree")
    expected = [None if x == "&" else int(x) for x in binary_tree_tokens(graph, [str(i) for i in range(len(cores))])]
    if c["tree"] != expected:
        raise ValueError("Tree and connection records disagree")
    for a in rw.GetAtoms():
        h = combined.GetAtomWithIdx(a.GetIdx()).GetTotalNumHs() - consumed[a.GetIdx()]
        if h < 0:
            raise ValueError("Connections exceed hydrogen caps")
        a.SetNumExplicitHs(h)
        a.SetNoImplicit(True)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    seen = set()
    for target, tag, neighbors in c["atom_stereo"]:
        idx = address(target)
        if idx in seen or tag not in (1, 2):
            raise ValueError("Invalid atom stereo record")
        seen.add(idx)
        old = [address(n) for n in neighbors]
        new = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
        if sorted(old) != sorted(new):
            raise ValueError("Stereo neighbours disagree")
        permutation = [old.index(n) for n in new]
        odd = sum(permutation[i] > permutation[j] for i in range(len(new)) for j in range(i + 1, len(new))) % 2
        atom = mol.GetAtomWithIdx(idx)
        atom.SetChiralTag(Chem.ChiralType(tag))
        if odd:
            atom.InvertChirality()
    for left, right, stereo, neighbors in c["bond_stereo"]:
        a, b = address(left), address(right)
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE or stereo not in (2, 3):
            raise ValueError("Invalid bond stereo")
        n = [address(x) for x in neighbors]
        if bond.GetBeginAtomIdx() != a:
            n.reverse()
        bond.SetStereoAtoms(*n)
        bond.SetStereo(Chem.BondStereo(stereo))
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def decode(text):
    doc = json.loads(text)
    if doc["format"] != FORMAT or not doc["components"]:
        raise ValueError("Unsupported or empty external-TSDY document")
    parts = [_decode_component(c) for c in doc["components"]]
    smiles = ".".join(Chem.MolToSmiles(m, isomericSmiles=True) for m in parts)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid reconstructed molecule")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def verbose_model_tokens(text):
    """Clean cores are words; metadata numbers are digit tokens, never OOV IDs."""
    doc = json.loads(text)
    tokens = ["<EXT_TSDY_V1>"]

    def fields(value):
        if isinstance(value, list):
            tokens.append("<LIST>")
            for item in value:
                fields(item)
            tokens.append("</LIST>")
        elif type(value) is int:
            tokens.extend(["<INT>", *str(value), "</INT>"])
        else:
            tokens.append("<" + value + ">")

    for c in doc["components"]:
        tokens.append("<COMPONENT>")
        child_links = {link[1][0]: link for link in c["links"]}
        for fi in c["tree"]:
            if fi is None:
                tokens.append("<NULL>")
                continue
            tokens.extend(["<FRAGMENT>", c["cores"][fi]])
            fields(fi)
            if fi in child_links:
                tokens.append("<CONNECT>")
                fields(child_links[fi])
            tokens.append("</FRAGMENT>")
        for key in ("atom_stereo", "bond_stereo"):
            tokens.append("<" + key.upper() + ">")
            fields(c[key])
        tokens.append("</COMPONENT>")
    return tokens


def decode_verbose_tokens(tokens):
    """Decode the actual model-token stream, not a hidden JSON side channel."""
    it = iter(tokens)

    def expect(want):
        if next(it, None) != want:
            raise ValueError("Malformed external-TSDY token stream: expected " + want)

    def value(first=None):
        token = next(it) if first is None else first
        if token == "<LIST>":
            result = []
            for token in it:
                if token == "</LIST>":
                    return result
                result.append(value(token))
            raise ValueError("Unclosed list")
        if token == "<INT>":
            digits = []
            for token in it:
                if token == "</INT>":
                    if not digits:
                        raise ValueError("Empty integer")
                    return int("".join(digits))
                if len(token) != 1 or token not in "0123456789":
                    raise ValueError("Invalid digit")
                digits.append(token)
            raise ValueError("Unclosed integer")
        if token == "<SINGLE>":
            return "SINGLE"
        raise ValueError("Invalid metadata token")

    expect("<EXT_TSDY_V1>")
    components = []
    for start in it:
        if start != "<COMPONENT>":
            raise ValueError("Expected component")
        cores, tree, links = {}, [], []
        for token in it:
            if token == "<ATOM_STEREO>":
                break
            if token == "<NULL>":
                tree.append(None)
                continue
            if token != "<FRAGMENT>":
                raise ValueError("Expected fragment")
            core = next(it)
            idx = value()
            if type(idx) is not int or idx in cores:
                raise ValueError("Invalid fragment identity")
            cores[idx] = core
            tree.append(idx)
            end = next(it)
            if end == "<CONNECT>":
                links.append(value())
                expect("</FRAGMENT>")
            elif end != "</FRAGMENT>":
                raise ValueError("Expected fragment end")
        atom_stereo = value()
        expect("<BOND_STEREO>")
        bond_stereo = value()
        expect("</COMPONENT>")
        if sorted(cores) != list(range(len(cores))):
            raise ValueError("Noncontiguous fragment identities")
        components.append(
            dict(
                cores=[cores[i] for i in range(len(cores))],
                tree=tree,
                links=links,
                atom_stereo=atom_stereo,
                bond_stereo=bond_stereo,
            )
        )
    return decode(json.dumps(dict(format=FORMAT, components=components)))


# Fixed-arity records: no nested list syntax or repeated child address.
# Small indices cost one token. Larger indices have a compositional escape.
COMPACT_GRAMMAR = (
    {f"<N{i}>" for i in range(64)}
    | set("0123456789")
    | {"<BIG>", "</BIG>", "<EXT_TSDY_V2>", "<COMPONENT>", "</COMPONENT>", "<F>", "<NULL>", "<LINK>", "<A>", "<B>"}
)


def model_tokens(text):
    doc = json.loads(text)
    if doc["format"] != FORMAT:
        raise ValueError("Unsupported document")
    out = ["<EXT_TSDY_V2>"]

    def number(n):
        if type(n) is not int or n < 0:
            raise ValueError("Invalid nonnegative integer")
        out.extend([f"<N{n}>"] if n < 64 else ["<BIG>", *str(n), "</BIG>"])

    def addr(a):
        for n in a:
            number(n)

    for c in doc["components"]:
        out.append("<COMPONENT>")
        links = {link[1][0]: link for link in c["links"]}
        for fi in c["tree"]:
            if fi is None:
                out.append("<NULL>")
                continue
            out.extend(["<F>", c["cores"][fi]])
            number(fi)
            if fi in links:
                left, right, kind = links[fi]
                if kind != "SINGLE":
                    raise ValueError("Version 2 connections must be single bonds")
                out.append("<LINK>")
                addr(left)
                number(right[1])  # child fragment is the current fragment
        for target, tag, neighbors in c["atom_stereo"]:
            out.append("<A>")
            addr(target)
            number(tag)
            number(len(neighbors))
            for a in neighbors:
                addr(a)
        for left, right, stereo, neighbors in c["bond_stereo"]:
            out.append("<B>")
            addr(left)
            addr(right)
            number(stereo)
            for a in neighbors:
                addr(a)
        out.append("</COMPONENT>")
    return out


def decode_tokens(tokens):
    if tokens and tokens[0] == "<EXT_TSDY_V1>":
        return decode_verbose_tokens(tokens)
    it = iter(tokens)

    def take():
        try:
            return next(it)
        except StopIteration as exc:
            raise ValueError("Truncated compact token stream") from exc

    def number():
        t = take()
        if t in COMPACT_GRAMMAR and t.startswith("<N") and t != "<NULL>":
            return int(t[2:-1])
        if t != "<BIG>":
            raise ValueError("Expected integer token")
        digits = []
        while (t := take()) != "</BIG>":
            if t not in set("0123456789"):
                raise ValueError("Invalid integer escape")
            digits.append(t)
        if not digits:
            raise ValueError("Empty integer escape")
        return int("".join(digits))

    def addr():
        return [number(), number()]

    if take() != "<EXT_TSDY_V2>":
        raise ValueError("Unsupported token format")
    components = []
    for start in it:
        if start != "<COMPONENT>":
            raise ValueError("Expected component")
        cores, tree, links, atoms, bonds = {}, [], [], [], []
        current = None
        while (t := take()) != "</COMPONENT>":
            if t == "<F>":
                core, fi = take(), number()
                if fi in cores:
                    raise ValueError("Duplicate fragment")
                cores[fi] = core
                tree.append(fi)
                current = fi
            elif t == "<NULL>":
                tree.append(None)
                current = None
            elif t == "<LINK>":
                if current is None:
                    raise ValueError("Connection not adjacent to fragment")
                links.append([addr(), [current, number()], "SINGLE"])
                current = None
            elif t == "<A>":
                target, tag, count = addr(), number(), number()
                if count not in (3, 4):
                    raise ValueError("Invalid tetrahedral neighbor count")
                atoms.append([target, tag, [addr() for _ in range(count)]])
                current = None
            elif t == "<B>":
                bonds.append([addr(), addr(), number(), [addr(), addr()]])
                current = None
            else:
                raise ValueError("Invalid compact record")
        if sorted(cores) != list(range(len(cores))):
            raise ValueError("Noncontiguous fragment identities")
        components.append(
            dict(
                cores=[cores[i] for i in range(len(cores))],
                tree=tree,
                links=links,
                atom_stereo=atoms,
                bond_stereo=bonds,
            )
        )
    return decode(json.dumps(dict(format=FORMAT, components=components)))


# V3: implicit fragment IDs and parents from the first-child/next-sibling tree.
model_tokens_v2 = model_tokens
decode_tokens_v2 = decode_tokens
V3_GRAMMAR = (
    {f"<N{i}>" for i in range(64)}
    | set("0123456789")
    | {"<BIG>", "</BIG>", "<NULL>", "<DOT>", "<ACW>", "<ACCW>", "<B>"}
)
COMPACT_GRAMMAR = V3_GRAMMAR


def _sites(core):
    return [a.GetIdx() for a in parse_core(core).GetAtoms() if a.GetTotalNumHs() > 0]


def model_tokens(text):
    doc = json.loads(text)
    if doc["format"] != FORMAT:
        raise ValueError("Unsupported document")
    out = []

    def number(n):
        out.extend([f"<N{n}>"] if 0 <= n < 64 else ["<BIG>", *str(n), "</BIG>"])

    for ci, c in enumerate(doc["components"]):
        if ci:
            out.append("<DOT>")
        # Remove upstream's artificial root placeholder, then trailing nulls.
        tree = c["tree"][:1] + c["tree"][2:]
        while tree and tree[-1] is None:
            tree.pop()
        order = [fi for fi in tree if fi is not None]
        remap = {fi: i for i, fi in enumerate(order)}
        bychild = {right[0]: (left, right) for left, right, kind in c["links"]}

        def addr(a):
            number(remap[a[0]])
            number(a[1])

        for fi in tree:
            if fi is None:
                out.append("<NULL>")
                continue
            out.append(c["cores"][fi])
            if fi in bychild:
                left, right = bychild[fi]
                for endpoint in (left, right):
                    sites = _sites(c["cores"][endpoint[0]])
                    if endpoint[1] not in sites:
                        raise ValueError("Endpoint is not a hydrogen-bearing site")
                    if len(sites) != 1:
                        number(endpoint[1])
        for target, tag, neighbors in c["atom_stereo"]:
            mapped = [(remap[x[0]], x[1]) for x in neighbors]
            rank = [sorted(mapped).index(x) for x in mapped]
            odd = sum(rank[i] > rank[j] for i in range(len(rank)) for j in range(i + 1, len(rank))) % 2
            normalized_tag = 3 - tag if odd else tag
            out.append("<ACW>" if normalized_tag == 1 else "<ACCW>")
            number(remap[target[0]])
            if parse_core(c["cores"][target[0]]).GetNumAtoms() != 1:
                number(target[1])
        for left, right, stereo, neighbors in c["bond_stereo"]:
            out.append("<B>")
            addr(left)
            addr(right)
            number(stereo)
            for a in neighbors:
                addr(a)
    return out


def decode_tokens(tokens):
    if tokens and tokens[0] in ("<EXT_TSDY_V1>", "<EXT_TSDY_V2>"):
        return decode_tokens_v2(tokens)
    from collections import deque

    tokens = list(tokens)
    pos = 0

    def take():
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("Truncated V3 record")
        t = tokens[pos]
        pos += 1
        return t

    def number():
        t = take()
        if t in V3_GRAMMAR and t.startswith("<N") and t != "<NULL>":
            return int(t[2:-1])
        if t != "<BIG>":
            raise ValueError("Expected V3 index")
        ds = []
        while (t := take()) != "</BIG>":
            if t not in set("0123456789"):
                raise ValueError("Invalid index digit")
            ds.append(t)
        if not ds:
            raise ValueError("Empty index")
        return int("".join(ds))

    def addr():
        return [number(), number()]

    components = []
    while pos < len(tokens):
        cores = []
        parents = {}
        graph = nx.Graph()
        links = []
        atoms = []
        bonds = []
        # Each slot is (binary parent, left-child versus right-sibling).
        slots = deque([(None, "root")])
        stereo_started = False
        while pos < len(tokens) and tokens[pos] != "<DOT>":
            token = take()
            if token in ("<ACW>", "<ACCW>", "<B>"):
                stereo_started = True
                if token in ("<ACW>", "<ACCW>"):
                    fi = number()
                    if not 0 <= fi < len(cores):
                        raise ValueError("Invalid stereo fragment")
                    core = parse_core(cores[fi])
                    ai = 0 if core.GetNumAtoms() == 1 else number()
                    if not 0 <= ai < core.GetNumAtoms():
                        raise ValueError("Invalid stereo atom")
                    neighbors = [[fi, n.GetIdx()] for n in core.GetAtomWithIdx(ai).GetNeighbors()]
                    for left, right, kind in links:
                        if left == [fi, ai]:
                            neighbors.append(right)
                        if right == [fi, ai]:
                            neighbors.append(left)
                    atoms.append([[fi, ai], 1 if token == "<ACW>" else 2, sorted(neighbors)])
                else:
                    bonds.append([addr(), addr(), number(), [addr(), addr()]])
                continue
            if stereo_started:
                raise ValueError("Tree token after stereo suffix")
            if not slots:
                raise ValueError("Extra tree node")
            binary_parent, side = slots.popleft()
            if token == "<NULL>":
                if side == "root":
                    raise ValueError("Null root")
                continue
            if token in V3_GRAMMAR:
                raise ValueError("Expected core")
            parse_core(token)
            fi = len(cores)
            cores.append(token)
            graph.add_node(fi)
            if side == "root":
                parent = None
            elif side == "left":
                parent = binary_parent
            else:
                parent = parents[binary_parent]
            if side != "root" and parent is None:
                raise ValueError("Root cannot have sibling")
            parents[fi] = parent
            if parent is not None:
                endpoints = []
                for f in (parent, fi):
                    sites = _sites(cores[f])
                    if not sites:
                        raise ValueError("Core has no attachment site")
                    atom = sites[0] if len(sites) == 1 else number()
                    if atom not in sites:
                        raise ValueError("Invalid attachment site")
                    endpoints.append([f, atom])
                links.append([*endpoints, "SINGLE"])
                graph.add_edge(parent, fi)
            slots.extend(((fi, "left"), (fi, "right")))
        if not cores:
            raise ValueError("Empty component")
        # Unwritten trailing slots are null; the explicit connectivity is now
        # rebuilt entirely from the tree and endpoint records.
        tree = [None if t == "&" else int(t) for t in binary_tree_tokens(graph, [str(i) for i in range(len(cores))])]
        components.append(dict(cores=cores, tree=tree, links=links, atom_stereo=atoms, bond_stereo=bonds))
        if pos < len(tokens):
            take()
            if pos == len(tokens):
                raise ValueError("Trailing component separator")
    return decode(json.dumps(dict(format=FORMAT, components=components)))


TOKEN_FORMAT = "external-tsdy-v3.1"


def checked_training_tokens(smiles, max_content_tokens=510):
    """Fail rather than silently truncate or substitute unsupported chemistry.

    Callers must apply a documented common supported-domain policy before this
    function. A passing fixed-sample audit is not a full-corpus guarantee.
    """
    tokens = model_tokens(encode(smiles))
    if len(tokens) > max_content_tokens:
        raise ValueError(f"External TSDY requires {len(tokens)} content tokens; limit is {max_content_tokens}")
    return tokens
