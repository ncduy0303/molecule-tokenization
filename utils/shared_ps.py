"""Deterministic PS-style coarsening of the shared exocyclic partition.

Learns ordered, connected-subgraph merge rules, not a neural VAE or a model
token vocabulary. Global counts are updated only for affected molecules.
The reference implementation retains molecule states in RAM; measure memory
and throughput before attempting a 2M corpus.
"""

from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import heapq
import json
from pathlib import Path

from rdkit import Chem, rdBase

from utils.cache_utils import file_digest
from utils.shared_fragmentation import EXOCYCLIC_SMARTS, partition_from_cuts, partition_molecule

FORMAT = "shared-exocyclic-ps-v2"


class MergeState:
    def __init__(self, partition):
        self.mol = partition.mol
        self.key_mol = Chem.Mol(self.mol)
        Chem.RemoveStereochemistry(self.key_mol)
        self.groups = {i: set(group) for i, group in enumerate(partition.groups)}
        self.owner = {a: i for i, group in self.groups.items() for a in group}
        self.cuts = set(partition.cut_bonds)
        self._candidates = None

    def candidates(self):
        if self._candidates is None:
            candidates = defaultdict(list)
            for idx in sorted(self.cuts):
                bond = self.mol.GetBondWithIdx(idx)
                a, b = self.owner[bond.GetBeginAtomIdx()], self.owner[bond.GetEndAtomIdx()]
                atoms = sorted(self.groups[a] | self.groups[b])
                # Same union-graph key for training and inference. Bond order,
                # charge and isotopes are retained; stereo is not a mining key.
                key = Chem.MolFragmentToSmiles(self.key_mol, atomsToUse=atoms, canonical=True, isomericSmiles=True)
                candidates[key].append((idx, a, b))
            self._candidates = dict(candidates)
        return self._candidates

    def counts(self):
        return Counter({key: len(pairs) for key, pairs in self.candidates().items()})

    def merge(self, key):
        changed = False
        used = set()
        for idx, a, b in self.candidates().get(key, ()):
            # Non-overlapping occurrences only. Later occurrences are reconsidered
            # at a later rule, with their new union graph, as in graph BPE.
            if a in used or b in used:
                continue
            used.update((a, b))
            self.groups[a].update(self.groups.pop(b))
            for atom in self.groups[a]:
                self.owner[atom] = a
            self.cuts.remove(idx)
            changed = True
        if changed:
            self._candidates = None
        return changed

    def partition(self):
        return partition_from_cuts(self.mol, self.cuts)


@dataclass
class SharedPSModel:
    rules: tuple[str, ...]
    provenance: dict

    def __post_init__(self):
        self._ranks = defaultdict(list)
        for i, key in enumerate(self.rules):
            self._ranks[key].append(i)

    def apply(self, partition):
        state = MergeState(partition)
        start = 0
        while True:
            available = []
            for key in state.candidates():
                ranks = self._ranks.get(key, ())
                i = bisect_left(ranks, start)
                if i < len(ranks):
                    available.append((ranks[i], key))
            if not available:
                return state.partition()
            rank, key = min(available)
            state.merge(key)
            start = rank + 1

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Existing experiments must not silently acquire new merge rules.
        with path.open("x") as stream:
            json.dump(
                {
                    "format": FORMAT,
                    "cut_smarts": EXOCYCLIC_SMARTS,
                    "rdkit_version": rdBase.rdkitVersion,
                    "rules": self.rules,
                    "provenance": self.provenance,
                },
                stream,
                indent=2,
            )


@lru_cache(maxsize=4)
def _load_model(path, digest):
    payload = json.loads(Path(path).read_text())
    if file_digest(path) != digest:
        raise ValueError("PS merge model changed while being loaded")
    if payload.get("format") != FORMAT or payload.get("cut_smarts") != EXOCYCLIC_SMARTS:
        raise ValueError(
            "Not a current shared exocyclic PS model; re-mine with the bridge-safe cut rule (legacy models are incompatible)"
        )
    if payload.get("rdkit_version") != rdBase.rdkitVersion:
        raise ValueError("PS merge model RDKit version differs from the current environment")
    rules = payload.get("rules")
    if not isinstance(rules, list) or any(not isinstance(r, str) or not r for r in rules):
        raise ValueError("Invalid PS merge rules")
    return SharedPSModel(tuple(rules), payload.get("provenance", {}))


def load_model(path):
    resolved = str(Path(path).resolve())
    return _load_model(resolved, file_digest(resolved))


def train_shared_ps(smiles, num_merges=1000, min_frequency=10, provenance=None, progress=None):
    if num_merges < 0 or min_frequency < 1:
        raise ValueError("num_merges must be non-negative and min_frequency must be positive")
    states = [MergeState(partition_molecule(smi)) for smi in smiles]
    counts = Counter()
    members = defaultdict(set)
    for i, state in enumerate(states):
        local = state.counts()
        counts.update(local)
        for key in local:
            members[key].add(i)
    heap = [(-count, key) for key, count in counts.items()]
    heapq.heapify(heap)
    rules = []
    for step in range(num_merges):
        while heap:
            negative, key = heapq.heappop(heap)
            if -negative == counts[key]:
                break
        else:
            break
        frequency = counts[key]
        if frequency < min_frequency:
            break
        affected = sorted(members[key])
        changed_keys = set()
        for i in affected:
            state = states[i]
            old = state.counts()
            counts.subtract(old)
            for candidate in old:
                members[candidate].discard(i)
            state.merge(key)
            new = state.counts()
            counts.update(new)
            for candidate in new:
                members[candidate].add(i)
            changed_keys.update(old)
            changed_keys.update(new)
        for candidate in changed_keys:
            if counts[candidate] > 0:
                heapq.heappush(heap, (-counts[candidate], candidate))
            else:
                members.pop(candidate, None)
        rules.append(key)
        # Bound stale heap entries left by lazy frequency updates.
        if len(heap) > 4 * max(1, len(members)):
            heap = [(-counts[candidate], candidate) for candidate in members if counts[candidate] > 0]
            heapq.heapify(heap)
        if progress is not None:
            progress(step + 1, key, frequency, len(affected))
    return SharedPSModel(tuple(rules), provenance or {})
