# Legacy PS-fragSMILES

This is the ring-initialized principal-subgraph implementation used by the earlier PS-fragSMILES experiments.

- `mol_bpe_ring.py`: initialize ring-based blocks, mine merges and decompose molecules.
- `molecule.py`: subgraph representation and atom mappings.
- `ps_encode.py`: convert the decomposition to fragSMILES.

For shared fragmentation experiments, use `utils/shared_fragmentation.py` and `utils/shared_ps.py` through `algorithm=train_shared_ps`. Their `ps_merges.json` format is not interchangeable with legacy `ps_vocab.txt` artifacts. See [configuration options](../../configurations/README.md).
