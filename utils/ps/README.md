# Principal-subgraph fragmentation

This code implements the principal-subgraph path used by PS-fragSMILES. The CA report describes it as an adaptation of PS-VAE-style graph BPE that collapses ring systems before mining, to support fragSMILES assembly.

| File                  | Role                                                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `mol_bpe_ring.py`     | Detect ring systems, initialize molecular subgraphs, mine frequent adjacent merges, save/load the PS vocabulary, and decompose molecules |
| `molecule.py`         | Molecular subgraph nodes and edges, atom mappings, reconstruction, and SVG inspection                                                    |
| `ps_encode.py`        | Convert PS decomposition into a `chemicalgof` graph and serialize fragSMILES                                                             |
| `utils/chem_utils.py` | RDKit helpers and atom valence table                                                                                                     |
| `utils/logger.py`     | Console logging                                                                                                                          |

`TokenizerTrainingExperiment._train_ps` trains the raw PS vocabulary, converts the corpus, counts serialized tokens, and writes a Hugging Face-compatible tokenizer. `PSFragSMILESTokenizer` retains both the serialized-token vocabulary and the PS vocabulary needed to encode new SMILES.
