== Shared fragmentation and principal-subgraph merging

=== Exocyclic starting partition
All four representations use the same starting fragments. RDKit parses a canonical isomeric SMILES for each molecule. The partitioner selects non-ring single bonds using chemicalgof's exocyclic cut rule @mastrolorito2025fragsmiles, including its restriction on charged atom pairs. It cuts a selected bond only if removing that bond disconnects its endpoints. This bridge check keeps bonds within a fragment whenever another path connects the same atoms.

Each connected group of atoms left after cutting forms one fragment. The implementation records its atom indices and the cut bonds in the original molecule. The fragment graph is therefore a tree for each connected molecular component, even when a fragment itself contains rings. The partitioner checks that every atom belongs to exactly one fragment and that the recorded cuts coincide with the fragment boundaries.

=== Serialization of the common partition
Each serializer receives the same atom groups and cut bonds. The formats differ in how they express the connections:
- *fragSMILES* uses chemicalgof to write the fragments with external connection annotations.
- *TSID* marks both ends of each cut with matching labelled dummy atoms, such as `[1*]`. It writes the fragments in breadth-first order on a first-child/next-sibling binary tree.
- *TSDY* uses the same traversal and keeps the dummy atoms, but removes their pairing identifiers (i.e., `[1*]` becomes `[*]`).
- *TSIS* retains the pairing identifiers and traverses the original fragment tree in breadth-first order, without binary-tree placeholders.

Each tokenizer builds its own vocabulary from the resulting strings. This allows the four notations to be compared at the same fragment boundaries.

=== Learning and applying PS merges
The merger adapts the principal-subgraph mining procedure of from @kong2022molecule by starting from exocyclic fragments instead of individual atoms. It learns an ordered set of rules from the tokenizer-training corpus:

+ For each pair of adjacent fragments, form their union and count occurrences of its canonical SMILES across the corpus. The matching key retains bond orders, charges, and isotopes but omits stereochemistry. The original graph retains the stereochemical information for serialization.
+ Select the most frequent union and merge its non-overlapping occurrences. Lexical order resolves frequency ties, and bond-index order resolves competing occurrences within a molecule.
+ Update counts only for molecules containing the selected union. Repeat until the rule budget is reached or no candidate meets the minimum frequency.

At encoding time, all four serializers use the same frozen rules in the same order. Each merge removes one boundary between neighbouring fragments, preserving the tree structure. The merge budget controls which fragments can form. The tokenizer vocabulary budget separately controls which serialized units receive a token ID or replaced by UNK.
