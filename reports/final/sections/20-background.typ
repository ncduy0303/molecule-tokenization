= Background and related work

== Molecular representations and language models

=== From molecular graphs to sequences
A molecule can be described as a graph whose vertices represent atoms and whose edges represent bonds. Sequence-based approaches express this structure as a string so that it can be processed by a language model. SMILES uses atom symbols together with branch, bond, ring-closure, and stereochemical notation @weininger1988smiles. Consequently, adjacent characters are not necessarily separate atoms, and graph neighbours need not appear next to each other in the string.

SELFIES is an alternative representation designed to produce valid molecular structures from sequences under its semantic rules @krenn2020self. This addresses a problem that is particularly relevant to molecular generation: whether a generated string can be interpreted as a valid molecule. SMILES and SELFIES thus differ in how their syntax constrains the mapping from strings to molecular structures.

=== Pretraining and property prediction
ChemBERTa applies the RoBERTa architecture to molecular strings @chithrananda2020chemberta @liu2019roberta. In masked language modeling (MLM), selected input tokens are masked, and the model predicts their original identities using the surrounding sequence. This provides a training objective that does not require experimentally measured property labels. Through this objective, the model learns contextual relationships between tokens in molecular strings.

A pretrained model can subsequently be finetuned on labelled molecules for property prediction. ChemBERTa-2 also investigates multi-task regression, in which pretraining predicts computed molecular properties rather than masked tokens @ahmad2022chemberta. MLM learns from token context, whereas multi-task regression provides supervision through computed properties.

== Character, glyph, and atom-based tokenization

=== Statistical subword methods
Character-level tokenization uses small units, while subword methods combine units into longer recurring sequences. Byte-pair encoding (BPE) repeatedly merges frequent adjacent pairs @gage1994new. The initial units matter: starting from individual characters produces a different search space from starting with atom-level units. A learned sequence token is not necessarily a connected molecular fragment, since its boundaries are determined in the string representation.

The Unigram language-model approach instead learns a probabilistic vocabulary of candidate segments and prunes entries during training @kudo2018subword. A further alternative, the Partition Cover Approach to Tokenization (PCATT), formulates vocabulary selection as a coverage problem and uses a greedy selection procedure @lim2025partition.

=== Glyph-based tokenization
Character-level tokenization separates a string into individual characters without recognizing multi-character element symbols. For example, the characters `Sc` occur within `[Sc]`, denoting scandium, but also at the start of `Sc1ccccc1`, where `S` denotes sulfur and `c` an aromatic carbon. The surrounding syntax distinguishes these cases; the issue is not ambiguity in a complete valid SMILES string, but the loss of element boundaries when the notation is divided into characters.

SMIRK uses chemically meaningful glyphs to preserve such boundaries while decomposing bracketed atom specifications into reusable components @wadell2026tokenization. Thus, `Sc` inside `[Sc]` is an element glyph, whereas the sulfur and aromatic carbon in `Sc1ccccc1` remain separate units. Compared with atom-wise tokenization, glyph-based tokenization also separates an atom's element from annotations such as charge or hydrogen count. @intro-smiles illustrates this difference between atom-wise and glyph-based segmentation.

BPE can then be applied to the glyph sequence to learn frequently occurring combinations. Its pair-merging procedure remains the same, but it starts from glyphs rather than individual characters, preserving the initial glyph boundaries within the merged tokens.

=== Atom-based pair encoding
SMILES Pair Encoding (SPE) applies a BPE-style procedure to an atom-tokenized SMILES sequence: frequent adjacent token pairs are merged into progressively larger units @li2021smiles. Atom Pair Encoding (APE) similarly learns pair merges from atom-based units @leon2024comparing. Unlike character- or glyph-based segmentation, this starting representation keeps bracketed atom expressions together rather than splitting their internal annotations. In all three settings, pair merging operates on a sequence; the difference lies in the units supplied to the algorithm. @background-spe illustrates how learned sequence units differ from the initial atom-wise segmentation.

#figure(
  image("../../CA/images/SPE_illustration.jpeg", width: 100%),
  caption: [Atom-level segmentation and SMILES Pair Encoding. Source: @li2021smiles.],
) <background-spe>

Preserving atom boundaries does not make string-based merging equivalent to graph-based merging. Branches and ring closures can separate connected atoms in the serialized string. This motivates methods that partition the molecular graph before constructing the sequence, rather than relying only on adjacent sequence units.

== Fragment-based molecular representations

=== Partitioning and serialization
Fragment-based representations introduce two distinct operations. First, a fragmentation rule divides the graph into substructures. Second, a serialization scheme records those substructures and their relationships as a sequence. Breaking of Retrosynthetically Interesting Chemical Substructures (BRICS) uses chemically defined bond-cleavage rules to construct fragment spaces @degen2008brics. The t-SMILES framework supports BRICS and other fragmentation methods, including matched molecular pair analysis (MMPA) @wu2024t. The resulting fragment distribution can therefore change even when the serialization scheme is held constant.

After serialization, the molecular string must be divided into tokens for the language model. One approach assigns a single token to each complete fragment string. In this case, different attachment annotations produce distinct vocabulary entries, even when the underlying fragment core is the same. Alternatively, subword tokenization divides fragment strings into smaller units that can be shared across different fragments.

=== SAFE
Sequential Attachment-based Fragment Embedding (SAFE) groups molecular notation into fragment blocks separated by dots and uses ring-closure notation to preserve connections between blocks @noutahi2024gotta. The fragment arrangement makes the substructures more explicit than in a conventional SMILES traversal. @background-safe illustrates the representation.

#figure(
  image("../../CA/images/SAFE_illustration.gif", width: 100%),
  caption: [Fragment blocks in SAFE. Source: @noutahi2024gotta.],
) <background-safe>

In SAFE, the connection symbols occur within the fragment blocks. Consequently, a fragment core can appear in different annotated strings depending on its connections. Subword tokenization can share portions of those strings, while whole-block tokenization treats each distinct block as a separate vocabulary item.

=== t-SMILES and its variants
The t-SMILES framework represents fragment relationships using a tree structure. Its original variants include TSSA, which uses shared atoms; TSDY, which uses dummy atoms without connection identifiers; and TSID, which uses dummy atoms with identifiers @wu2024t. In the full-binary-tree representation, `&` denotes an empty tree node and `^` separates fragment strings.

The later TSIS representation simplifies TSID by reading from the acyclic molecular tree rather than the full binary tree. Related variants investigate alternative traversal orders @wu2024hierarchical. @background-tsmiles shows the relationship between the fragment graph and these tree representations.

#figure(
  image("../../CA/images/tSMILES_illustration.png", width: 85%),
  caption: [Fragmentation and tree representations used by t-SMILES. Source: @wu2024hierarchical.],
) <background-tsmiles>

Dummy atoms, written as `*`, mark attachment sites within a fragment. TSID and TSIS additionally label these sites with connection identifiers (e.g., `[*1]`, `[*2]`), allowing corresponding sites in different fragments to be paired during reconstruction. TSDY retains the dummy atoms but omits these identifiers, so the explicit pairing information is lost.

Even without identifiers, fragments containing the same core can differ in the positions or number of dummy atoms. Under whole-fragment tokenization, these different strings require separate vocabulary entries. This motivates separating attachment information from the fragment core.

=== fragSMILES
fragSMILES represents fragments together with external connection annotations, providing a reference for separating fragment content from assembly information @mastrolorito2025fragsmiles. Its default fragmentation strategy cuts single bonds that are not part of any ring. The sequence is organized around a main path in the fragment graph, with branches recorded recursively. @background-fragsmiles illustrates the progression from an atom graph to fragments and a textual representation.

#figure(
  image("../../CA/images/fragSMILES_illustration.png", width: 100%),
  caption: [Fragment-based graph construction and connection annotations in fragSMILES. Source: @mastrolorito2025fragsmiles.],
) <background-fragsmiles>

External connection annotations allow fragment content to be represented separately from its position in the assembled molecule. This offers a different balance from embedding dummy atoms and their identifiers inside fragment strings: the fragment core can be reused across attachment patterns, while separate syntax carries the assembly information.

== Data-driven fragment extraction

=== Principal-subgraph mining
Principal-subgraph mining learns recurring molecular substructures from a corpus. This mining procedure was combined with a variational autoencoder and an assembly mechanism in PS-VAE @kong2022molecule. The mining procedure repeatedly merges frequently occurring adjacent subgraphs. Conceptually, it resembles pair merging in sequence tokenization, but adjacency is defined on the molecular graph.

As illustrated in @background-psm, the original mining procedure starts with individual atoms and repeatedly considers unions of neighbouring fragments. Candidate unions are counted across the corpus, and the most frequent pattern is added to the vocabulary. Its occurrences are merged before the next iteration @kong2022molecule.

#figure(
  image("../assets/PSM.png", width: 100%),
  caption: [Principal-subgraph mining on a toy molecular dataset. Panels (a) to (c) show atom initialization and two successive merges. Colours indicate the fragment partition. Reproduced from Kong's explanation of PS-VAE @kong2022psvaeweb.],
) <background-psm>
