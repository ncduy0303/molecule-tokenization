= Introduction

== Background and motivation
Drawing inspiration from advances in natural language processing, researchers have developed chemical language models that learn from molecular structures represented as sequences @sultan2024transformers. A widely used representation is the Simplified Molecular Input Line Entry System (SMILES), which expresses a molecular graph as a string containing atom symbols and information about bonds, branches, rings, and stereochemistry @weininger1988smiles. Models such as ChemBERTa use self-supervised pretraining on molecular strings before being adapted to property prediction tasks @chithrananda2020chemberta.

Before a model can process a molecular string, a tokenizer divides it into units and maps those units to entries in a vocabulary. Different tokenization schemes can divide the same notation differently, as illustrated in @intro-smiles. These choices determine which structural and syntactic information is grouped into a single token.

#figure(
  image("../../CA/images/SMILES_illustration.png", width: 100%),
  caption: [SMILES notation and tokenization granularity. Panel (a) shows structural features and panel (b) compares atom-wise and glyph-based tokenization. Source: @wadell2026tokenization.],
) <intro-smiles>

This project focuses particularly on fragment-based tokenization, in which a molecule is divided into connected groups of atoms before being represented as a sequence. Two design choices are involved: which bonds are cut to form fragments, and how the fragments and their connections are written in the sequence. Existing approaches include t-SMILES, a family of tree-based molecular representations @wu2024t @wu2024hierarchical, and fragSMILES, a fragment-based molecular representation @mastrolorito2025fragsmiles.

A central question is how to represent a fragment independently of its connections to neighbouring fragments. If attachment annotations are included inside a fragment's vocabulary entry, the same underlying core can require different entries depending on its attachment positions or number of attachments. Keeping connection information separate may allow those forms to share a vocabulary entry. However, the connection information must still be represented if the original molecule is to be reconstructed, and this introduces additional syntax.

There is also a trade-off between grouping atoms into larger fragments and covering unfamiliar molecules with a finite vocabulary. In a tokenizer that assigns one token to each fragment, an unseen fragment is mapped to an unknown token unless another encoding mechanism is provided. Larger fragments may shorten sequences without improving coverage or predictive performance. These considerations motivate evaluating sequence length, unknown-token rates, and downstream prediction together.

== Objectives and research questions
The objective is to compare molecular tokenization methods and investigate whether fragment-based representations can share vocabulary entries more effectively while retaining the information needed for reconstruction. The evaluation follows the pipeline in @intro-pipeline: construct a tokenizer, pretrain a chemical language model, and finetune the model for molecular property prediction.

#figure(
  image("../../CA/images/Pipeline_illustration.png", width: 90%),
  caption: [Conceptual evaluation pipeline. Corpus names are illustrative; the methodology chapter specifies the datasets and splits used.],
) <intro-pipeline>

The project addresses three research questions:
+ How do the choice of fragmentation rules and the merging of adjacent fragments affect sequence length, vocabulary coverage, and molecular property prediction?
+ Can separating fragment cores from attachment information improve vocabulary sharing when the fragment partition is held fixed?
+ To what extent do improvements in coverage or compression correspond to improvements in downstream predictive performance?

== Contributions and scope
The project makes three practical contributions:
+ *A controlled fragmentation framework.* Different sequence representations can use the same molecular partition, making their assembly schemes easier to compare. The framework also supports learned merging of adjacent fragments, building on principal-subgraph mining @kong2022molecule.
+ *A representation with separate fragment and attachment information.* Building on t-SMILES @wu2024t, the implementation represents clean fragment cores separately from the information needed to reconnect them and preserve supported stereochemistry. The design chapter introduces the particular t-SMILES variant and the enhanced encoding syntax.
+ *An evaluation of representation trade-offs.* The experiments combine property prediction with measurements of sequence length, unknown-token rates, and fragment vocabulary sharing. This allows improvements in one measure to be considered alongside their costs and limitations in others.
