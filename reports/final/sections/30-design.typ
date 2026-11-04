= System design and implementation

== Benchmark framework and design objectives

=== Initial benchmark pipeline
The benchmark compares molecular tokenizers using a common RoBERTa architecture. It includes character- and glyph-based methods (BPE, PCATT, and SMIRK configurations), atom-based methods (SPE and APE), and fragment-word tokenizers for t-SMILES and fragSMILES. Selected configurations also use SAFE as the input representation. The t-SMILES configurations use either BRICS or MMPA fragmentation.

Each run has three stages:
+ *Vocabulary construction:* fit the tokenizer to the selected molecular corpus, or load an existing vocabulary. The APE baseline uses the authors' pretrained vocabulary.
+ *Masked language model (MLM) pretraining:* convert molecules to token IDs and train a RoBERTa encoder to predict masked tokens.
+ *Property prediction:* finetune the encoder for binary classification on two MoleculeNet datasets: HIV, which measures activity against the human immunodeficiency virus, and BBBP, which labels compounds by their ability to cross the blood-brain barrier @wu2018moleculenet.

=== Controlled fragment comparisons
Comparing BRICS-based t-SMILES with default fragSMILES changes both the bonds cut and the notation used to reconnect the fragments. The controlled benchmark separates these choices. As shown in @design-shared-pipeline, each serializer receives either the same exocyclic partition or that partition after applying a fixed set of principal-subgraph (PS) merge rules.

#figure(
  {
    let stage(body) = block(width: 100%, inset: 7pt, stroke: 0.5pt, radius: 3pt, align(center, body))
    stage[Input SMILES $->$ molecular graph $->$ shared exocyclic partition]
    align(center)[$arrow.b$]
    grid(
      columns: (1fr, 1fr),
      gutter: 12pt,
      stage[*Base path* \ Keep the starting fragments], stage[*PS path* \ Apply frozen graph-merge rules],
    )
    align(center)[$arrow.b$]
    stage[Selected serializer: fragSMILES / TSID / TSDY / TSIS]
    align(center)[$arrow.b$]
    stage[Representation-specific tokens and vocabulary lookup]
    align(center)[$arrow.b$]
    stage[MLM pretraining $->$ property prediction]
  },
  caption: [Shared-fragmentation pipeline. The base path retains the original fragments; the principal-subgraph (PS) path merges adjacent fragments before serialization.],
) <design-shared-pipeline>

RDKit parses molecules and constructs fragments @rdkitsoftware. NetworkX identifies graph bridges and connected components @hagberg2008networkx. The fragSMILES adapter uses chemicalgof for assembly @mastrolorito2025fragsmiles; the t-SMILES adapters implement the published tree traversals on the partition @wu2024t @wu2024hierarchical.

#include "31-shared-fragmentation.typ"
#include "32-enhanced-tsdy.typ"

== Reproducibility and implementation checks

*Corpus separation.* Across all configurations, the tokenizer-training rows had zero overlap with the MLM training, validation, and test rows. This check used PubChem row IDs. Distinct rows can still represent the same molecule. This separation prevents the tokenizer from learning its vocabulary from the exact records used for MLM training or evaluation.

*Finetuning trials.* Each trial used a different finetuning seed with the same tokenizer, pretrained model, and dataset split. Three trials were run for each configuration, and the results were averaged. Averaging reduces the influence of any single finetuning run on the comparison.

*Vocabulary coverage.* The intrinsic evaluation measures sequence length and unknown-token rates. Matching fragments by their atoms allows base and enhanced TSDY to be compared on the same fragment occurrences.
