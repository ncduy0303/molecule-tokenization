// ==========================================
// 1. TEMPLATE CONFIGURATION & SETUP
// ==========================================

// #show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => if y == 0 {
    (bottom: 0.7pt + black)
  },
  align: (x, y) => (
    if x > 0 { center } else { left }
  ),
)

#let project(
  title: "",
  author: "",
  matric_num: "",
  academic_year: "",
  project_id: "",
  supervisor: "",
  evaluator: "",
  body,
) = {
  // Document metadata and base settings
  set document(author: author, title: title)
  set text(font: "New Computer Modern", size: 10pt)
  set page(
    paper: "a4",
    margin: (left: 3cm, right: 3cm, top: 3cm, bottom: 3cm),
  )

  // Title Page Layout
  align(center)[
    #text(size: 14pt)[B.Comp. Dissertation (CA Report)] \

    #v(3cm)

    #text(size: 20pt, weight: "bold")[#title]

    #v(3cm)

    #text(size: 12pt)[by] \
    #v(0.5em)
    #text(size: 14pt)[#author] \
    #v(0.5em)
    #text(size: 12pt)[#matric_num]

    #v(3cm)

    #text(size: 12pt)[
      Department of Computer Science \
      School of Computing \
      National University of Singapore \
      Academic Year #academic_year
    ]

    #v(4cm)

    #align(left)[
      // Adding a block to center the supervisor/evaluator details visually
      // while keeping the text left-aligned within the block
      #box(width: 100%)[
        #set align(center)
        #grid(
          columns: (auto, auto),
          gutter: 1.5em,
          align: left,
          [Project ID:], [#project_id],
          [Project Supervisor:], [#supervisor],
          [Main Evaluator:], [#evaluator],
        )
      ]
    ]
  ]

  pagebreak()

  show outline.entry: it => {
    if it.level == 1 {
      v(1em)
      strong(it)
    } else {
      it
    }
  }

  outline(
    title: "Contents",
    indent: auto,
    depth: 3,
  )

  pagebreak()

  // Main Body Setup
  set page(numbering: "1")
  counter(page).update(1)
  set par(justify: true, leading: 0.65em)
  set heading(numbering: "1.1")

  // Add some spacing after headings
  show heading: it => [
    #pad(bottom: 0.5em, it)
  ]

  body
}


// ==========================================
// 2. DOCUMENT CONTENT
// ==========================================

#show: project.with(
  title: "Advanced Tokenization Methods for Molecular Foundation Models",
  author: "Nguyen Cao Duy",
  matric_num: "A0258078R",
  academic_year: "2025-2026",
  project_id: "H437020",
  supervisor: "Dr. Bian Yatao",
  evaluator: "Dr. Chen Yu",
)

= Introduction

== Project Overview

Drawing inspiration from the success of transformer architectures in Natural Language Processing (NLP), researchers have increasingly framed fundamental molecular tasks such as property prediction and de novo drug design as sequence modeling problems @sultan2024transformers. At the core of this approach is the Simplified Molecular Input Line Entry System (SMILES), a 1D string notation that linearizes 2D molecular graphs utilizing a depth-first search traversal @weininger1988smiles. By representing molecules as sequences of alphanumeric characters, Chemical Language Models (CLMs) like ChemBERTa can leverage self-supervised pretraining on massive databases of unlabeled SMILES strings to learn generalizable representations of the chemical space @chithrananda2020chemberta.

#figure(
  image("images/SMILES_illustration.png"),
  caption: [(a) Illustration of key features encoded in SMILES notation. (b) Example of different tokenization strategies for SMILES such as character-level splitting, atom-wise splitting, and glyph-based splitting. @wadell2026tokenization],
) <SMILES_illustration>

One important design choice in CLMs is the selection of a tokenization strategy, as it directly influences the granularity of molecular representations and the model’s ability to capture chemical semantics. In practice, this introduces a trade-off between semantic representation and tokenization efficiency. General-purpose tokenizers, such as Byte-Pair Encoding (BPE), segment strings based purely on statistical frequency. While computationally efficient, these algorithms lack chemical awareness and may fragment critical structural motifs, hindering the model’s ability to learn meaningful chemical patterns. In contrast, chemically-aware tokenizers that operate at the atom or fragment level better preserve underlying chemical structure, enabling richer semantic understanding. However, these approaches are vulnerable to the out-of-vocabulary (OOV) problem: previously unseen motifs are mapped to an "unknown" (UNK) token, leading to a loss of information. This issue is exacerbated by the vastness of chemical space, where the number of possible substructures is effectively unbounded. For instance, the number of valid bracketed atom specifications is estimated to be 28 trillion @wadell2026tokenization.

== Project Objectives
The primary objective of this project is to systematically benchmark chemical language tokenization methods and develop improved tokenization strategies for molecular foundation models. By analyzing how different tokenization schemes influence representation learning, this project aims to both understand and advance the role of tokenization in chemical language modeling.

Specifically, the project focuses on:

+ *Systematically Benchmark Existing Tokenizers:* Conduct a comprehensive evaluation of existing tokenization paradigms—including character-, glyph-, atom-, and fragment-based methods—by measuring their impact on downstream molecular property prediction tasks (@pipeline_illustration). This includes analyzing trade-offs between vocabulary size, token fertility, UNK token rates, as well as predictive metrics on benchmark datasets.
+ *Design of an Improved Tokenization Scheme:* Building on the benchmark results, the second objective is to develop and refine advanced tokenization strategies that overcome the limitations of current representations. This focuses on optimizing existing fragment-based frameworks, such as t-SMILES @wu2024t and fragSMILES @mastrolorito2025fragsmiles, to enhance their chemical awareness and structural robustness. The goal is to formulate a more effective and generalizable tokenization scheme that addresses the out-of-vocabulary problem while preserving the chemical semantics necessary for robust representation learning in CLMs.

#figure(
  image("images/pipeline_illustration.png", width: 80%),
  caption: [*High-level pipeline for benchmarking SMILES tokenization strategies:* a tokenizer is learned from a large unlabeled corpus, used to pretrain a CLM, and subsequently fine-tuned on smaller downstream datasets for task-specific evaluation.],
) <pipeline_illustration>

= Literature Review

== Overview of Chemical Language Models

=== From Traditional Descriptors to Sequence Models
Historically, cheminformatics relied heavily on expert-crafted chemical fingerprints to encode structural features into fixed-length vectors. More recently, Graph Neural Networks (GNNs) emerged as a powerful alternative, operating directly on 2D molecular topologies. However, inspired by the success of Transformers in NLP, researchers have increasingly adopted sequence-based CLMs as a promising alternative to both traditional fingerprints and GNN-learned representations @chithrananda2020chemberta. The de-facto standard for molecular representation in CLMs is the SMILES notation, which linearizes molecular graphs into 1D strings @weininger1988smiles.

Alternative representations have also been proposed to address some limitations in SMILES generation. Most notably, SELFIES was developed to guarantee 100% robustness, ensuring that every possible SELFIE sequence maps to a valid molecule @krenn2020self. However, the grammatical rules of SELFIES are significantly more complex, while providing no significant performance gain on downstream property prediction tasks @chithrananda2020chemberta @leon2024comparing. Because SMILES provides a more intuitive and compact structural encoding, it remains the dominant notation for CLMs.

=== Architectures and Pretraining Objectives
To process these linear strings, CLMs such as ChemBERTa @chithrananda2020chemberta utilize the RoBERTa architecture @liu2019roberta to learn generalizable representations of the chemical space. The primary training objective for these foundation models is Masked Language Modeling (MLM). During MLM pretraining, a percentage of the tokens in a SMILES string are hidden, and the model is tasked with predicting the missing tokens based on the surrounding context. This forces the model to implicitly learn fundamental chemical rules, such as valency or ring closure.

Recent literature has also explored alternative pretraining objectives beyond MLM. For example, the ChemBERTa-2 study compared MLM against Multi-Task Regression (MTR) @ahmad2022chemberta. In MTR pretraining, the model is tasked with simultaneously predicting multiple computed molecular properties directly from the SMILES string. While MLM excels at teaching the model pure chemical syntax, alternative objectives like MTR can provide strong inductive biases for predicting explicit molecular properties during downstream finetuning, leading to improved performance on MoleculeNet benchmarks.


== Character, Glyph, and Atom-Based Tokenization

=== Statistical Sub-Word Tokenization Algorithms
Sequence-based CLMs face a fundamental computational challenge regarding sequence length and representation. If a molecule is represented solely by its foundational characters or glyphs (e.g., `C`, `O`, `=`, `(`), the resulting sequences become excessively long. This exhausts the context window of Transformer architectures and forces the network to expend computational capacity learning basic chemical valency before the higher-order structural semantics. To mitigate this, sub-word tokenization is employed as a compression technique to group smaller structural units into frequently occurring larger tokens.

The most ubiquitous statistical sub-word tokenization strategies adapted from NLP are BPE @gage1994new and the Unigram language model @kudo2018subword. BPE operates via a bottom-up merging strategy, iteratively combining the most frequently co-occurring adjacent token pairs into single tokens, starting from an initial vocabulary (e.g., individual characters or glyphs). Conversely, the Unigram model employs a top-down probabilistic approach: it initializes with a massive predefined vocabulary and iteratively prunes the least useful tokens by maximizing the marginal likelihood of the training corpus.

To explore alternatives to traditional statistical merging and probabilistic pruning, recent NLP research has proposed the Partition Cover Approach to Tokenization (PCATT) @lim2025partition, which formulates tokenization as a stronger version of the weighted maximum coverage problem which is known to be NP-hard. By utilizing a polynomial-time greedy algorithm that iteratively adds tokens to the vocabulary based on their marginal contribution to corpus coverage, PCATT can avoid the strict constraints of adjacent pairwise merges that characterize BPE and thus has the potential to capture more complex token patterns.

While PCATT remains a general-purpose, statistical NLP algorithm, applying it at the foundational character and multi-character (glyph) levels provides a possible experiment for this project. It allows for the evaluation of whether a partition cover formulation can capture the underlying syntax of SMILES strings more effectively than traditional algorithms like BPE.

=== Glyph-Based Tokenization
An inherent issue with character-based tokenization on SMILES strings is the ambiguity in the notation. For example, the sequence `Sc` could represent the element Scandium, or it could represent a Sulfur atom next to an aromatic carbon. To address this, the SMIRK tokenizer was introduced to provide full coverage of the OpenSMILES specification by systematically standardizing the vocabulary with meaningful multi-character glyphs @wadell2026tokenization. This ensures that downstream statistical algorithms like BPE do not conflate different chemical symbols due to overlapping string representations.

#figure(
  {
    let vocab = (
      "#": 1,
      "$": 2,
      "%": 3,
      "(": 4,
      ")": 5,
      "*": 6,
      "+": 7,
      "-": 8,
      ".": 9,
      "/": 10,
      "0": 11,
      "1": 12,
      "2": 13,
      "3": 14,
      "4": 15,
      "5": 16,
      "6": 17,
      "7": 18,
      "8": 19,
      "9": 20,
      ":": 21,
      "=": 22,
      "@": 23,
      "@@": 24,
      "@AL": 25,
      "@OH": 26,
      "@SP": 27,
      "@TB": 28,
      "@TH": 29,
      "Ac": 30,
      "Ag": 31,
      "Al": 32,
      "Am": 33,
      "Ar": 34,
      "As": 35,
      "At": 36,
      "Au": 37,
      "B": 38,
      "Ba": 39,
      "Be": 40,
      "Bh": 41,
      "Bi": 42,
      "Bk": 43,
      "Br": 44,
      "C": 45,
      "Ca": 46,
      "Cd": 47,
      "Ce": 48,
      "Cf": 49,
      "Cl": 50,
      "Cm": 51,
      "Cn": 52,
      "Co": 53,
      "Cr": 54,
      "Cs": 55,
      "Cu": 56,
      "Db": 57,
      "Ds": 58,
      "Dy": 59,
      "Er": 60,
      "Es": 61,
      "Eu": 62,
      "F": 63,
      "Fe": 64,
      "Fl": 65,
      "Fm": 66,
      "Fr": 67,
      "Ga": 68,
      "Gd": 69,
      "Ge": 70,
      "H": 71,
      "He": 72,
      "Hf": 73,
      "Hg": 74,
      "Ho": 75,
      "Hs": 76,
      "I": 77,
      "In": 78,
      "Ir": 79,
      "K": 80,
      "Kr": 81,
      "La": 82,
      "Li": 83,
      "Lr": 84,
      "Lu": 85,
      "Lv": 86,
      "Mc": 87,
      "Md": 88,
      "Mg": 89,
      "Mn": 90,
      "Mo": 91,
      "Mt": 92,
      "N": 93,
      "Na": 94,
      "Nb": 95,
      "Nd": 96,
      "Ne": 97,
      "Nh": 98,
      "Ni": 99,
      "No": 100,
      "Np": 101,
      "O": 102,
      "Og": 103,
      "Os": 104,
      "P": 105,
      "Pa": 106,
      "Pb": 107,
      "Pd": 108,
      "Pm": 109,
      "Po": 110,
      "Pr": 111,
      "Pt": 112,
      "Pu": 113,
      "Ra": 114,
      "Rb": 115,
      "Re": 116,
      "Rf": 117,
      "Rg": 118,
      "Rh": 119,
      "Rn": 120,
      "Ru": 121,
      "S": 122,
      "Sb": 123,
      "Sc": 124,
      "Se": 125,
      "Sg": 126,
      "Si": 127,
      "Sm": 128,
      "Sn": 129,
      "Sr": 130,
      "Ta": 131,
      "Tb": 132,
      "Tc": 133,
      "Te": 134,
      "Th": 135,
      "Ti": 136,
      "Tl": 137,
      "Tm": 138,
      "Ts": 139,
      "U": 140,
      "V": 141,
      "W": 142,
      "Xe": 143,
      "Y": 144,
      "Yb": 145,
      "Zn": 146,
      "Zr": 147,
      "[": 148,
      "\\": 149,
      "]": 150,
      "as": 151,
      "b": 152,
      "c": 153,
      "n": 154,
      "o": 155,
      "p": 156,
      "s": 157,
      "se": 158,
    )

    table(
      columns: 16,
      stroke: 0.5pt,
      align: center + horizon,
      ..vocab.pairs().map(((token, id)) => [#token \ #id]).flatten(),
    )
  },
  caption: [Complete SMIRK tokenizer vocabulary of 165 unique tokens (after including the 7 special tokens ([BOS], [EOS], [UNK], [SEP], [PAD], [CLS], [MASK])],
) <SMIRK_illustration>

=== Atom-Based Tokenization
To inject chemical awareness directly into the tokenization process, methodologies such as SMILES Pair Encoding (SPE) @li2021smiles and Atom Pair Encoding (APE) @leon2024comparing were developed. These algorithms treat bracketed atoms with stereochemical or isotopic specifications (e.g., `[C@@]` or `[NH4+]`) as indivisible units, using the regular expression `r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"`, rather than splitting them into individual characters like `[`, `C`, `@`, `@`, `]`; and then apply statistical merging algorithms similar to BPE on top of this atom-centric vocabulary, as illustrated in @SPE_illustration.

#figure(
  image("images/SPE_illustration.jpeg"),
  caption: [Illustration of the SPE tokenization process @li2021smiles.],
) <SPE_illustration>

While APE and SPE successfully preserve the integrity of individual atoms, their reliance on sequential pairwise merging exposes a fundamental limitation of the SMILES notation itself. Because SMILES linearizes 2D molecular graphs using a DFS traversal, molecular substructures in the 2D graph might be separated by long sequences of intervening characters in the 1D string. Consequently, statistical merging of adjacent 1D tokens might fail to capture true 2D chemical semantics because it merges tokens based on their proximity in the DFS traversal rather than their spatial connectivity in the actual molecule, creating sub-word tokens that are not meaningful chemical fragments. Overcoming this limitation requires moving toward explicit fragment-based approaches that partition the underlying 2D molecular graph prior to tokenization.

== Fragment-Based Tokenization

=== 2D Partitioning and Fragment Assembly
To overcome the limitations of 1D sequential merging, recent frameworks prioritize the 2D topology of the molecule by partitioning the graph into chemically meaningful substructures (fragments) prior to sequence conversion. Historically, this partitioning relied on hand-crafted chemical rules, such as the Breaking Retrosynthetically Interesting Chemical Substructures (BRICS) algorithm, which systematically cleaves molecules at specific synthetically accessible bonds.

Once partitioned, the challenge lies in serializing these disjointed 2D fragments back into a contiguous 1D sequence similar to SMILES. Several contemporary frameworks have proposed distinct assembly strategies:
- *Sequential Attachment-based Fragment Embedding (SAFE) @noutahi2024gotta:* SAFE represents the molecule as an unordered sequence of fragment blocks, separated by SMILES non-bonding notations (e.g., `.`). It also utilizes SMILES ring-closure digits to indicate attachment points between disconnected fragments.
#figure(
  image("images/SAFE_illustration.gif"),
  caption: [Illustration of the SAFE representation, where the colored fragments are more easily identifiable as compared to the original SMILES string @noutahi2024gotta.],
) <SAFE_illustration>
- *t-SMILES @wu2024t @wu2024hierarchical:* t-SMILES transforms the fragment-based molecular graph into either an Acyclic Molecular Tree (AMT) or a Full Binary Tree (FBT). It introduces two novel syntactic symbols: `&` to represent dummy fragment nodes within the FBT, and `^` to separate adjacent fragments. In total, the framework introduced seven distinct variants characterized by their connection logic and traversal strategies. The three original variants employ a breadth-first search (BFS) traversal over the FBT: TSSA (shared atoms), TSDY (dummy atoms without IDs), and TSID (dummy atoms with IDs). Building upon these, four newer, simplified variants (TSIS, TSISD, TSISO, and TSISR) were introduced to explore alternative traversal patterns: BFS, DFS, decreasing fragment size, and random ordering, respectively. All four of these simplified variants utilize dummy atoms with IDs to show fragment connectivity.
#figure(
  image("images/tSMILES_illustration.png", width: 70%),
  caption: [Illustration of the t-SMILES representation @wu2024hierarchical.],
) <tSMILES_illustration>
- *fragSMILES @mastrolorito2025fragsmiles:* fragSMILES starts with the longest path in the fragment-based graph as the base sequence and then recursively builds branches off this main path. It introduces the `<id>` notation to indicate the connection points between fragments.
#figure(
  image("images/fragSMILES_illustration.png"),
  caption: [Illustration of the fragSMILES representation @mastrolorito2025fragsmiles.],
) <fragSMILES_illustration>

=== Tokenization Strategies for Fragment Assembly
After serializing the fragments into a 1D string, the tokenizer must map each fragment to a discrete token ID. A critical bottleneck in fragment-based representations is ensuring vocabulary consistency. Ideally, identical chemical fragments should map to the exact same token ID regardless of where they appear in a molecule's topology.

However, frameworks such as SAFE and certain t-SMILES variants (i.e., using dummy atoms with IDs) embed the connection syntax directly inside the fragment string. For example, when 4-Methylbiphenyl (SMILES: `Cc1ccc(cc1)c2ccccc2`) is split using BRICS and serialized using SAFE, the two identical benzene rings (i.e., `c1ccccc1`) within the same molecule might be represented as entirely different string tokens `c13ccc4cc1.c14ccccc1.C3`. This forces the vocabulary to include multiple redundant tokens for the same chemical fragment, artificially inflating the vocabulary size and making it more difficult for the model to learn generalizable chemical semantics.

To resolve this issue, tokenization strategies need to decouple the connection logic from the fragment syntax. Frameworks like fragSMILES achieve this by placing connection markers (i.e., `<id>` tags) outside the fragment string. It is a possible experiment for this project to explore modifications to the SAFE or t-SMILES framework to similarly decouple the connection syntax, and evaluate how this impacts downstream performance.

== Data-Driven Fragment Extraction

=== Limitations of Rule-Based Fragmentation
While fragment-based tokenizers excel at capturing high-level chemical semantics, their reliance on hand-crafted fragmentation algorithms might introduce non-representative fragments of the training corpus into the vocabulary. Because these fragmentation rules are based on predefined chemical intuition rather than the actual structural distribution of the dataset being modeled, they can fail to capture the most frequent patterns within a specific corpus @kong2022molecule.

Furthermore, because the chemical space is effectively infinite, rule-based algorithms can encounter novel fragments during downstream tasks that were absent in the pretraining phase. These unseen fragments are forcefully mapped to an `[UNK]` token, resulting in a direct loss of critical predictive information.

=== Principal Subgraph Extraction Algorithm
To better adapt to dataset-specific distributions of chemical substructures and learn a more robust fragment vocabulary, recent literature has explored data-driven vocabulary extraction. The Principal Subgraph Variational Auto-Encoder (PS-VAE) proposes a framework to dynamically extract a vocabulary of principal subgraphs based on their statistical frequency within the pretraining corpus @kong2022molecule, instead of relying on hand-crafted fragmentation rules.

The extraction algorithm operates similar to the BPE algorithm, but acts directly on the 2D molecular graph rather than a 1D string. Starting from an initial vocabulary of individual atoms and bonds, the merge-and-update algorithm iteratively identifies the most frequently co-occurring adjacent subgraphs and merges them into a new, larger principal subgraph. Combining this approach with a fragment-based assembly strategy provides a promising experiment for a data-driven fragment-based tokenization scheme.

= Current Progress

== Methodology
To systematically evaluate the tokenization strategies outlined in the literature review, this project employs a three-stage experimental pipeline: Tokenizer Construction, Self-Supervised Pretraining, and Downstream Finetuning. This methodology closely mirrors the evaluation framework established in other comparative studies, notably the Atom Pair Encoding and ChemBERTa setups @leon2024comparing @chithrananda2020chemberta.

=== Tokenizer Construction
With the exception of the heavily compute-bound APE baseline (which utilized the original authors' pretrained vocabulary), all tokenization algorithms were trained from scratch to ensure a controlled comparative environment. The tokenizers were trained on a randomly sampled corpus of 2 million (2M) molecules from the PubChem10M dataset @chithrananda2020chemberta. Furthermore, the algorithms were configured to extract target vocabulary sizes of 5,000 (5K) tokens, utilizing a standardized set of special tokens ([BOS], [EOS], [UNK], [SEP], [PAD], [CLS], [MASK]) across all tokenizers.

=== Self-Supervised Pretraining
Following vocabulary construction, sequence-based CLMs were pretrained using the MLM objective. The underlying architecture for all experiments is a tailored RoBERTa model configured with 6 hidden layers, 12 attention heads, a hidden state dimension of 768, an intermediate feed-forward size of 1536, and a maximum sequence length of 514 tokens.

Pretraining was conducted on a disjoint subset of the PubChem10M dataset, partitioned into 1,000,000 training (1M), 100,000 validation (100K), and 100,000 testing (100K) molecules. Each model was trained for 20 epochs with a batch size of 64, and a learning rate of $5 times 10^(-5)$ (linear schedule with warmup ratio of 0.1) using the AdamW optimizer. During training, a dynamic masking probability of 15% was applied to the input sequences.

=== Downstream Finetuning
To evaluate the semantic richness and generalization capabilities of the learned representations, the pretrained CLMs were subsequently fine-tuned on task-specific molecular property prediction benchmarks. Specifically, the models were evaluated on the BBBP (Blood-Brain Barrier Penetration) and HIV (Human Immunodeficiency Virus) binary classification datasets from the MoleculeNet suite @wu2018moleculenet.

To ensure reproducibility, the dataset partitioning strictly followed the established train/validation/test splits provided by the DeepChem library @Ramsundar-et-al-2019. The fine-tuning phase was executed for 5 epochs with similar hyperparameters to the pretraining phase. The main metric for downstream performance evaluation is the Area Under the Receiver Operating Characteristic Curve (ROC-AUC), which provides a robust measure of the model's ability to discriminate between positive and negative classes across varying classification thresholds. By holding the model architecture and hyperparameters completely constant across all runs, any variance in downstream predictive performance can be attributed to the differences in tokenization strategies.

== Experimental Results
Following the pretraining and finetuning of various distinct model configurations, the test ROC-AUC scores on the HIV and BBBP datasets were aggregated. Rather than analyzing each run in isolation, the results were evaluated across five primary tokenization paradigms to identify overarching trends. A summary of the top-performing configurations within each baseline category is presented in @experimental_result_summary_table. #footnote[Full experimental logs are available at #link("https://wandb.ai/ncduy0303/molecule-tokenization"). The corresponding implementation pipeline can be found at #link("https://github.com/ncduy0303/molecule-tokenization").]

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    inset: (x: 8pt, y: 4pt),
    table.header([Category], [Top Configuration], [HIV ROC], [BBBP ROC]),
    table.cell(rowspan: 1)[Character],
    [SMILY-SBPE], [0.7726], [*0.7452*],
    table.cell(rowspan: 1)[Glyph],
    [SMILY-SMIRK-SPCATT], [*0.7730*], [0.7212],
    table.cell(rowspan: 1)[Atom],
    [SMILY-SPE], [0.7470], [0.7292],
    table.cell(rowspan: 2)[Fragment (t-SMILES)],
    [TSMILES-TSSA-M], [0.7192], [0.6867],
    [TSMILES-TSDY-M], [0.7014], [0.7002],
    table.cell(rowspan: 1)[Fragment (fragSMILES)],
    [FRAGSMILES], [0.7506], [0.7204],
  ),
  caption: [Summary of top-performing configurations across the five tokenization categories on downstream HIV and BBBP datasets.],
) <experimental_result_summary_table>


=== The Effect of Structural Pre-Splitting (Character/Glyph)
By first decomposing SMILES strings using the regular expression `r"\.|%\d{2}|[\(\)]|[/\\]|\[.*?]|\d"`, structural pre-splitting reduces the vocabulary diversity before applying BPE or PCATT tokenization. This two-stage approach (producing SBPE and SPCATT) yields better memory efficiency by decreasing the number of unique tokens that tokenization algorithms must process @wadell2026tokenization. This might also improve downstream performance especially for character-based tokenization, while no significant improvement is observed for glyph-based tokenization (@structural_pre_splitting_table).

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    inset: (x: 8pt, y: 4pt),
    table.header([Configuration], [HIV ROC], [BBBP ROC]),
    [SMILY-BPE], [0.7370], [0.7154],
    [SMILY-SBPE], [0.7726], [0.7452],
    table.hline(stroke: 0.1pt),
    [SMILY-SMIRK-GPE], [0.7301], [0.6850],
    [SMILY-SMIRK-SGPE], [0.7360], [0.6785],
  ),
  caption: [Performance comparison of structural pre-splitting configurations on character and glyph-based BPE tokenization.],
) <structural_pre_splitting_table>

=== SMILES vs. SAFE (Character/Glyph/Atom)
Because SAFE representations utilize the standard SMILES alphabet and are fundamentally valid SMILES strings @noutahi2024gotta, identical character, glyph, and atom-based tokenization algorithms were applied to both formats for comparative evaluation. As shown in @smiles_vs_safe_table, across matched configurations, standard SMILES representations consistently outperformed SAFE representations using the same tokenization strategy on downstream property prediction tasks.

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    inset: (x: 8pt, y: 4pt),
    table.header([Configuration], [$Delta$ HIV ROC], [$Delta$ BBBP ROC]),
    [BPE], [0.0007], [-0.0566],
    [SBPE], [-0.0365], [-0.0725],
    [SPCATT], [0.0036], [-0.0797],
    table.hline(stroke: 0.1pt),
    [SMIRK], [-0.0049], [-0.0573],
    [SMIRK-GPE], [-0.0211], [-0.0519],
    [SMIRK-SGPE], [0.0001], [-0.0600],
    [SMIRK-SPCATT], [-0.0248], [-0.0050],
    table.hline(stroke: 0.1pt),
    [APE], [0.0148], [-0.0399],
    [SPE], [-0.0148], [-0.0568],
  ),
  caption: [Performance delta when transitioning from standard SMILES to SAFE. Negative $Delta$ values indicate that the SAFE strings underperform compared to the original SMILES strings.],
) <smiles_vs_safe_table>

=== The Effect of Fragment Assembly and Fragmentation Rules on t-SMILES
To isolate the effects of fragment assembly and fragmentation logic, a focused evaluation was conducted on eight specific t-SMILES configurations (utilizing a 5K vocabulary where each fragment separated by `^` is treated as an indivisible token, ordered by decreasing frequency in the pretraining corpus). This allows for a direct comparison of how different connection grammars and fragmentation rules impact downstream performance. These configurations cross-examined four connection variants (TSSA, TSDY, TSID, TSIS) with two distinct 2D fragmentation algorithms: BRICS and Matched Molecular Pair Analysis (MMPA)#footnote[MMPA fragmentation rules utilize the RDKit implementation: #link("https://www.rdkit.org/docs/source/rdkit.Chem.rdMMPA.html")].

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    inset: (x: 8pt, y: 4pt),
    table.header([Configuration], [Fragmentation Rule], [HIV ROC], [BBBP ROC]),
    [TSSA], [BRICS], [0.6025], [0.6197],
    [TSDY], [BRICS], [0.6052], [0.6270],
    [TSID], [BRICS], [0.5564], [0.5682],
    [TSIS], [BRICS], [0.5390], [0.5660],
    table.hline(stroke: 0.1pt),
    [TSSA-M], [MMPA], [0.7192], [0.6867],
    [TSDY-M], [MMPA], [0.7014], [0.7002],
    [TSID-M], [MMPA], [0.6854], [0.6746],
    [TSIS-M], [MMPA], [0.7011], [0.6549],
  ),
  caption: [Performance comparison of t-SMILES configurations evaluating the impact of connection grammar (TSSA/TSDY vs. TSID/TSIS) and molecular fragmentation rules (BRICS vs. MMPA).],
) <tsmiles_evaluation_table>

- *The Impact of Fragmentation Assembly:* As highlighted in the literature review, variants like TSID and TSIS embed explicit connection indices directly inside the fragment tokens. This causes severe vocabulary redundancy, as identical structural motifs with different connection IDs are treated as entirely distinct tokens by the language model. In contrast, TSSA (which utilizes shared atoms) and TSDY (which utilizes dummy atoms without IDs) simplify this connection logic. This theoretical advantage is empirically validated in @tsmiles_evaluation_table where across matched configurations, TSSA and TSDY consistently outperform TSID and TSIS.
- *The Impact of Fragmentation Rules:* The choice of the underlying 2D graph fragmentation algorithm may also impact downstream performance. As shown in @tsmiles_evaluation_table, transitioning from BRICS to MMPA fragmentation yields a notable increase in performance across the evaluated variants. This confirms that the initial fragmentation step acts as a critical inductive bias; which requires careful consideration to obtain optimal downstream performance.

=== Comparing Fragment-Based Tokenization: t-SMILES vs. fragSMILES
As shown in @experimental_result_summary_table, the fragSMILES configuration significantly outperforms the best-performing t-SMILES configurations (TSSA-M and TSDY-M) on both the HIV and BBBP datasets. This performance gap might be attributed to two primary hypotheses concerning the core differences between the frameworks:
- *Fragmentation Rules:* t-SMILES framework relies on external rules like BRICS or MMPA by default. In contrast, the fragSMILES framework cleaves the molecule at all exocyclic single bonds @mastrolorito2025fragsmiles. This approach may yield a more consistent and statistically uniform fragment distribution that is easier for representation learning.
- *Notation Reconstructability:* While TSSA and TSDY improve upon TSID for fragment tokenization by removing the embedded connection IDs, they sacrifice exact graph reconstructability. Because the explicit connection points are removed, a TSSA or TSDY string might map back to multiple valid molecular graphs. fragSMILES, however, retains the explicit `<id>` tags during its recursive graph traversal, ensuring a 1:1 mapping between the 1D string and the original 2D molecule without introducing the vocabulary redundancy of TSID.

=== Data-Driven Fragmentation: PS-FRAGSMILES
To address the limitations inherent to rule-based fragmentation, the data-driven Principal Subgraph (PS-VAE) extraction algorithm was integrated into the fragSMILES framework (forming PS-FRAGSMILES). Because the fragSMILES sequence assembly recursively builds branches and inherently expects an acyclic molecular graph, a critical algorithmic modification was required. The tokenization pipeline was adapted to first isolate and collapse all ring structures into indivisible single nodes prior to subgraph mining. This forced the resulting fragment structures to remain acyclic, maintaining compatibility with the fragSMILES library. Despite this structural adaptation, preliminary evaluations (@ps_fragsmiles_table) indicate that the data-driven PS-FRAGSMILES tokenizer currently underperform the default rule-based FRAGSMILES tokenizer.

#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    inset: (x: 8pt, y: 4pt),
    table.header([Configuration], [Tokenizer Corpus Size], [HIV ROC], [BBBP ROC]),
    [FRAGSMILES], [2,000,000], [0.7506], [0.7204],
    [PS-FRAGSMILES], [20,000], [0.7269], [0.7117],
  ),
  caption: [Performance comparison between FRAGSMILES and PS-FRAGSMILES. The PS-FRAGSMILES tokenizers were constrained to a much smaller training corpus due to computational bottlenecks of the PS-VAE library.],
) <ps_fragsmiles_table>

*Implications for Future Work:* In addition to optimizing the PS-VAE code to process a larger corpus, future experiments will also explore integrating PS-VAE with the `t-SMILES` framework. Because t-SMILES naturally resolves cyclic structures by constructing an Acyclic Molecular Tree prior to sequence traversal, it can handle arbitrary cyclic subgraphs. Integrating PS-VAE with t-SMILES will completely eliminate the need for the ad-hoc "ring collapsing" modifications currently required by fragSMILES, theoretically allowing for a more compact vocabulary.

= Research Plan (Next Semester)

Based on the empirical insights gathered during the first semester, the research plan for the remainder of the FYP will focus on bridging the performance gaps between current fragment-based tokenizations compared to the standard baselines.

== Weeks 1 - 3
- *Standardizing Fragmentation Rules:* Standardize the fragmentation rules used by t-SMILES and fragSMILES to decouple the fragmentation logic from the sequence assembly logic, allowing for a 1:1 comparison of their respective notations.
- *Optimizing PS-VAE:* Scale the PS-FRAGSMILES tokenizer to the full 2M pretraining corpus for a 1:1 comparison against the default FRAGSMILES tokenizer.
- *Developing PS-TSMILES:* Integrate the newly optimized PS-VAE extraction logic directly into the t-SMILES framework, and evaluate the resulting PS-TSMILES tokenizer on downstream benchmarks.

== Weeks 4 - 6
- *Augmenting t-SMILES Representation:* Design, implement and evaluate a modification to the t-SMILES framework to move the connection syntax outside the fragment string, similar to the approach used by fragSMILES. This augmentation aims to externalize the connection logic from the core fragment string, preventing vocabulary redundancy (unlike TSID) while ensuring strict 1:1 graph reconstructability (unlike TSSA/TSDY).

== Weeks 7 - 12
- *Comprehensive Benchmarking:* Execute downstream finetuning across a broader suite of MoleculeNet datasets as well as additional intrinsic evaluation metrics such as fertility or UNK token rate for a more holistic assessment of the tokenization strategies @wadell2026tokenization.
- *Documentation and Thesis Writing:* Analyze the results, conduct ablation studies, and compile the findings into the final report.

// ==========================================
// 3. REFERENCES
// ==========================================
#pagebreak()

= References

#bibliography("main.bib", style: "apa", title: none)
