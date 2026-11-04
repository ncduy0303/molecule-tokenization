#import "../template.typ": metric-row-fill
= Experimental methodology

== Benchmark comparisons

*Initial benchmark.* The initial experiments compared character-, glyph-, atom-, and fragment-based tokenization on HIV and BBBP. Corresponding SMILES and SAFE configurations used the same tokenization method to test the effect of changing the input representation. For t-SMILES, each of TSSA, TSDY, TSID, and TSIS was evaluated with both BRICS and MMPA fragmentation. The main comparison used a target vocabulary of 5,000 entries for learned tokenizers. The APE baseline used the published vocabulary @leon2024comparing.

*Shared fragmentation.* Four representations (fragSMILES, TSDY, TSID, and TSIS) used the same exocyclic partition. Each was tested with and without principal-subgraph (PS) merging, giving eight conditions. Comparing the two conditions within a representation measures the effect of enabling merges. Comparing representations within either condition holds the atom partition fixed.

*Enhanced TSDY.* The comparison evaluates base and enhanced TSDY on HIV and BBBP from MoleculeNet @wu2018moleculenet. Both use the same exocyclic partition without merging, so the fragments are held fixed while their encoding changes.

== Vocabulary construction and pretraining

*Corpus selection.* The controlled experiments used canonical SMILES from PubChem10M#footnote[Dataset: #link("https://huggingface.co/datasets/mikemayuare/PubChem10M_SMILES_SELFIES")[PubChem10M SMILES/SELFIES].]. A fixed sample of 2,000,000 rows (seed 42) supplied the tokenizer corpus. The MLM sampler excluded these rows and selected 1,000,000 training, 100,000 validation, and 100,000 test rows. All controlled representations used these selections.

*Vocabulary budget.* The shared and enhanced experiments each allocated 5,000 entries per tokenizer, including special tokens. The PS miner used the tokenizer corpus, a budget of 1,000 merge rules, and a minimum frequency of 10. The same learned rules were applied to all four PS conditions. For enhanced TSDY, 81 grammar tokens and seven special tokens occupy fixed entries. The remaining 4,912 entries hold the most frequent chemical cores in the tokenizer corpus.

*Model training.* Each controlled tokenizer had its own RoBERTa encoder @liu2019roberta pretrained from scratch with seed 42. The encoder had six layers, 12 attention heads, hidden size 768, and feed-forward size 1,536. Training used the Hugging Face Transformers Trainer @wolf2020transformers with the settings in @method-training. Dynamic masking selected 15% of eligible positions as prediction targets. Tokenizer construction ran on CPU nodes (AMD EPYC 7763), while model training ran on GPU nodes (NVIDIA H100 96GB) on the NUS School of Computing cluster using Slurm (Appendix B).

#figure(
  table(
    fill: metric-row-fill,
    columns: (1.3fr, 1fr, 1fr),
    align: (left, center, center),
    inset: 5pt,
    table.header([*Setting*], [*MLM*], [*Finetuning*]),
    [Epochs], [20], [Up to 5 (early stopping)],
    [Batch size per device], [64], [64],
    [Learning rate], [$5 times 10^(-5)$], [$5 times 10^(-5)$],
    [Optimizer], [AdamW], [AdamW],
    [Weight decay], [0.01], [0.01],
    [Learning-rate schedule], [Linear], [Linear],
    [Warmup fraction], [0.1], [0.1],
  ),
  caption: [Training settings for the controlled comparisons.],
) <method-training>

Model inputs allowed 510 content tokens plus two boundary tokens. Longer sequences were truncated for model training and evaluation. Vocabulary fitting and intrinsic length measurements used the complete sequence.

== Downstream evaluation

*Data splits.* The binary labels indicate whether a compound inhibits HIV replication or crosses the blood-brain barrier, respectively @wu2018moleculenet. HIV and BBBP used DeepChem's scaffold splitter with target proportions of 80/10/10 @Ramsundar-et-al-2019. This assigns molecules with the same scaffold to the same split. Keeping scaffold groups intact gives the counts in @method-splits.

#figure(
  table(
    fill: metric-row-fill,
    columns: (2.5fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    inset: 5pt,
    table.header([*Task and protocol*], [*Train*], [*Validation*], [*Test*], [*Total*]),
    [HIV], [32,901], [4,113], [4,106], [41,120],
    [BBBP], [1,640], [205], [194], [2,039],
  ),
  caption: [Number of molecules in each downstream data split.],
) <method-splits>

*Model selection and scoring.* Each classification run started from the final MLM weights for its representation. The area under the receiver operating characteristic curve (ROC-AUC) measures how often the model ranks a positive example above a negative one, counting ties as half. Validation ROC-AUC was evaluated each epoch, with early stopping after three evaluations without improvement. The checkpoint with the highest validation ROC-AUC supplied the test predictions.

The downstream protocol uses three finetuning seeds (42, 43, and 44) per configuration. The split and pretrained encoder are fixed across trials. Each comparison reports the mean and sample standard deviation of the three test ROC-AUC values.

== Intrinsic evaluation

The intrinsic comparison in Appendix D covers 33 tokenizer configurations. Each was evaluated on the same 53,159 input rows: 10,000 PubChem rows sampled from a saved test corpus (seed 42), 41,120 HIV rows, and 2,039 BBBP rows. The audit used each tokenizer's saved vocabulary and recorded metrics separately for PubChem and for each HIV and BBBP split. Appendix D presents all 33 configurations for each input group. Section 5.4 reports selected comparisons on the downstream test splits.

=== Encoding and vocabulary metrics
- *Encoding success rate.* Fraction of attempted rows that produced a token sequence successfully. The strict SAFE audit counted conversion failures even when the historical training pipeline would have used a SMILES fallback.
- *Mean sequence length.* Mean number of content tokens per successfully encoded molecule, measured before truncation. Connection and tree tokens count towards length. Model boundary tokens and padding are excluded.
- *Token fertility (tokens per heavy atom).* Total content tokens divided by total non-hydrogen atoms across the successfully encoded molecules. This measures sequence length relative to molecular size.
- *UNK token rate.* Number of positions assigned the unknown-token ID divided by the number of content tokens across all successfully encoded molecules.
- *Molecules containing UNK rate.* Fraction of successfully encoded molecules with at least one unknown token. Each affected molecule counts once, regardless of how many unknown tokens it contains.

Paired comparisons used rows that both tokenizers encoded successfully.

=== Fragment sharing
- *Decorated forms per core.* For base and enhanced TSDY, fragments were paired by their atoms in the shared partition. For each clean core, the audit counted the distinct base TSDY fragment strings that mapped to it.
- *Fragment UNK rate.* Unknown fragment occurrences divided by all matched fragment occurrences. Excluding tree and attachment tokens gives both representations the same denominator.
