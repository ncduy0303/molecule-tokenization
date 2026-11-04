#import "../template.typ": metric-row-fill
= Results and discussion

== Initial benchmark of existing tokenizers

*String baselines.* Plain SMIRK on SMILES had the highest observed mean test ROC-AUC among the 25 initial configurations on both HIV (0.7755 $plus.minus$ 0.0120) and BBBP (0.7467 $plus.minus$ 0.0082). Adding structurally pre-split glyph-pair encoding (SMIRK-SGPE) gave a nearly identical HIV mean (0.7752 $plus.minus$ 0.0199). Two comparisons using ordinary SMILES BPE as the baseline separate changes to the tokenizer from changes to the input representation.

- *Change the tokenization rule.* Adding structural pre-splitting#footnote[During vocabulary training, structural pre-splitting isolates branch parentheses, ring numbers, bracketed atoms, component separators, and directional bond marks. For example, `CC(=O)O` yields the spans `CC`, `(`, `=O`, `)`, and `O`. BPE learns merges within these spans, while PCATT uses their counts to learn vocabulary entries.] increased mean test ROC-AUC slightly on HIV (+0.0026) and more on BBBP (+0.0222).
- *Change the molecular representation.* Keeping the BPE training rule fixed, switching from SMILES to SAFE lowered the mean on both HIV (−0.0042) and BBBP (−0.0423). SAFE had a lower BBBP mean in all eight paired comparisons. On HIV, APE was the only tokenizer for which SAFE had a higher mean (+0.0227).

Appendix C reports the full three-seed results.

*Fragmentation and assembly.* Switching the fragmentation method from BRICS to MMPA increased mean ROC-AUC for every t-SMILES notation on both tasks (@initial-ts). The effect of assembly notation varied by task. With BRICS, TSSA and TSDY both ranked above TSID and TSIS on both tasks. With MMPA, TSIS had the highest HIV mean (0.7268), while TSDY had the highest BBBP mean (0.6954).

TSID and TSIS include numbered attachment sites in fragment words @wu2024t @wu2024hierarchical. Changing a connection label from `[1*]` to `[2*]` can split occurrences of the same motif across vocabulary entries. TSSA and TSDY avoid that source of duplication. This provides a reason to test vocabulary coverage directly. The reversal on MMPA HIV shows why the identifiers alone cannot explain the predictive ranking.

#figure(
  table(
    fill: metric-row-fill,
    columns: (0.6fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    inset: 4pt,
    table.header([*Notation*], [*HIV (BRICS)*], [*HIV (MMPA)*], [*BBBP (BRICS)*], [*BBBP (MMPA)*]),
    [TSSA],
    [$0.6015 plus.minus 0.0162$],
    [$0.7015 plus.minus 0.0189$],
    [$0.6175 plus.minus 0.0325$],
    [$0.6776 plus.minus 0.0042$],

    [TSDY],
    [$0.6147 plus.minus 0.0164$],
    [$0.7007 plus.minus 0.0248$],
    [$0.6259 plus.minus 0.0054$],
    [$0.6954 plus.minus 0.0058$],

    [TSID],
    [$0.5766 plus.minus 0.0190$],
    [$0.7054 plus.minus 0.0337$],
    [$0.5538 plus.minus 0.0281$],
    [$0.6874 plus.minus 0.0047$],

    [TSIS],
    [$0.5586 plus.minus 0.0081$],
    [$0.7268 plus.minus 0.0212$],
    [$0.5603 plus.minus 0.0339$],
    [$0.6569 plus.minus 0.0102$],
  ),
  caption: [Initial test ROC-AUC by fragmentation rule and t-SMILES notation. Values are means $plus.minus$ sample standard deviations across finetuning seeds 42, 43, and 44.],
) <initial-ts>

*Comparison with fragSMILES.* The shared base fragSMILES reference from @shared-scores scored 0.7615 on HIV and 0.7184 on BBBP, above all the t-SMILES variants in @initial-ts. Two choices changed together: fragSMILES cut exocyclic single bonds and recorded connections outside its fragment words. The scores alone cannot distinguish their contributions because they do not use the same fragmentation rule (BRICS/MMPA vs. cutting exocyclic single bonds). Section 5.2 compares serializers on the same fragmentation rule and measures the effect of merging the fragments. Section 5.3 keeps the TSDY partition fixed and changes how attachment sites are encoded.

== Shared fragmentation and PS merging

With the base exocyclic partition, TSID had the highest mean HIV score and fragSMILES had the highest mean BBBP score (@shared-scores). PS merging reduced the mean BBBP score for every representation. On HIV, TSDY was the only representation whose mean increased after merging.

#figure(
  table(
    fill: metric-row-fill,
    columns: (1.1fr, 1fr, 1fr),
    align: (left, right, right),
    inset: (x: 6pt, y: 4pt),
    table.header([*Representation*], [*HIV ROC-AUC*], [*BBBP ROC-AUC*]),
    [fragSMILES], [$0.7615 plus.minus 0.0130$], [$0.7184 plus.minus 0.0102$],
    [PS-fragSMILES], [$0.7310 plus.minus 0.0112$], [$0.6911 plus.minus 0.0086$],
    [TSID], [$0.7632 plus.minus 0.0074$], [$0.7004 plus.minus 0.0026$],
    [PS-TSID], [$0.7342 plus.minus 0.0129$], [$0.6485 plus.minus 0.0214$],
    [TSDY], [$0.7181 plus.minus 0.0306$], [$0.6909 plus.minus 0.0123$],
    [PS-TSDY], [$0.7334 plus.minus 0.0264$], [$0.6827 plus.minus 0.0105$],
    [TSIS], [$0.7436 plus.minus 0.0139$], [$0.7089 plus.minus 0.0051$],
    [PS-TSIS], [$0.7188 plus.minus 0.0177$], [$0.6436 plus.minus 0.0188$],
  ),
  caption: [Test ROC-AUC with the shared exocyclic partition, before and after PS merging. Values are means $plus.minus$ sample standard deviations across three finetuning seeds.],
) <shared-scores>

The effect of merging also depended on the assembly notation. On BBBP, PS merging lowered mean ROC-AUC by 5.19 percentage points for TSID and 6.53 for TSIS, compared with 0.82 for TSDY. On HIV, the TSID and TSIS means also fell, while TSDY rose from 0.7181 to 0.7334. TSDY was less adversely affected by merging than TSID or TSIS on either task. The PubChem measurements in @intrinsic-pubchem support the coverage part of this explanation. After merging, the UNK token rate was 4.97% for TSDY, compared with 11.43% for TSID and 16.32% for TSIS. These rates describe vocabulary coverage. They do not isolate its effect on ROC-AUC.

== Enhanced TSDY on HIV and BBBP

Enhanced TSDY scored higher than base TSDY at each matched seed on both tasks. The mean gains were 1.91 ROC-AUC percentage points on HIV and 3.03 on BBBP (@enhanced-scores).

#figure(
  table(
    fill: metric-row-fill,
    columns: (1.1fr, 1fr, 1fr),
    align: (left, right, right),
    inset: (x: 6pt, y: 4pt),
    table.header([*Representation*], [*HIV ROC-AUC*], [*BBBP ROC-AUC*]),
    [Base TSDY#footnote[The HIV baseline was finetuned again on the same training rows as enhanced TSDY. The BBBP baseline is reused from @shared-scores.]],
    [$0.7363 plus.minus 0.0104$],
    [$0.6909 plus.minus 0.0123$],

    [Enhanced TSDY], [$0.7554 plus.minus 0.0120$], [$0.7212 plus.minus 0.0060$],
  ),
  caption: [Base and enhanced TSDY test ROC-AUC on HIV and BBBP. Values are means $plus.minus$ sample standard deviation. Both representations use the same molecules within each split.],
) <enhanced-scores>

*HIV training inputs.* Of the 32,901 HIV training rows, only source row 35278 could not be encoded by enhanced TSDY. RDKit parses two sulfur-to-iron bonds in this molecule as dative bonds, which the encoder does not support. @hiv-dative compares the dataset string with the canonical SMILES produced by RDKit.

#figure(
  grid(
    columns: 1,
    row-gutter: 8pt,
    align: left,
    [*Dataset SMILES*],
    [`C1C[N+]23CCC[N+]4(C1)CC[S-]1[Fe+2]24([SH+]CC3)`
      `[S-]2CC[N+]34CCC[N+]5(CCC3)CC[SH+][Fe+2]1254`],
    [*RDKit canonical SMILES*],
    [`C1C[N+]23CCC[N+]4(C1)CC[S-]1->[Fe+2]567[SH+]CC` \
      `[N+]58CCC[N+]6(CCC8)CC[S-]7->[Fe+2]124[SH+]CC3`],
  ),
  kind: image,
  caption: [HIV training row 35278. Each `->` in the canonical string marks a dative bond.],
) <hiv-dative>

Both TSDY models therefore used the same 32,900 training rows. All 4,113 validation rows and all 4,106 test rows passed the encoding check and were retained.

== Sequence length and vocabulary coverage

For TSDY, changing the fragmentation rule from BRICS to MMPA increased the mean PubChem sequence length from 9.01 to 21.08 tokens per molecule, but reduced the UNK token rate from 10.19% to 1.57%. The longer sequences used fragments that the vocabulary covered more often.

*Effect of PS merging.* Merging reduced token fertility for every shared representation, but increased the fraction of content tokens mapped to UNK (@intrinsic-pubchem). TSID and TSDY illustrate why both measures matter. Their fertility fell from 1.02 to 0.46 in both cases, yet the merged TSID vocabulary left 11.43% of token positions unknown compared with 4.97% for TSDY. The coverage difference is consistent with the cost of storing numbered attachment sites discussed in Section 5.1.

#figure(
  table(
    fill: metric-row-fill,
    columns: (1.1fr, 1fr, 1fr),
    align: (left, right, right),
    inset: 5pt,
    table.header([*Representation*], [*Fertility*], [*UNK token rate (%)*]),
    [fragSMILES], [$0.79 -> 0.46$], [$0.25 -> 0.48$],
    [TSID], [$1.02 -> 0.46$], [$3.11 -> 11.43$],
    [TSDY], [$1.02 -> 0.46$], [$0.72 -> 4.97$],
    [TSIS], [$0.82 -> 0.32$], [$3.83 -> 16.32$],
  ),
  caption: [Token fertility and UNK token rate on 10,000 PubChem molecules. Each arrow runs from the base value to the PS value. Fertility counts content tokens per heavy atom. The UNK token rate uses all content-token positions.],
) <intrinsic-pubchem>

*Sharing across attachment patterns.* The 10,000 PubChem molecules contained 108,850 fragment occurrences. Counting each distinct fragment spelling once gave 4,113 decorated forms in base TSDY and 1,815 clean cores in enhanced TSDY. For example, pyridine occurred 906 times across 40 decorated forms, all of which used the same enhanced core word, `c1ccncc1`. @sharing-coverage compares vocabulary coverage occurrence by occurrence, pairing the two encodings of the same group of atoms.

#figure(
  table(
    fill: metric-row-fill,
    columns: (0.9fr, 1fr, 1.2fr, 1.2fr),
    align: (left, right, right, right),
    inset: (x: 6pt, y: 4pt),
    table.header(
      [*Input set*],
      [*Fragment occurrences*],
      [*Base TSDY fragment UNK rate (%)*],
      [*Enhanced TSDY fragment UNK rate (%)*],
    ),
    [PubChem], [108,850], [1.71], [0.46],
    [HIV test], [31,240], [10.58], [6.78],
    [BBBP test], [1,873], [10.20], [2.08],
  ),
  caption: [Fragment UNK rates for matched base and enhanced TSDY fragments. Rates count repeated occurrences and exclude tree and attachment tokens.],
) <sharing-coverage>

On BBBP, the fragment UNK rate fell from 10.20% to 2.08%, even as mean sequence length rose from 23.44 to 28.30 tokens. The same core word can cover several attachment patterns, while the tokens that specify those attachments still occupy sequence positions.

#include "51-discussion.typ"
