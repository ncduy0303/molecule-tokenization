#import "../template.typ": metric-row-fill
#heading(numbering: none)[Appendix C: Downstream results by configuration]

Scores are mean test ROC-AUC $plus.minus$ sample standard deviation over finetuning seeds 42, 43, and 44. The table lists the 25 initial configurations and the shared base fragSMILES reference from @shared-scores. SBPE, SPCATT, and SGPE denote structurally pre-split BPE, PCATT, and glyph-pair encoding, respectively.

#figure(
  table(
    fill: metric-row-fill,
    columns: (0.85fr, 1.7fr, 1.2fr, 1.2fr),
    align: (left, left, right, right),
    inset: (x: 5pt, y: 3pt),
    table.header([*Family*], [*Configuration*], [*HIV ROC-AUC*], [*BBBP ROC-AUC*]),
    [Character], [SMILES BPE], [$0.7529 plus.minus 0.0190$], [$0.7136 plus.minus 0.0066$],
    [], [SMILES SBPE], [$0.7554 plus.minus 0.0253$], [$0.7358 plus.minus 0.0098$],
    [], [SMILES SPCATT], [$0.7688 plus.minus 0.0141$], [$0.7287 plus.minus 0.0028$],
    [], [SAFE BPE], [$0.7487 plus.minus 0.0171$], [$0.6713 plus.minus 0.0045$],
    [], [SAFE SBPE], [$0.7506 plus.minus 0.0200$], [$0.6725 plus.minus 0.0289$],
    [], [SAFE SPCATT], [$0.7413 plus.minus 0.0090$], [$0.6608 plus.minus 0.0317$],
    table.hline(),
    [Glyph], [SMILES SMIRK], [$0.7755 plus.minus 0.0120$], [$0.7467 plus.minus 0.0082$],
    [], [SMILES SMIRK-SGPE], [$0.7752 plus.minus 0.0199$], [$0.7416 plus.minus 0.0045$],
    [], [SMILES SMIRK-SPCATT], [$0.7672 plus.minus 0.0374$], [$0.7237 plus.minus 0.0129$],
    [], [SAFE SMIRK], [$0.7362 plus.minus 0.0169$], [$0.6692 plus.minus 0.0171$],
    [], [SAFE SMIRK-GPE], [$0.7440 plus.minus 0.0058$], [$0.6943 plus.minus 0.0105$],
    [], [SAFE SMIRK-SGPE], [$0.7379 plus.minus 0.0238$], [$0.6747 plus.minus 0.0417$],
    [], [SAFE SMIRK-SPCATT], [$0.7239 plus.minus 0.0039$], [$0.7172 plus.minus 0.0021$],
    table.hline(),
    [Atom], [SMILES APE], [$0.7409 plus.minus 0.0300$], [$0.7085 plus.minus 0.0095$],
    [], [SMILES SPE], [$0.7678 plus.minus 0.0128$], [$0.7219 plus.minus 0.0026$],
    [], [SAFE APE], [$0.7636 plus.minus 0.0037$], [$0.6728 plus.minus 0.0174$],
    [], [SAFE SPE], [$0.7197 plus.minus 0.0194$], [$0.6837 plus.minus 0.0181$],
    table.hline(),
    [Fragment], [TSSA BRICS], [$0.6015 plus.minus 0.0162$], [$0.6175 plus.minus 0.0325$],
    [], [TSSA MMPA], [$0.7015 plus.minus 0.0189$], [$0.6776 plus.minus 0.0042$],
    [], [TSDY BRICS], [$0.6147 plus.minus 0.0164$], [$0.6259 plus.minus 0.0054$],
    [], [TSDY MMPA], [$0.7007 plus.minus 0.0248$], [$0.6954 plus.minus 0.0058$],
    [], [TSID BRICS], [$0.5766 plus.minus 0.0190$], [$0.5538 plus.minus 0.0281$],
    [], [TSID MMPA], [$0.7054 plus.minus 0.0337$], [$0.6874 plus.minus 0.0047$],
    [], [TSIS BRICS], [$0.5586 plus.minus 0.0081$], [$0.5603 plus.minus 0.0339$],
    [], [TSIS MMPA], [$0.7268 plus.minus 0.0212$], [$0.6569 plus.minus 0.0102$],
    [], [Base fragSMILES], [$0.7615 plus.minus 0.0130$], [$0.7184 plus.minus 0.0102$],
  ),
  caption: [Initial benchmark test ROC-AUC by tokenizer family.],
) <initial-scores>

#figure(
  table(
    fill: metric-row-fill,
    columns: (1.5fr, 1fr, 1fr),
    align: (left, right, right),
    inset: (x: 5pt, y: 3pt),
    table.header([*Tokenizer*], [*HIV $Delta$ROC-AUC*], [*BBBP $Delta$ROC-AUC*]),
    [BPE], [−0.0042], [−0.0423],
    [SBPE], [−0.0048], [−0.0632],
    [SPCATT], [−0.0274], [−0.0679],
    [SMIRK], [−0.0393], [−0.0775],
    [SMIRK-SGPE], [−0.0373], [−0.0669],
    [SMIRK-SPCATT], [−0.0433], [−0.0065],
    [APE], [+0.0227], [−0.0357],
    [SPE], [−0.0481], [−0.0382],
  ),
  caption: [Initial mean test ROC-AUC differences (SAFE minus SMILES).],
) <initial-pairs>
