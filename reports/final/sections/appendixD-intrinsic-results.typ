#import "../template.typ": metric-row-fill
#heading(numbering: none)[Appendix D: Intrinsic results by configuration]

Each pair of tables covers one input group. Length, fertility, and UNK measures use the successfully encoded rows.

#let records = csv("../data/intrinsic-by-split.csv", row-type: dictionary)
#let splits = (
  ("pubchem", "PubChem"),
  ("hiv_train", "HIV training"),
  ("hiv_validation", "HIV validation"),
  ("hiv_test", "HIV test"),
  ("bbbp_train", "BBBP training"),
  ("bbbp_validation", "BBBP validation"),
  ("bbbp_test", "BBBP test"),
)
#for (split, title) in splits {
  if split != "pubchem" { pagebreak() }
  set par(leading: 2pt, spacing: 5pt)
  set figure(gap: 4pt)
  strong[#title]
  for (family, label) in (("string", "SMILES and SAFE tokenizers"), ("fragment", "Fragment tokenizers")) {
    let rows = records.filter(r => r.split == split and r.family == family)
    figure(
      table(
        columns: (2.5fr, 1fr, 1fr, 1fr, 1fr, 1fr),
        fill: metric-row-fill,
        align: (left, right, right, right, right, right),
        inset: (x: 5pt, y: 2.8pt),
        table.header(
          [*Tokenizer*],
          [*Encoded rows*],
          [*Mean tokens used*],
          [*Fertility*],
          [*UNK token rate (%)*],
          [*Molecules with UNK (%)*],
        ),
        ..rows.map(r => (r.name, r.encoded, r.length, r.fertility, r.unk, r.molunk)).flatten(),
      ),
      caption: [#label on the #title inputs (#(rows.first().attempted) attempted rows).],
    )
  }
}
