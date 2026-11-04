#let metric-row-fill(_, row) = if row > 0 and calc.even(row) { rgb("#eaf1f5") }

#let review-note(body) = block(width: 100%, inset: 8pt, fill: luma(94%))[
  *Draft for review:* #body
]

#let report(body) = {
  set text(font: "Times New Roman", size: 12pt, lang: "en", top-edge: 0.7em, bottom-edge: 0.3em)
  set page(paper: "a4", margin: (left: 3cm, right: 3cm, top: 3cm, bottom: 3cm))
  set par(justify: true, leading: 6pt, spacing: 12pt)
  set heading(numbering: "1.1")
  show heading: it => block(above: 18pt, below: 9pt, it)
  show link: it => if type(it.dest) == str { text(fill: rgb("#164fa3"), it) } else { it }
  show table: set par(justify: false)
  set table(inset: 5pt, stroke: 0.4pt)
  body
}
