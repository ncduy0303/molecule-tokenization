#import "../metadata.typ": *
#align(center)[
  #set par(justify: false)
  #v(1.5cm)
  B.Comp. Dissertation
  #v(1.5cm)
  #text(size: 16pt, weight: "bold")[#title]
  #v(3cm)
  By

  #author
  #v(3cm)
  Department of Computer Science

  School of Computing

  National University of Singapore

  #academic-year
  #v(3cm)
  #grid(
    columns: (auto, 1fr),
    gutter: 12pt,
    align: left,
    [Project No:], [#project-id],
    [Advisor:], [#supervisor],
    [Deliverables:], [Report: 1 Volume \ Program: 1 GitHub Repository],
  )
]
