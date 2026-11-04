#import "metadata.typ": *
#import "template.typ": report
#show: report
#set document(title: title, author: author)
#set page(header: if draft { align(right, text(size: 9pt, fill: gray)[REVIEW DRAFT - NOT FOR SUBMISSION]) })
#include "sections/00-front-cover.typ"
#pagebreak()
#include "sections/00-title.typ"
#pagebreak()
#set page(numbering: "i", number-align: center)
#counter(page).update(1)
#include "sections/01-abstract.typ"
#pagebreak()
#outline(title: [Contents], depth: 2)
#pagebreak()
#set page(numbering: "1")
#counter(page).update(1)
#include "sections/10-introduction.typ"
#pagebreak()
#include "sections/20-background.typ"
#pagebreak()
#include "sections/30-design.typ"
#pagebreak()
#include "sections/40-methodology.typ"
#pagebreak()
#include "sections/50-results.typ"
#pagebreak()
#include "sections/60-conclusion.typ"
#pagebreak()
#bibliography("references.bib", style: "apa", title: [References])
#pagebreak()
#include "sections/appendixA-ai-assistance.typ"
#pagebreak()
#include "sections/appendixB-reproducibility.typ"
#pagebreak()
#include "sections/appendixC-benchmark-results.typ"
#pagebreak()
#include "sections/appendixD-intrinsic-results.typ"

#pagebreak()
#set page(numbering: none, header: none, footer: none)
#include "sections/99-back-cover.typ"
