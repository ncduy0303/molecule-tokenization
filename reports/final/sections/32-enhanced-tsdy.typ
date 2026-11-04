== Enhanced TSDY

=== Core words and attachment sites
In standard TSDY, dummy atoms mark attachment sites inside each fragment string. A benzene ring with one substituent and a benzene ring with two substituents therefore occupy different entries in a whole-fragment vocabulary. Enhanced TSDY uses the same exocyclic partition without PS merging. It caps the cut sites with hydrogen and writes each fragment as a canonical SMILES core. Both forms then use the ring word `c1ccccc1`. Separate tokens identify the attachment sites. This separation follows the clean-fragment approach of fragSMILES @mastrolorito2025fragsmiles and retains TSDY's tree traversal @wu2024t.

=== Token stream and decoding
The encoder writes the cores in breadth-first order on a first-child/next-sibling binary tree. Each core string is one token. The decoder reads the tree to identify each fragment's parent. An internal empty slot requires a `<NULL>` token, while trailing empty slots can be omitted. A `<DOT>` starts a new disconnected component.


#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    align: center,
    [*Adjacent methyl groups*], [*Opposite methyl groups*],
    [`Cc1ccccc1C`], [`Cc1ccc(C)cc1`],
    image("../assets/xylene-adjacent.svg", width: 100%), image("../assets/xylene-opposite.svg", width: 100%),
    [`C c1ccccc1 <N2>` `<NULL> C <N3>`], [`C c1ccccc1 <N2>` `<NULL> C <N5>`],
  ),
  kind: image,
  caption: [Enhanced-TSDY token streams for two dimethylbenzene isomers. Ring labels give core atom indices. Traversal proceeds from left to right at each level. The left branch points to the first child and the right branch to the next sibling. The empty sibling slot ($emptyset$) emits `<NULL>`.],
) <enhanced-example>

For each child, the decoder restores a single bond between the parent and child cores and removes a hydrogen cap at each endpoint. If a core has only one hydrogen-bearing atom, the decoder can infer that endpoint. Otherwise, the encoder writes the atom's index in the parsed core after the child word (for example, `<N1>` denotes atom index 1, counting from zero). When both endpoints need indices, the parent's comes first.

In @enhanced-example, the first methyl group attaches to ring atom 2 in both molecules. The second methyl group attaches to atom 3 in the adjacent isomer and atom 5 in the opposite isomer. The methyl core has only one atom, so it needs no index.

#figure(
  table(
    columns: (1fr, 2.6fr),
    align: (left, left),
    inset: 5pt,
    table.header([*Token entry*], [*Meaning and examples*]),
    [Core SMILES], [One word per core. For ethanol, `CCO` becomes `C C <NULL> O`.],
    [`<NULL>`, `<DOT>`],
    [Empty tree slot (ethanol above) and component separator. The salt `[Na+].[Cl-]` becomes `[Cl-] <DOT> [Na+]`.],

    [`<N0>` to `<N63>`], [Small nonnegative integers for atom indices and stereo records.],
    [`<BIG>`, digits, `</BIG>`],
    [For indices of 64 or more, the markers enclose the digits of one integer, as in `<BIG> 6 4 </BIG>`. Reusing digit tokens keeps the vocabulary size fixed as indices grow.],

    [`<ACW>`, `<ACCW>`],
    [Tetrahedral stereo suffixes for lactic acid:
      `C[C@H](O)C(=O)O` ends in `<ACCW> <N1>`.
      `C[C@@H](O)C(=O)O` ends in `<ACW> <N1>`.],

    [`<B>`],
    [For `F/C=C/F`, the stereo suffix is
      `<B> <N1> <N0> <N1> <N1>`
      `<N3> <N0> <N0> <N2> <N0>`.],
  ),
  kind: table,
  caption: [Main token families in enhanced TSDY. Grammar entries are separate vocabulary words, alongside chemical-core words.],
) <enhanced-grammar>

=== Stereochemistry and chemistry scope
Molecules can have identical atoms and bonds but differ in how their atoms are arranged in space. Enhanced TSDY writes this information after the tree of each component.

*Tetrahedral centres.* In lactic acid, the central carbon bonds to four different groups (H, OH, methyl, and carboxyl) pointing towards the corners of a tetrahedron#footnote[See #link("https://www.rdkit.org/docs/RDKit_Book.html#tetrahedral-stereogenic-atoms")[The RDKit Book, “Tetrahedral Stereogenic atoms”].]. These groups can occupy two mirror-image arrangements. The SMILES strings `C[C@H](O)C(=O)O` and `C[C@@H](O)C(=O)O` distinguish them. Both produce the same tree tokens, `C C <NULL> O <NULL> <NULL> C=O O`. The left panel of @enhanced-stereo shows the first arrangement. The wedge marks the orientation recorded after the tree. The first stream appends `<ACCW> <N1>` and the second appends `<ACW> <N1>`. Here fragment 1 contains just the central carbon. The encoder orders its neighbours by fragment and atom index and records the orientation relative to that order. After reconnecting the bonds, the decoder uses the same ordering to recover the specified arrangement.

*Double-bond stereochemistry.* The `=` in `C=C` specifies a double bond (bond order two). The slash marks on its neighbouring bonds specify the relative arrangement of the attached groups#footnote[See #link("https://www.rdkit.org/docs/RDKit_Book.html#stereogenic-bonds")[The RDKit Book, “Stereogenic bonds”].]. In `F/C=C/F`, the fluorine atoms lie on opposite sides of the carbon-carbon double bond. In `F/C=C\F`, they lie on the same side, as drawn in the right panel of @enhanced-stereo. Both produce the tree tokens `F C=C <N0> <NULL> F <N1>`. The following record for the same-side form is grouped by field below. Each atom address lists its fragment index first and its atom index second.

#figure(
  table(
    columns: (0.65fr, 1fr, 1fr, 0.85fr, 1fr, 1fr),
    inset: 3pt,
    table.header([*Start*], [*Carbon 1*], [*Carbon 2*], [*Relation*], [*Fluorine 1*], [*Fluorine 2*]),
    [`<B>`], [`<N1> <N0>`], [`<N1> <N1>`], [`<N2>`], [`<N0> <N0>`], [`<N2> <N0>`],
  ),
  caption: [Double-bond stereo record for same-side difluoroethene.],
)
Both carbons belong to fragment 1, the `C=C` core. The final two addresses identify the neighbours whose relative positions define the configuration. The opposite-side form uses `<N3>` in the relation field.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 8pt,
    align: center,
    [*Lactic acid*], [*Same-side difluoroethene*],
    [`C[C@H](O)C(=O)O`], [`F/C=C\F`],
    image("../assets/lactate-tree.svg", width: 100%), image("../assets/cis-dfe-tree.svg", width: 100%),
    [`C C <NULL> O` \ `<NULL> <NULL> C=O O` \ `<ACCW> <N1>`],
    [`F C=C <N0> <NULL> F <N1>` \ `<B> <N1> <N0> <N1> <N1>` \ `<N2> <N0> <N0> <N2> <N0>`],
  ),
  kind: image,
  caption: [Enhanced-TSDY token streams for two molecules with stereochemistry.],
) <enhanced-stereo>

*Chemistry retained in cores.* The word `[Na+]` preserves sodium's positive charge. The word `[13CH4]` preserves carbon-13, an isotope of carbon with a different neutron count from carbon-12. The methyl radical `[CH3]` retains its unpaired electron. Bonds inside a core also retain their type, as illustrated by the double bond in the `C=C` core above.

*Scope.* Enhanced TSDY excludes dative bonds. RDKit extends SMILES with arrows for dative bonds#footnote[See #link("https://www.rdkit.org/docs/RDKit_Book.html#dative-bonds")[The RDKit Book, “Dative bonds”].]. A minimal example is `N->[Fe+2]`. Nitrogen donates both bonding electrons to the iron ion, and the arrow points from donor to acceptor. A separate limit concerns the shape around an atom. In the lactic acid example, four groups point towards the corners of a tetrahedron. Around a square-planar platinum centre, four neighbours instead occupy the corners of a flat square#footnote[See #link("https://www.rdkit.org/docs/RDKit_Book.html#square-planar")[The RDKit Book, “Square planar”].]. For example, `Cl[Pt@SP1](Cl)(N)N` and `Cl[Pt@SP2](Cl)(N)N` specify different arrangements of the same neighbours. Enhanced TSDY has no token to preserve that distinction, so it rejects both inputs. @unsupported-chemistry shows the donor arrow and the two arrangements around platinum.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 8pt,
    align: center,
    [*Dative bond*], [*Square-planar SP1*], [*Square-planar SP2*],
    [`N->[Fe+2]`], [`Cl[Pt@SP1](Cl)(N)N`], [`Cl[Pt@SP2](Cl)(N)N`],

    image("../assets/scope-iron.svg", width: 100%),
    image("../assets/scope-sp1.svg", width: 100%),
    image("../assets/scope-sp2.svg", width: 100%),
  ),
  kind: image,
  caption: [Dative bonding and square-planar stereochemistry outside the enhanced TSDY domain.],
) <unsupported-chemistry>
