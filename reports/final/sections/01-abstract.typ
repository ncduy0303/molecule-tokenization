#heading(numbering: none)[Abstract]

A chemical language model reads a molecule as a sequence of tokens. Its tokenizer determines whether a token represents a character, an atom, or a larger molecular fragment. This project compares tokenization methods at these different scales through language model pretraining and molecular property prediction. In the initial benchmark, a glyph-based tokenizer achieved the highest mean test score on both the HIV activity and blood-brain barrier penetration tasks.

The fragment experiments examine two choices: where to cut a molecule and how to record the connections between its fragments. Merging frequently adjacent fragments shortened tokenized sequences, but increased the proportion of unknown tokens and produced mixed prediction results. With the fragment boundaries held fixed, storing attachment sites separately allowed different attachment patterns of the same core to share one vocabulary entry. On both tasks, the mean test score of this enhanced TSDY representation was higher than that of the original representation with attachment sites embedded in the fragment words. These findings show why sequence length and vocabulary coverage should be evaluated alongside predictive performance.

#v(0.5cm)
*Subject Descriptors:*
#block(inset: (left: 2em))[10010147.10010257 Computing methodologies $->$ Machine learning \
  10010405.10010444.10010450 Applied computing $->$ Life and medical sciences $->$ Bioinformatics]

#v(0.5cm)
*Keywords:*
#block(inset: (
  left: 2em,
))[Molecular tokenization, molecular representations, chemical language models, molecular property prediction.]

#v(0.5cm)
*Implementation Software and Hardware:*
#block(inset: (
  left: 2em,
))[Python, RDKit, PyTorch, Hugging Face Transformers, NUS School of Computing cluster (AMD EPYC 7763 CPU and NVIDIA H100 GPU).]
