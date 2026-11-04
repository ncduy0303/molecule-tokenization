= Conclusions

== Summary of findings

The initial benchmark identified SMIRK as a strong glyph-based baseline. Among the t-SMILES tokenizers, changing the fragmentation rule from BRICS to MMPA improved prediction across all tested notations, while the best assembly notation depended on the task.

The controlled experiments showed that shortening a sequence and improving vocabulary coverage are distinct objectives. Principal-subgraph merging shortened sequences, but more fragments fell outside the vocabulary and prediction did not improve consistently. Enhanced TSDY instead used a common core word for fragments with different attachment patterns. This improved fragment coverage and prediction over base TSDY on both tasks, even when the attachment records made sequences longer.

== Limitations

*Task scope.* The downstream evaluation covered HIV and BBBP classification with one scaffold split per dataset. It leaves open how the tokenizers perform on other molecular properties or in molecule generation.

*Attachment encoding.* Enhanced TSDY changed the core words, connection syntax, and stereo records together. Their individual contributions to prediction remain uncertain.

*Chemical coverage.* Enhanced TSDY supports a limited set of bond types and stereochemical arrangements. Rare cores can also fall outside the learned vocabulary.

== Recommendations for future work

*Isolate core sharing.* Compare two tokenizers with identical fragment boundaries, traversal, and connection records. At the same vocabulary size, give one tokenizer a shared word per core and the other distinct words for different attachment patterns.

*Vary the merge count.* Test whether fewer PS merges can shorten sequences while keeping more fragments in the vocabulary.

*Broaden property prediction.* Evaluate additional datasets, including a regression task such as predicting molecular solubility. This would test whether the findings extend beyond binary classification.

*Test molecule generation.* Train models to generate token sequences and compare how often those sequences decode into valid molecules. This would test whether the connection syntax is practical for producing new structures.
