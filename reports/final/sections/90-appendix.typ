#heading(numbering: none)[Appendix B: Reproducibility details]

*Computing environment.* Tokenizer construction and principal-subgraph mining used AMD EPYC 7763 CPU nodes. Training jobs used NVIDIA H100 96GB GPU nodes. The training environment used PyTorch 2.10.0, CUDA 12.8, RDKit 2025.9.4 and Hugging Face Transformers 4.57.6.

#let address(url) = link(url, text(url))

*Code.* The repository contains the source code for the project. \
#address("https://github.com/ncduy0303/molecule-tokenization")

*Data.* The tokenizer and pretraining corpora use this PubChem10M dataset. \
#address(
  "https://huggingface.co/datasets/mikemayuare/PubChem10M_SMILES_SELFIES",
)

For downstream finetuning, the loader downloads the MoleculeNet HIV and BBBP datasets from DeepChem as CSV files.

- HIV: #address("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv")
- BBBP: #address("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv")

*Results.* The Weights & Biases project records training curves and test scores. \
#address("https://wandb.ai/ncduy0303/molecule-tokenization")
