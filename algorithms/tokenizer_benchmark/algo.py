"""
Algorithm module for molecule tokenizer benchmarking.

This is not a PyTorch Lightning module. Instead, it wraps a HuggingFace
tokenizer and a RoBERTa model for masked language modeling (MLM).
The HF Trainer handles the training loop.
"""

from omegaconf import DictConfig


class MoleculeMLMAlgo:
    """
    Builds a tokenizer and a from-scratch RoBERTa model for MLM.

    Used by TokenizerBenchmarkExperiment. Each algorithm yaml
    (e.g. smirk_roberta, atomwise_roberta) specifies:
      - tokenizer.type  (smirk | auto)
      - model.*         (hidden_size, num_layers, etc.)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tokenizer = self._build_tokenizer()
        self.model = self._build_model()

    def _build_tokenizer(self):
        tok_cfg = self.cfg.tokenizer
        tok_type = tok_cfg.type

        if tok_type == "smirk":
            from smirk import SmirkTokenizerFast

            tokenizer = SmirkTokenizerFast(tok_cfg.get("pretrained", None))
        elif tok_type == "auto":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tok_cfg.pretrained, trust_remote_code=True
            )
        elif tok_type == "ape":
            from apetokenizer.ape_tokenizer import APETokenizer

            tokenizer = APETokenizer()
            tokenizer.load_vocabulary(tok_cfg.pretrained)
        elif tok_type == "pcatt":
            from pcatt.hf.greedtok import GreedTok

            tokenizer = GreedTok.from_pretrained(tok_cfg.pretrained)
        else:
            raise ValueError(
                f"Unknown tokenizer type: '{tok_type}'. "
                "Supported: 'smirk', 'auto', 'ape'."
            )

        # Ensure pad token exists (needed by DataCollatorForLanguageModeling)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

        return tokenizer

    def _build_model(self):
        from transformers import RobertaForMaskedLM

        # If a pretrained HF model hub ID is specified, load weights from it
        pretrained_model = self.cfg.model.get("pretrained", None)
        if pretrained_model:
            return RobertaForMaskedLM.from_pretrained(pretrained_model)

        # Otherwise, train from scratch with the specified architecture
        from transformers import RobertaConfig

        m = self.cfg.model
        config = RobertaConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=m.hidden_size,
            intermediate_size=m.intermediate_size,
            num_hidden_layers=m.num_hidden_layers,
            num_attention_heads=m.num_attention_heads,
            max_position_embeddings=m.max_position_embeddings,
        )
        return RobertaForMaskedLM(config)
