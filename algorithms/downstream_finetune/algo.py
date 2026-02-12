"""
Algorithm module for downstream classification finetuning.

Loads a pretrained RoBERTa model and replaces the MLM head with a
classification head (RobertaForSequenceClassification).
"""

from omegaconf import DictConfig


class MoleculeClassificationAlgo:
    """
    Builds a tokenizer + RoBERTa sequence classifier from a pretrained
    MLM checkpoint (e.g. mikemayuare/SMILYBPE).

    Config keys:
      - tokenizer.type       (auto | smirk)
      - tokenizer.pretrained (HF hub ID)
      - model.pretrained     (HF hub ID for the pretrained RoBERTa)
      - model.num_labels     (number of classification targets)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.tokenizer = self._build_tokenizer()
        self.model = None  # built later once num_labels is known

    def _build_tokenizer(self):
        tok_cfg = self.cfg.tokenizer
        tok_type = tok_cfg.type

        if tok_type == "smirk":
            from smirk import SmirkTokenizerFast

            tokenizer = SmirkTokenizerFast()
        elif tok_type == "auto":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                tok_cfg.pretrained, trust_remote_code=True
            )
        else:
            raise ValueError(
                f"Unknown tokenizer type: '{tok_type}'. "
                "Supported: 'smirk', 'auto'"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

        return tokenizer

    def build_model(self, num_labels: int):
        """
        Build RobertaForSequenceClassification from a pretrained MLM model.
        Must be called after dataset is loaded (to know num_labels).
        """
        from transformers import RobertaForSequenceClassification

        pretrained = self.cfg.model.pretrained
        self.model = RobertaForSequenceClassification.from_pretrained(
            pretrained,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        # Align pad token ID with tokenizer
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        return self.model
