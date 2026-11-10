from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    model: str = "transformer"
    data_dir: str = str(ROOT / "data/iwslt15_en_vi")
    tokenizer_dir: str = str(ROOT / "data/tokenizer")
    output_dir: str = ""
    device: str = "auto"
    seed: int = 42
    epochs: int = 30
    patience: int = 5
    max_len: int = 100
    max_tokens: int = 4096
    eval_max_tokens: int = 2048
    accum_steps: int = 1
    num_workers: int = 0
    amp: bool = False
    label_smoothing: float = 0.1
    clip_norm: float = 1.0
    max_new_tokens: int = 160
    length_penalty: float = 1.0
    src_vocab_size: int = 0
    tgt_vocab_size: int = 0
    smoke: bool = False

    def validate(self):
        positive = (
            "epochs",
            "max_len",
            "max_tokens",
            "eval_max_tokens",
            "accum_steps",
            "max_new_tokens",
        )
        for key in positive:
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.patience < 0 or self.num_workers < 0:
            raise ValueError("patience and num_workers must be nonnegative")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.clip_norm <= 0 or self.length_penalty < 0:
            raise ValueError("Invalid clipping or length penalty")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)


def config_from_dict(values):
    from translate_seq2seq.config import Seq2SeqConfig
    from translate_transformers.config import TransformerConfig

    classes = {"transformer": TransformerConfig, "seq2seq": Seq2SeqConfig}
    cls = classes[values["model"]]
    allowed = {f.name for f in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    config = cls(**values)
    config.validate()
    return config


def build_model(config):
    if min(config.src_vocab_size, config.tgt_vocab_size) <= 4:
        raise ValueError("Load tokenizer vocabulary sizes before building the model")
    if config.model == "transformer":
        from translate_transformers.model import TransformerNMT

        return TransformerNMT(config)
    if config.model == "seq2seq":
        from translate_seq2seq.model import Seq2SeqNMT

        return Seq2SeqNMT(config)
    raise ValueError(config.model)
