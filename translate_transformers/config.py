from dataclasses import dataclass

from common.config import TrainConfig


@dataclass
class TransformerConfig(TrainConfig):
    model: str = "transformer"
    d_model: int = 512
    n_heads: int = 4
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.3
    attn_dropout: float = 0.1
    tie_embeddings: bool = True
    lr_scale: float = 2.0
    warmup_steps: int = 4000
    weight_decay: float = 1e-4

    def validate(self):
        super().validate()
        if (
            min(
                self.d_model,
                self.n_heads,
                self.d_ff,
                self.n_enc_layers,
                self.n_dec_layers,
                self.warmup_steps,
            )
            < 1
        ):
            raise ValueError("Transformer dimensions and warmup must be positive")
        if self.d_model % self.n_heads or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by n_heads")
        if not 0 <= self.attn_dropout < 1 or self.lr_scale <= 0:
            raise ValueError("Invalid attention dropout or lr_scale")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative")
