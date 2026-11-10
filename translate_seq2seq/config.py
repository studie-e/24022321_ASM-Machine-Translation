from dataclasses import dataclass

from common.config import TrainConfig


@dataclass
class Seq2SeqConfig(TrainConfig):
    model: str = "seq2seq"
    emb_dim: int = 512
    hidden_dim: int = 512
    n_enc_layers: int = 1
    n_dec_layers: int = 2
    dropout: float = 0.2
    tie_embeddings: bool = True
    lr: float = 1e-3
    clip_norm: float = 5.0

    def validate(self):
        super().validate()
        if min(self.emb_dim, self.hidden_dim, self.n_enc_layers, self.n_dec_layers) < 1:
            raise ValueError("LSTM dimensions must be positive")
        if self.tie_embeddings and self.emb_dim != self.hidden_dim:
            raise ValueError("Weight tying requires emb_dim == hidden_dim")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
