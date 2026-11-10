import pytest
import torch

from common.config import build_model
from translate_seq2seq.config import Seq2SeqConfig
from translate_transformers.config import TransformerConfig


def pytest_sessionstart(session):
    torch.set_num_threads(2)


@pytest.fixture(params=["transformer", "seq2seq"])
def model(request):
    torch.manual_seed(7)
    if request.param == "transformer":
        cfg = TransformerConfig(
            d_model=16,
            n_heads=2,
            n_enc_layers=1,
            n_dec_layers=1,
            d_ff=32,
            dropout=0,
            attn_dropout=0,
        )
    else:
        cfg = Seq2SeqConfig(emb_dim=16, hidden_dim=16, dropout=0)
    cfg.src_vocab_size = cfg.tgt_vocab_size = 12
    return build_model(cfg)


@pytest.fixture
def batch():
    return {
        "src": torch.tensor([[4, 5, 3], [6, 3, 0]]),
        "src_len": torch.tensor([3, 2]),
        "tgt_in": torch.tensor([[2, 7, 8], [2, 9, 0]]),
        "tgt_out": torch.tensor([[7, 8, 3], [9, 3, 0]]),
    }
