import math

import torch
from torch import nn
from torch.nn import functional as F

from common.tokenizer import PAD_ID


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, initial_length=256):
        super().__init__()
        if d_model % 2:
            raise ValueError("d_model must be even")
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "pe", self._table(initial_length, torch.device("cpu")), persistent=False
        )

    def _table(self, length, device):
        positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=device).float()
            * (-math.log(10000.0) / self.d_model)
        )
        angles = positions * frequencies
        table = torch.empty(length, self.d_model, device=device)
        table[:, 0::2], table[:, 1::2] = angles.sin(), angles.cos()
        return table[None]

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            self.pe = self._table(x.size(1), x.device)
        return self.dropout(x * math.sqrt(self.d_model) + self.pe[:, : x.size(1)].to(x.dtype))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.heads, self.head_dim = heads, d_model // heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, allowed):
        def split(x):
            return x.reshape(x.size(0), x.size(1), self.heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q(query)), split(self.k(key)), split(self.v(value))
        result = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed[:, None],
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        result = result.transpose(1, 2).contiguous().flatten(2)
        return self.out(result)


class FeedForward(nn.Sequential):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attention = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.attn_dropout)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, allowed):
        normalized = self.norm1(x)
        x = x + self.dropout(self.attention(normalized, normalized, normalized, allowed))
        return x + self.dropout(self.ff(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(cfg.d_model) for _ in range(3)]
        self.self_attention = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.attn_dropout)
        self.cross_attention = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.attn_dropout)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, target_allowed, source_allowed):
        normalized = self.norm1(x)
        x = x + self.dropout(
            self.self_attention(normalized, normalized, normalized, target_allowed)
        )
        x = x + self.dropout(self.cross_attention(self.norm2(x), memory, memory, source_allowed))
        return x + self.dropout(self.ff(self.norm3(x)))


class TransformerNMT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.src_embedding = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.tgt_embedding = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=PAD_ID)
        self.positions = PositionalEncoding(cfg.d_model, cfg.dropout)
        self.encoder = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_enc_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_dec_layers)])
        self.encoder_norm, self.decoder_norm = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.output = nn.Linear(cfg.d_model, cfg.tgt_vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output.weight = self.tgt_embedding.weight
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        for embedding in (self.src_embedding, self.tgt_embedding):
            nn.init.normal_(embedding.weight, std=cfg.d_model**-0.5)
            with torch.no_grad():
                embedding.weight[PAD_ID].zero_()

    def init_decode(self, src, src_len):
        allowed = src.ne(PAD_ID)[:, None, :]
        x = self.positions(self.src_embedding(src))
        for layer in self.encoder:
            x = layer(x, allowed)
        return {"memory": self.encoder_norm(x), "allowed": allowed}

    def _decode_hidden(self, tokens, state):
        length = tokens.size(1)
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).tril()
        allowed = tokens.ne(PAD_ID)[:, None, :] & causal[None]
        x = self.positions(self.tgt_embedding(tokens))
        for layer in self.decoder:
            x = layer(x, state["memory"], allowed, state["allowed"])
        return self.decoder_norm(x)

    def _decode(self, tokens, state):
        return self.output(self._decode_hidden(tokens, state))

    def forward(self, src, src_len, tgt_in):
        return self._decode(tgt_in, self.init_decode(src, src_len))

    def decode_step(self, tokens, state):
        return self.output(self._decode_hidden(tokens, state)[:, -1]), state
