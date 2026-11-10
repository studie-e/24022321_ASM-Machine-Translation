import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.tokenizer import PAD_ID


class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, query, keys, memory, mask):
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1).float() * self.scale
        weights = scores.masked_fill(mask, -torch.inf).softmax(-1).to(memory.dtype)
        return torch.bmm(weights[:, None], memory).squeeze(1)


class Seq2SeqNMT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        e, h = cfg.emb_dim, cfg.hidden_dim
        self.src_embedding = nn.Embedding(cfg.src_vocab_size, e, padding_idx=PAD_ID)
        self.tgt_embedding = nn.Embedding(cfg.tgt_vocab_size, e, padding_idx=PAD_ID)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.encoder = nn.LSTM(
            e,
            h,
            cfg.n_enc_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.n_enc_layers > 1 else 0,
        )
        self.decoder = nn.LSTM(
            e + h,
            h,
            cfg.n_dec_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.n_dec_layers > 1 else 0,
        )
        self.memory_projection = nn.Linear(2 * h, h, bias=False)
        self.bridge_h, self.bridge_c = nn.Linear(2 * h, h), nn.Linear(2 * h, h)
        self.attention = LuongAttention(h)
        self.combine = nn.Linear(2 * h, h, bias=False)
        self.output_dropout = nn.Dropout(cfg.dropout)
        self.output = nn.Linear(h, cfg.tgt_vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output.weight = self.tgt_embedding.weight
        self._reset_parameters()

    def _reset_parameters(self):
        for name, parameter in self.named_parameters():
            if "weight_hh" in name:
                for gate in parameter.chunk(4, dim=0):
                    nn.init.orthogonal_(gate)
            elif parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
                if "bias_ih" in name:
                    with torch.no_grad():
                        parameter.chunk(4)[1].fill_(1)
        for embedding in (self.src_embedding, self.tgt_embedding):
            nn.init.normal_(embedding.weight, std=self.cfg.emb_dim**-0.5)
            with torch.no_grad():
                embedding.weight[PAD_ID].zero_()

    def init_decode(self, src, src_len):
        embedded = self.embedding_dropout(self.src_embedding(src))
        with torch.autocast(device_type=src.device.type, enabled=False):
            packed = pack_padded_sequence(
                embedded.float(), src_len.cpu(), batch_first=True, enforce_sorted=False
            )
            packed, (h, c) = self.encoder(packed)
            memory, _ = pad_packed_sequence(packed, batch_first=True, total_length=src.size(1))
            h = torch.tanh(self.bridge_h(torch.cat([h[-2], h[-1]], dim=-1)))
            c = torch.tanh(self.bridge_c(torch.cat([c[-2], c[-1]], dim=-1)))
        memory = self.memory_projection(memory)
        return {
            "memory": memory,
            "keys": self.attention.key_projection(memory),
            "mask": src.eq(PAD_ID),
            "h": h[:, None].repeat(1, self.cfg.n_dec_layers, 1),
            "c": c[:, None].repeat(1, self.cfg.n_dec_layers, 1),
            "feed": memory.new_zeros(src.size(0), self.cfg.hidden_dim),
        }

    def _step(self, embedded, state):
        x = torch.cat([embedded, state["feed"]], dim=-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            h, c = [state[key].transpose(0, 1).contiguous().float() for key in ("h", "c")]
            output, (h, c) = self.decoder(x[:, None].float(), (h, c))
        query = output[:, 0].to(state["memory"].dtype)
        context = self.attention(query, state["keys"], state["memory"], state["mask"])
        feed = torch.tanh(self.combine(torch.cat([context, query], dim=-1)))
        new_state = {**state, "h": h.transpose(0, 1), "c": c.transpose(0, 1), "feed": feed}
        return feed, new_state

    def forward(self, src, src_len, tgt_in):
        state = self.init_decode(src, src_len)
        embeddings = self.embedding_dropout(self.tgt_embedding(tgt_in))
        outputs = []
        for index in range(tgt_in.size(1)):
            output, state = self._step(embeddings[:, index], state)
            outputs.append(output)
        return self.output(self.output_dropout(torch.stack(outputs, dim=1)))

    def decode_step(self, tokens, state):
        output, state = self._step(self.tgt_embedding(tokens[:, -1]), state)
        return self.output(self.output_dropout(output)), state
