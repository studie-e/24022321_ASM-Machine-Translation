import math

import torch
from torch.nn.utils.rnn import pad_sequence

from common.tokenizer import BOS_ID, EOS_ID, PAD_ID


def select_state(state, indices):
    return {key: value.index_select(0, indices) for key, value in state.items()}


def token_log_probs(logits):
    logits = logits.float().clone()
    logits[:, [PAD_ID, BOS_ID]] = -torch.inf
    return logits.log_softmax(-1)


@torch.no_grad()
def greedy_decode(model, src, src_len, max_new_tokens=160):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    state = model.init_decode(src, src_len)
    tokens = src.new_full((src.size(0), 1), BOS_ID)
    finished = src.new_zeros(src.size(0), dtype=torch.bool)
    for _ in range(max_new_tokens):
        logits, state = model.decode_step(tokens, state)
        next_token = token_log_probs(logits).argmax(-1)
        next_token = next_token.masked_fill(finished, PAD_ID)
        tokens = torch.cat([tokens, next_token[:, None]], dim=1)
        finished |= next_token.eq(EOS_ID)
        if bool(finished.all()):
            break
    return tokens[:, 1:]


def _beam_one(model, initial_state, device, beam_size, max_new_tokens, alpha):
    state = initial_state
    tokens = torch.full((1, 1), BOS_ID, device=device, dtype=torch.long)
    scores = torch.zeros(1, device=device)
    completed = []
    for length in range(1, max_new_tokens + 1):
        logits, advanced = model.decode_step(tokens, state)
        log_probs = token_log_probs(logits)
        vocab_size = log_probs.size(-1)
        candidates = scores[:, None] + log_probs
        values, indices = candidates.flatten().topk(min(2 * beam_size, candidates.numel()))
        parents, next_tokens, next_scores = [], [], []
        for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist())):
            if not math.isfinite(value):
                continue
            parent, token = divmod(index, vocab_size)
            if token == EOS_ID:
                if rank < beam_size:
                    sequence = torch.cat([tokens[parent, 1:], tokens.new_tensor([EOS_ID])])
                    completed.append((value / ((5 + length) / 6) ** alpha, sequence))
            elif len(parents) < beam_size:
                parents.append(parent)
                next_tokens.append(token)
                next_scores.append(value)
        completed = sorted(completed, key=lambda item: item[0], reverse=True)[:beam_size]
        if not parents:
            break
        parent_index = tokens.new_tensor(parents)
        state = select_state(advanced, parent_index)
        tokens = torch.cat([tokens[parent_index], tokens.new_tensor(next_tokens)[:, None]], dim=1)
        scores = torch.tensor(next_scores, device=device)
        if len(completed) == beam_size:
            optimistic = float(scores.max()) / ((5 + max_new_tokens) / 6) ** alpha
            if completed[-1][0] >= optimistic:
                break
    if completed:
        return completed[0][1]
    return tokens[int(scores.argmax()), 1:]


@torch.no_grad()
def beam_search(model, src, src_len, beam_size=5, max_new_tokens=160, length_penalty=1.0):
    if beam_size < 1 or max_new_tokens < 1 or length_penalty < 0:
        raise ValueError("Invalid decoding parameters")
    if beam_size == 1:
        return greedy_decode(model, src, src_len, max_new_tokens)
    initial = model.init_decode(src, src_len)
    sequences = []
    for index in range(src.size(0)):
        state = select_state(initial, src.new_tensor([index]))
        sequences.append(
            _beam_one(model, state, src.device, beam_size, max_new_tokens, length_penalty)
        )
    return pad_sequence(sequences, batch_first=True, padding_value=PAD_ID)
