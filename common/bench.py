import time

import torch

from common.decoding import beam_search
from common.utils import synchronize


@torch.no_grad()
def measure_latency(
    model, src, src_len, beam_size=1, max_new_tokens=160, length_penalty=1.0, warmup=2, repeats=5
):
    if repeats < 1 or warmup < 0 or src.size(0) != 1:
        raise ValueError("Latency benchmark requires one source sentence and positive repeats")
    model.eval()
    device = src.device
    for _ in range(warmup):
        beam_search(model, src, src_len, beam_size, max_new_tokens, length_penalty)
    synchronize(device)
    started = time.perf_counter()
    output = None
    for _ in range(repeats):
        output = beam_search(model, src, src_len, beam_size, max_new_tokens, length_penalty)
    synchronize(device)
    return {
        "milliseconds_per_sentence": (time.perf_counter() - started) * 1000 / repeats,
        "source_tokens": int(src_len[0]),
        "output_tensor_length": output.size(1),
        "beam_size": beam_size,
        "max_new_tokens": max_new_tokens,
        "warmup": warmup,
        "repeats": repeats,
        "device": str(device),
        "precision": "float32",
    }
