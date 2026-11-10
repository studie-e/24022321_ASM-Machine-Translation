import math

import pytest
import torch
import torch.nn.functional as F

from common.decoding import beam_search, greedy_decode
from common.loss import loss_sums
from translate_transformers.model import PositionalEncoding


def test_forward_backward(model, batch):
    logits = model(batch["src"], batch["src_len"], batch["tgt_in"])
    assert logits.shape == (2, 3, 12)
    loss, nll, count = loss_sums(logits, batch["tgt_out"], 0.1)
    assert count == 5 and torch.isfinite(nll)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0


def test_no_future_information(model, batch):
    model.eval()
    target = torch.tensor([[2, 7, 8], [2, 9, 10]])
    reference = model(batch["src"], batch["src_len"], target)
    target[:, 2] = 11
    changed = model(batch["src"], batch["src_len"], target)
    torch.testing.assert_close(reference[:, :2], changed[:, :2], atol=1e-6, rtol=1e-5)


def test_padding_invariance(model, batch):
    model.eval()
    reference = model(batch["src"], batch["src_len"], batch["tgt_in"])
    changed = model(F.pad(batch["src"], (0, 4)), batch["src_len"], F.pad(batch["tgt_in"], (0, 3)))
    torch.testing.assert_close(reference, changed[:, :3], atol=2e-6, rtol=1e-5)


def test_autoregressive_step_matches_teacher_forcing(model, batch):
    model.eval()
    target = torch.tensor([[2, 7, 8], [2, 9, 10]])
    reference = model(batch["src"], batch["src_len"], target)
    state = model.init_decode(batch["src"], batch["src_len"])
    for step in range(target.size(1)):
        logits, state = model.decode_step(target[:, : step + 1], state)
        torch.testing.assert_close(logits, reference[:, step], atol=2e-6, rtol=1e-5)


def test_beam_one_matches_greedy(model, batch):
    model.eval()
    greedy = greedy_decode(model, batch["src"], batch["src_len"], 6)
    beam = beam_search(model, batch["src"], batch["src_len"], 1, 6)
    torch.testing.assert_close(greedy, beam)
    assert not greedy.eq(2).any()


def test_beam_batched_matches_single(model, batch):
    model.eval()
    together = beam_search(model, batch["src"], batch["src_len"], 3, 5)
    for i in range(2):
        alone = beam_search(model, batch["src"][i : i + 1], batch["src_len"][i : i + 1], 3, 5)[0]
        valid = alone.ne(0).sum()
        torch.testing.assert_close(together[i, :valid], alone[:valid])


def test_checkpoint_round_trip(model, batch, tmp_path):
    model.eval()
    expected = greedy_decode(model, batch["src"], batch["src_len"], 5)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1)
    model.load_state_dict(torch.load(path, weights_only=True))
    actual = greedy_decode(model, batch["src"], batch["src_len"], 5)
    torch.testing.assert_close(expected, actual)


def test_positional_encoding_formula_and_expansion():
    pe = PositionalEncoding(8, dropout=0, initial_length=2)
    result = pe(torch.zeros(1, 301, 8))
    assert result.shape == (1, 301, 8)
    for pos in (0, 1, 13, 300):
        for pair in range(4):
            angle = pos / 10000 ** (2 * pair / 8)
            assert float(result[0, pos, 2 * pair]) == pytest.approx(math.sin(angle), abs=2e-6)
            assert float(result[0, pos, 2 * pair + 1]) == pytest.approx(math.cos(angle), abs=2e-6)


def test_loss_ignores_padding_and_smoothing_separate():
    torch.manual_seed(5)
    logits, targets = torch.randn(2, 3, 12), torch.tensor([[4, 5, 3], [6, 3, 0]])
    loss, nll, count = loss_sums(logits, targets, 0.1)
    padded = loss_sums(F.pad(logits, (0, 0, 0, 4)), F.pad(targets, (0, 4)), 0.1)
    for a, b in zip((loss, nll, count), padded):
        torch.testing.assert_close(a, b)
    expected = F.cross_entropy(
        logits.reshape(-1, 12), targets.flatten(), ignore_index=0, reduction="sum"
    )
    torch.testing.assert_close(nll, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp(model, batch):
    model.cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(batch["src"].cuda(), batch["src_len"], batch["tgt_in"].cuda())
        loss = loss_sums(logits, batch["tgt_out"].cuda(), 0.1)[0]
    scaler = torch.amp.GradScaler("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    assert torch.isfinite(loss)


def test_single_sentence_latency(model, batch):
    from common.bench import measure_latency

    result = measure_latency(
        model, batch["src"][:1], batch["src_len"][:1], max_new_tokens=3, warmup=0, repeats=1
    )
    assert result["milliseconds_per_sentence"] > 0
    assert result["source_tokens"] == 3
