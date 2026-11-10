import copy
import math

import torch

from common.engine import write_history
from translate_transformers.model import MultiHeadAttention


def test_sdpa_matches_reference_outputs_and_gradients():
    torch.manual_seed(31)
    optimized = MultiHeadAttention(16, 2, 0).eval()
    reference = copy.deepcopy(optimized)
    x = torch.randn(2, 5, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    allowed = torch.ones(2, 5, 5, dtype=torch.bool).tril()
    allowed[1, :, 4] = False
    actual = optimized(x, x, x, allowed)
    def split(t):
        return t.reshape(2, 5, 2, 8).transpose(1, 2)
    q, k, v = (split(layer(y)) for layer in (reference.q, reference.k, reference.v))
    weights = ((q @ k.transpose(-2, -1)) / math.sqrt(8)).masked_fill(
        ~allowed[:, None], -torch.inf
    ).softmax(-1)
    expected = reference.out((weights @ v).transpose(1, 2).contiguous().flatten(2))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=1e-5)
    for a, b in zip(optimized.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=1e-5)


def test_decode_projects_only_last_token(model, batch):
    if model.cfg.model != 'transformer':
        return
    model.eval()
    shapes = []
    hook = model.output.register_forward_pre_hook(lambda module, args: shapes.append(args[0].shape))
    state = model.init_decode(batch['src'], batch['src_len'])
    model.decode_step(batch['tgt_in'], state)
    hook.remove()
    assert shapes == [torch.Size([2, model.cfg.d_model])]


def test_history_accepts_legacy_rows(tmp_path):
    write_history(tmp_path, [{'epoch': 1}, {'epoch': 2, 'amp': True}])
    assert (tmp_path / 'history.csv').read_text().splitlines() == ['epoch,amp', '1,', '2,True']
