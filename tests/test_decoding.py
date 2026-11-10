import torch
from torch import nn

from common.decoding import beam_search, greedy_decode


class ScriptedModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def init_decode(self, src, src_len):
        return {"marker": src[:, :1].clone(), "history": src.new_zeros(src.size(0), 1)}

    def decode_step(self, tokens, state):
        if tokens.size(1) > 1:
            expected = tokens[:, :-1].sum(1, keepdim=True)
            assert torch.equal(state["history"], expected), "Beam state was not reordered"
        state = {**state, "history": state["history"] + tokens[:, -1:]}
        logits = torch.full((tokens.size(0), 7), -100.0, device=tokens.device)
        logits[:, 0], logits[:, 2] = 1000, 1000
        if self.mode == "eos":
            logits[:, 3] = 5
        elif self.mode == "never_eos":
            logits[:, 4] = 5
            logits[:, 3] = -torch.inf
        else:
            logits[:, 4], logits[:, 5] = 2, 1.9
            if tokens.size(1) > 1:
                logits[tokens[:, -1].eq(5), 3] = 8
                logits[tokens[:, -1].eq(4), 6] = 2.5
        return logits, state


def test_immediate_eos_and_forbidden_tokens():
    src, lengths = torch.tensor([[4, 3]]), torch.tensor([2])
    model = ScriptedModel("eos")
    assert greedy_decode(model, src, lengths, 5).tolist() == [[3]]
    assert beam_search(model, src, lengths, 2, 5).tolist() == [[3]]


def test_no_eos_keeps_last_token():
    src, lengths = torch.tensor([[4, 3]]), torch.tensor([2])
    model = ScriptedModel("never_eos")
    assert beam_search(model, src, lengths, 2, 5).shape == (1, 5)
    assert greedy_decode(model, src, lengths, 5).tolist() == [[4] * 5]


def test_beam_reorders_state_and_can_outperform_greedy():
    src, lengths = torch.tensor([[4, 3]]), torch.tensor([2])
    model = ScriptedModel("branches")
    assert greedy_decode(model, src, lengths, 3)[0, 0] == 4
    result = beam_search(model, src, lengths, 2, 3, length_penalty=0)
    assert result.tolist() == [[5, 3]]
