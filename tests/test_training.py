from copy import deepcopy

import torch

from common.data import collate_batch
from common.engine import InverseSqrtSchedule, train_one_epoch
from common.utils import restore_rng, rng_state


def test_gradient_accumulation_matches_one_batch(model):
    model.cfg.label_smoothing = 0.1
    model.cfg.accum_steps = 2
    model.cfg.clip_norm = 1e6
    samples = [([4, 5, 3], [2, 7, 8, 3], 0), ([6, 3], [2, 9, 3], 1), ([5, 3], [2, 10, 3], 2)]
    first, second = deepcopy(model), deepcopy(model)
    optimizer_a = torch.optim.SGD(first.parameters(), lr=0.1)
    optimizer_b = torch.optim.SGD(second.parameters(), lr=0.1)
    device = torch.device("cpu")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    train_one_epoch(
        first,
        [collate_batch(samples[:2]), collate_batch(samples[2:])],
        optimizer_a,
        None,
        scaler,
        device,
        model.cfg,
    )
    train_one_epoch(second, [collate_batch(samples)], optimizer_b, None, scaler, device, model.cfg)
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_accumulation_flushes_partial_group(model):
    cfg = model.cfg
    cfg.accum_steps = 2
    sample = collate_batch([([4, 3], [2, 5, 3], 0)])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    result = train_one_epoch(
        model,
        [sample] * 3,
        optimizer,
        None,
        torch.amp.GradScaler("cuda", enabled=False),
        torch.device("cpu"),
        cfg,
    )
    assert result["optimizer_updates"] == 2
    assert result["train_tokens"] == 6


def test_scheduler_resume_and_rng():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter])
    schedule = InverseSqrtSchedule(optimizer, 16, 4, 2.0)
    schedule.prepare_step()
    expected = 2 * 16**-0.5 * 4**-1.5
    assert optimizer.param_groups[0]["lr"] == expected
    schedule.updates = 7
    state = schedule.state_dict()
    resumed = InverseSqrtSchedule(optimizer, 16, 4, 2.0)
    resumed.load_state_dict(state)
    resumed.prepare_step()
    assert optimizer.param_groups[0]["lr"] == 2 * 16**-0.5 * 8**-0.5
    rng = rng_state()
    expected = torch.rand(4)
    restore_rng(rng)
    torch.testing.assert_close(torch.rand(4), expected)
