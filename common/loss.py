import torch.nn.functional as F

from common.tokenizer import PAD_ID


def loss_sums(logits, targets, smoothing=0.0):
    logits = logits.float().reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    count = targets.ne(PAD_ID).sum()
    nll = F.cross_entropy(logits, targets, ignore_index=PAD_ID, reduction="sum")
    loss = (
        F.cross_entropy(
            logits, targets, ignore_index=PAD_ID, reduction="sum", label_smoothing=smoothing
        )
        if smoothing
        else nll
    )
    return loss, nll, count
