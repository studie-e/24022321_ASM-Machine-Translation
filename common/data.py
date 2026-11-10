import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from common.tokenizer import BOS_ID, EOS_ID, PAD_ID

SPLITS = {"train": "train", "dev": "tst2012", "test": "tst2013"}
EXPECTED_COUNTS = {"train": 133317, "dev": 1553, "test": 1268}


def load_split(directory, split, strict=True):
    stem = SPLITS[split]
    paths = [Path(directory) / f"{stem}.{lang}" for lang in ("en", "vi")]
    src, tgt = [p.read_text(encoding="utf-8").splitlines() for p in paths]
    src, tgt = [s.strip() for s in src], [s.strip() for s in tgt]
    if len(src) != len(tgt) or not src:
        raise ValueError(f"Invalid aligned corpus: {split}, {len(src)} vs {len(tgt)}")
    if strict and len(src) != EXPECTED_COUNTS[split]:
        raise ValueError(f"Expected {EXPECTED_COUNTS[split]} pairs in {split}, got {len(src)}")
    return src, tgt


class TranslationDataset(Dataset):
    def __init__(self, src, tgt, tok_src, tok_tgt, max_len=100, filter_long=False):
        if len(src) != len(tgt):
            raise ValueError("Source and target sizes differ")
        self.samples, self.raw_src, self.raw_tgt = [], [], []
        self.original_indices = []
        self.n_dropped = 0
        encoded_src, encoded_tgt = tok_src.encode_batch(src), tok_tgt.encode_batch(tgt)
        for original, (s, t) in enumerate(zip(encoded_src, encoded_tgt)):
            s, t = s + [EOS_ID], [BOS_ID] + t + [EOS_ID]
            empty = not src[original].strip() or not tgt[original].strip()
            if filter_long and (empty or max(len(s), len(t)) > max_len):
                self.n_dropped += 1
                continue
            self.samples.append((s, t))
            self.raw_src.append(src[original])
            self.raw_tgt.append(tgt[original])
            self.original_indices.append(original)
        if not self.samples:
            raise ValueError("Dataset is empty after preprocessing")
        self.lengths = [max(len(s), len(t)) for s, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return *self.samples[index], index


def collate_batch(batch):
    src, tgt, indices = zip(*batch)
    src_len = torch.tensor([len(s) for s in src], dtype=torch.long)
    src = pad_sequence([torch.tensor(s) for s in src], batch_first=True, padding_value=PAD_ID)
    tgt = pad_sequence([torch.tensor(t) for t in tgt], batch_first=True, padding_value=PAD_ID)
    return {
        "src": src,
        "src_len": src_len,
        "tgt_in": tgt[:, :-1].contiguous(),
        "tgt_out": tgt[:, 1:].contiguous(),
        "index": torch.tensor(indices),
    }


class TokenBatchSampler(Sampler):
    def __init__(self, lengths, max_tokens=4096, shuffle=True, seed=42):
        if not lengths or max_tokens < 1 or min(lengths) < 1:
            raise ValueError("Invalid lengths or token budget")
        self.lengths = lengths
        self.max_tokens, self.shuffle, self.seed = max_tokens, shuffle, seed
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
            indices = [
                i
                for start in range(0, len(indices), 1024)
                for i in sorted(indices[start : start + 1024], key=self.lengths.__getitem__)
            ]
        else:
            indices.sort(key=self.lengths.__getitem__)
        self.batches = []
        batch, longest = [], 0
        for index in indices:
            new_longest = max(longest, self.lengths[index])
            if batch and new_longest * (len(batch) + 1) > self.max_tokens:
                self.batches.append(batch)
                batch, longest = [], 0
            batch.append(index)
            longest = max(longest, self.lengths[index])
        if batch:
            self.batches.append(batch)
        if self.shuffle:
            rng.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def build_loader(dataset, max_tokens=4096, shuffle=False, seed=42, num_workers=0):
    sampler = TokenBatchSampler(dataset.lengths, max_tokens, shuffle, seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        persistent_workers=num_workers > 0,
    )
