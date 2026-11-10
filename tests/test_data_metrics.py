import pytest

from common.data import TokenBatchSampler, TranslationDataset, collate_batch
from common.metrics import corpus_metrics
from common.tokenizer import Tokenizer, train_tokenizer


class ToyTokenizer:
    def encode_batch(self, lines):
        return [[4] * len(line.split()) for line in lines]


def test_pair_filtering_and_target_shift():
    tok = ToyTokenizer()
    src, tgt = ["a", "a b c d", ""], ["x y", "z", "w"]
    train = TranslationDataset(src, tgt, tok, tok, max_len=4, filter_long=True)
    assert len(train) == 1 and train.n_dropped == 2
    batch = collate_batch([train[0]])
    assert batch["src"].tolist() == [[4, 3]]
    assert batch["tgt_in"].tolist() == [[2, 4, 4]]
    assert batch["tgt_out"].tolist() == [[4, 4, 3]]
    test = TranslationDataset(src, tgt, tok, tok, max_len=4, filter_long=False)
    assert len(test) == 3


def test_token_budget_and_epoch_shuffle():
    lengths = [2, 3, 10, 12, 15, 40] * 10
    sampler = TokenBatchSampler(lengths, max_tokens=30)
    first = list(sampler)
    for batch in first:
        assert len(batch) == 1 or max(lengths[i] for i in batch) * len(batch) <= 30
    assert sorted(i for batch in first for i in batch) == list(range(len(lengths)))
    sampler.set_epoch(1)
    assert first != list(sampler)
    sampler.set_epoch(0)
    assert first == list(sampler)


def test_tokenizer_special_tokens_and_roundtrip(tmp_path):
    path = tmp_path / "train.txt"
    path.write_text("Tôi thích học máy .\nBạn thích đọc sách .\n" * 20, encoding="utf-8")
    tok = train_tokenizer(path, tmp_path / "spm", vocab_size=80)
    text = "Tôi thích học máy ."
    ids = tok.encode(text)
    assert tok.decode([2, *ids, 3, *tok.encode("Bạn")]) == text
    assert Tokenizer(tmp_path / "spm.model").vocab_size == tok.vocab_size


def test_corpus_bleu_signature_and_case():
    reference = ["Tôi thích học máy .", "Bạn thích đọc sách ."]
    perfect = corpus_metrics(reference, reference)
    assert perfect["bleu"] == pytest.approx(100)
    assert "tok:none" in perfect["bleu_signature"]
    assert "case:mixed" in perfect["bleu_signature"]
    assert corpus_metrics([s.lower() for s in reference], reference)["bleu"] < 100
    with pytest.raises(ValueError):
        corpus_metrics(["a"], ["a", "b"])
