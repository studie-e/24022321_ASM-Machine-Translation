import json

import pytest
import torch

from common.data import EXPECTED_COUNTS
from common.engine import run_training
from common.utils import sha256
from compare import compare
from evaluate import evaluate_run
from scripts.prepare import prepare
from translate_seq2seq.config import Seq2SeqConfig
from translate_transformers.config import TransformerConfig


@pytest.fixture
def tiny_corpus(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    src = ["I love this book .", "You like this book .", "I read a book .", "You read this book ."]
    tgt = [
        "Tôi yêu cuốn sách này .",
        "Bạn thích cuốn sách này .",
        "Tôi đọc một cuốn sách .",
        "Bạn đọc cuốn sách này .",
    ]
    for name, stem in (("train", "train"), ("dev", "tst2012"), ("test", "tst2013")):
        monkeypatch.setitem(EXPECTED_COUNTS, name, 4)
        for lang, lines in (("en", src), ("vi", tgt)):
            (data / f"{stem}.{lang}").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    prepare(data, tokenizer, vocab_size=64, strict=True)
    return data, tokenizer


def config(name, root, data, tokenizer, epochs):
    if name == "transformer":
        cfg = TransformerConfig(
            d_model=8,
            n_heads=2,
            n_enc_layers=1,
            n_dec_layers=1,
            d_ff=16,
            dropout=0.1,
            attn_dropout=0.1,
            warmup_steps=2,
        )
    else:
        cfg = Seq2SeqConfig(emb_dim=8, hidden_dim=8, dropout=0.1)
    cfg.data_dir, cfg.tokenizer_dir, cfg.output_dir = str(data), str(tokenizer), str(root)
    cfg.device, cfg.epochs, cfg.patience = "cpu", epochs, 0
    cfg.max_tokens = cfg.eval_max_tokens = 64
    cfg.max_new_tokens = 6
    return cfg


@pytest.mark.parametrize("name", ["transformer", "seq2seq"])
def test_resume_matches_uninterrupted(name, tiny_corpus, tmp_path):
    data, tokenizer = tiny_corpus
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    run_training(config(name, uninterrupted, data, tokenizer, 2))
    run_training(config(name, resumed, data, tokenizer, 1))
    run_training(config(name, resumed, data, tokenizer, 2), resume=True)
    left = torch.load(uninterrupted / "last.pt", weights_only=False)
    right = torch.load(resumed / "last.pt", weights_only=False)
    assert left["global_updates"] == right["global_updates"]
    for key, value in left["model"].items():
        torch.testing.assert_close(value, right["model"][key], atol=0, rtol=0)
    assert not (resumed / "evaluation").exists()
    with pytest.raises(FileExistsError):
        run_training(config(name, resumed, data, tokenizer, 1))
    changed = config(name, resumed, data, tokenizer, 3)
    changed.max_len = 90
    with pytest.raises(ValueError):
        run_training(changed, resume=True)


def test_end_to_end_evaluate_and_compare(tiny_corpus, tmp_path):
    data, tokenizer = tiny_corpus
    runs = [tmp_path / name for name in ("transformer", "seq2seq")]
    for name, run in zip(("transformer", "seq2seq"), runs):
        run_training(config(name, run, data, tokenizer, 1))
        result = evaluate_run(run, data, device_name="cpu", max_new_tokens=6, show=1)
        assert result["results"]["greedy"]["sentences"] == 4
        assert result["checkpoint_sha256"] == sha256(run / "best.pt")
        for mode in ("greedy", "beam5"):
            path = run / "evaluation/test" / f"tst2013.hyp.{mode}.vi"
            assert len(path.read_text(encoding="utf-8").splitlines()) == 4
    rows = compare(runs, tmp_path / "comparison", plots=True)
    assert len(rows) == 4
    assert (tmp_path / "comparison/training_curves.png").exists()
    manifest_path = runs[1] / "evaluation/test/metrics.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["reference_sha256"] = "wrong"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        compare(runs, tmp_path / "bad_comparison")


def test_saved_tokenizer_cannot_be_silently_replaced(tiny_corpus, tmp_path):
    data, tokenizer = tiny_corpus
    first = sha256(tokenizer / "spm_en.model")
    prepare(data, tokenizer, vocab_size=64, strict=True)
    assert sha256(tokenizer / "spm_en.model") == first
    with pytest.raises(ValueError):
        prepare(data, tokenizer, vocab_size=80, strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_resume_fp32_checkpoint_with_amp(tiny_corpus, tmp_path):
    data, tokenizer = tiny_corpus
    run = tmp_path / 'precision_resume'
    cfg = config('transformer', run, data, tokenizer, 1)
    cfg.device = 'cuda'
    run_training(cfg)
    before = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    cfg.epochs, cfg.amp = 2, True
    run_training(cfg, resume=True)
    after = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    assert after['epoch'] == 2
    assert after['global_updates'] > before['global_updates']
    assert after['scaler']
    assert [row['amp'] for row in after['history']] == [False, True]
