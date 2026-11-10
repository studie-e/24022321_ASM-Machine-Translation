import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.config import ROOT, build_model
from common.data import TranslationDataset, build_loader, load_split
from common.engine import evaluate_loss, translate_dataset
from common.loss import loss_sums
from common.tokenizer import load_tokenizers
from common.utils import dump_json, get_device, set_seed
from translate_seq2seq.config import Seq2SeqConfig
from translate_transformers.config import TransformerConfig


def sanity_check(name, data_dir, tokenizer_dir, device, samples=64, steps=300, output=None):
    set_seed(42)
    tok_src, tok_tgt = load_tokenizers(tokenizer_dir)
    src, tgt = load_split(data_dir, "train")
    chosen = []
    for s, t in zip(src, tgt):
        if s and t and max(len(tok_src.encode(s)), len(tok_tgt.encode(t))) <= 15:
            chosen.append((s, t))
            if len(chosen) == samples:
                break
    if len(chosen) < samples:
        raise ValueError("Not enough short sentence pairs")
    src, tgt = map(list, zip(*chosen))
    dataset = TranslationDataset(src, tgt, tok_src, tok_tgt)
    cfg = (
        TransformerConfig(
            d_model=64,
            n_heads=4,
            n_enc_layers=2,
            n_dec_layers=2,
            d_ff=128,
            dropout=0,
            attn_dropout=0,
        )
        if name == "transformer"
        else Seq2SeqConfig(emb_dim=64, hidden_dim=64, dropout=0)
    )
    cfg.src_vocab_size, cfg.tgt_vocab_size = tok_src.vocab_size, tok_tgt.vocab_size
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loader = build_loader(dataset, max_tokens=2048)
    initial = evaluate_loss(model, loader, device)
    history = []
    iterator = iter(loader)
    for step in range(1, steps + 1):
        model.train()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        src_tensor, tgt_in, tgt_out = [
            batch[key].to(device) for key in ("src", "tgt_in", "tgt_out")
        ]
        optimizer.zero_grad(set_to_none=True)
        logits = model(src_tensor, batch["src_len"], tgt_in)
        loss, _, count = loss_sums(logits, tgt_out)
        loss = loss / count
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite sanity loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "loss": float(loss.detach())})
        if step % 50 == 0:
            print(f"{name} step={step} loss={history[-1]['loss']:.4f}", flush=True)
    final = evaluate_loss(model, loader, device)
    scores, hypotheses = translate_dataset(model, dataset, tok_tgt, device, max_new_tokens=32)
    report = {
        "model": name,
        "samples": samples,
        "steps": steps,
        "initial_nll": initial,
        "final_nll": final,
        "loss_ratio": final / initial,
        "metrics": scores,
        "examples": [
            {"source": s, "reference": t, "prediction": h}
            for s, t, h in zip(src[:8], tgt[:8], hypotheses[:8])
        ],
        "history": history,
    }
    if output:
        dump_json(Path(output) / f"sanity_{name}.json", report)
    print(f"{name}: NLL {initial:.4f} -> {final:.4f}; BLEU={scores['bleu']:.2f}", flush=True)
    for item in report["examples"][:3]:
        print(item)
    if final > initial * 0.5:
        raise RuntimeError("NLL did not fall by 50%; inspect the sanity run before full training")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["both", "transformer", "seq2seq"], default="both")
    parser.add_argument("--data-dir", default=str(ROOT / "data/iwslt15_en_vi"))
    parser.add_argument("--tokenizer-dir", default=str(ROOT / "data/tokenizer"))
    parser.add_argument("--output-dir", default=str(ROOT / "results/sanity"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    if args.samples < 1 or args.steps < 1:
        parser.error("samples and steps must be positive")
    names = ["transformer", "seq2seq"] if args.model == "both" else [args.model]
    for name in names:
        sanity_check(
            name,
            args.data_dir,
            args.tokenizer_dir,
            get_device(args.device),
            args.samples,
            args.steps,
            args.output_dir,
        )
