import argparse
import json
from pathlib import Path

import torch

from common.config import ROOT
from common.data import SPLITS, TranslationDataset, load_split
from common.decoding import beam_search
from common.engine import load_run, translate_dataset
from common.tokenizer import EOS_ID
from common.utils import dump_json, get_device, sha256


def evaluate_run(
    run_dir,
    data_dir=None,
    split="test",
    beams=(1, 5),
    device_name="auto",
    max_tokens=None,
    max_new_tokens=None,
    length_penalty=None,
    show=8,
    output_dir=None,
):
    device = get_device(device_name)
    model, cfg, tok_src, tok_tgt, checkpoint = load_run(run_dir, device)
    if cfg.smoke:
        raise ValueError(
            "Smoke checkpoints are diagnostic only. Train a non-smoke run for evaluation."
        )
    data_dir = Path(data_dir or cfg.data_dir)
    src, tgt = load_split(data_dir, split)
    dataset = TranslationDataset(src, tgt, tok_src, tok_tgt, filter_long=False)
    output = Path(output_dir) if output_dir else Path(run_dir) / "evaluation" / split
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_sha256": sha256(Path(run_dir) / "best.pt"),
        "checkpoint_epoch": checkpoint["epoch"],
        "model": cfg.model,
        "split": split,
        "source_sha256": sha256(data_dir / f"{SPLITS[split]}.en"),
        "reference_sha256": sha256(data_dir / f"{SPLITS[split]}.vi"),
        "training_fingerprint": checkpoint["fingerprint"],
        "src_vocab_size": tok_src.vocab_size,
        "tgt_vocab_size": tok_tgt.vocab_size,
        "results": {},
    }
    for beam in sorted(set(beams)):
        scores, hypotheses = translate_dataset(
            model,
            dataset,
            tok_tgt,
            device,
            max_tokens or cfg.eval_max_tokens,
            beam,
            max_new_tokens or cfg.max_new_tokens,
            cfg.length_penalty if length_penalty is None else length_penalty,
        )
        mode = "greedy" if beam == 1 else f"beam{beam}"
        manifest["results"][mode] = scores
        (output / f"{SPLITS[split]}.hyp.{mode}.vi").write_text(
            "\n".join(hypotheses) + "\n", encoding="utf-8"
        )
        examples = [
            {"index": i, "source": src[i], "reference": tgt[i], "prediction": hypotheses[i]}
            for i in range(min(show, len(src)))
        ]
        dump_json(output / f"examples.{mode}.json", examples)
        print(f"{mode}: BLEU={scores['bleu']:.2f}, chrF2={scores['chrf2']:.2f}", flush=True)
        for example in examples:
            print(json.dumps(example, ensure_ascii=False))
    (output / f"{SPLITS[split]}.ref.vi").write_text("\n".join(tgt) + "\n", encoding="utf-8")
    dump_json(output / "metrics.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "seq2seq"], default="transformer")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--beam", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--length-penalty", type=float)
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--text")
    args = parser.parse_args()
    if any(b < 1 for b in args.beam) or args.show < 0:
        parser.error("beam must be positive and show must be nonnegative")
    for key in ("max_tokens", "max_new_tokens"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.length_penalty is not None and args.length_penalty < 0:
        parser.error("length_penalty must be nonnegative")
    run_dir = args.run_dir or ROOT / "runs" / args.model
    if args.text is not None:
        device = get_device(args.device)
        model, cfg, src_tok, tgt_tok, _ = load_run(run_dir, device)
        src = torch.tensor([src_tok.encode(args.text) + [EOS_ID]], device=device)
        lengths = torch.tensor([src.size(1)])
        for beam in args.beam:
            result = beam_search(
                model,
                src,
                lengths,
                beam,
                args.max_new_tokens or cfg.max_new_tokens,
                cfg.length_penalty if args.length_penalty is None else args.length_penalty,
            )
            print(f"beam={beam}: {tgt_tok.decode(result[0].tolist())}")
        return
    evaluate_run(
        run_dir,
        args.data_dir,
        args.split,
        args.beam,
        args.device,
        args.max_tokens,
        args.max_new_tokens,
        args.length_penalty,
        args.show,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
