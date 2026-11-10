import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.bench import measure_latency
from common.data import load_split
from common.engine import load_run
from common.tokenizer import EOS_ID
from common.utils import dump_json, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    if min(args.beam, args.samples, args.repeats, args.max_new_tokens) < 1:
        parser.error("All numeric parameters must be positive")
    device = get_device(args.device)
    model, cfg, tokenizer, _, _ = load_run(args.run_dir, device)
    sources, _ = load_split(args.data_dir or cfg.data_dir, "test")
    measurements = []
    for index, source in enumerate(sources[: args.samples]):
        src = torch.tensor([tokenizer.encode(source) + [EOS_ID]], device=device)
        lengths = torch.tensor([src.size(1)])
        record = measure_latency(
            model,
            src,
            lengths,
            args.beam,
            args.max_new_tokens,
            cfg.length_penalty,
            repeats=args.repeats,
        )
        measurements.append({"test_index": index, **record})
    report = {
        "measurements": measurements,
        "mean_milliseconds_per_sentence": sum(m["milliseconds_per_sentence"] for m in measurements)
        / len(measurements),
        "note": "Batch size 1, first test sentences, excludes tokenization and file IO",
    }
    dump_json(args.run_dir / f"latency.beam{args.beam}.json", report)
    print(report["mean_milliseconds_per_sentence"])


if __name__ == "__main__":
    main()
