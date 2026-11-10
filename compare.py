import argparse
import csv
import json
from pathlib import Path

from common.config import ROOT
from common.utils import dump_json, sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(run_dirs, output, plots=False):
    if len(run_dirs) < 2:
        raise ValueError("At least two runs are required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = [
        (Path(path), read_json(Path(path) / "evaluation/test/metrics.json")) for path in run_dirs
    ]
    reference = records[0][1]
    reference_config = read_json(records[0][0] / "config.json")
    reference_data = read_json(records[0][0] / "data_stats.json")
    modes = set(reference["results"])
    rows = []
    for directory, metrics in records:
        if sha256(directory / "best.pt") != metrics["checkpoint_sha256"]:
            raise ValueError("Evaluation is stale: best.pt changed after evaluation")
        config = read_json(directory / "config.json")
        if config["max_len"] != reference_config["max_len"]:
            raise ValueError("Training length filters differ")
        if read_json(directory / "data_stats.json") != reference_data:
            raise ValueError("Effective training/dev datasets differ")
        for key in ("split", "source_sha256", "reference_sha256", "training_fingerprint"):
            if metrics[key] != reference[key]:
                raise ValueError(f"Incomparable runs: {key} differs")
        if set(metrics["results"]) != modes:
            raise ValueError("Runs must be evaluated with the same decoding modes")
        summary = read_json(directory / "summary.json")
        environment = read_json(directory / "environment.json")
        for mode in sorted(modes):
            scores = metrics["results"][mode]
            for key in (
                "bleu_signature",
                "chrf_signature",
                "inference_precision",
                "sentences",
                "beam_size",
                "max_new_tokens",
                "length_penalty",
            ):
                if scores[key] != reference["results"][mode][key]:
                    raise ValueError(f"Incomparable {mode}: {key} differs")
            rows.append(
                {
                    "run": directory.name,
                    "model": metrics["model"],
                    "decoding": mode,
                    "bleu": scores["bleu"],
                    "chrf2": scores["chrf2"],
                    "parameters": summary["parameters"],
                    "best_epoch": metrics["checkpoint_epoch"],
                    "train_seconds": summary["train_seconds"],
                    "training_device": environment["gpu"] or environment["device"],
                    "training_amp": config["amp"],
                    "truncated_sentences": scores["truncated_sentences"],
                }
            )
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump_json(output / "comparison.json", rows)
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [f"{row['run']}\n{row['decoding']}" for row in rows]
        ax.bar(labels, [row["bleu"] for row in rows])
        ax.set(ylabel="BLEU (0–100)", title="IWSLT English → Vietnamese")
        fig.tight_layout()
        fig.savefig(output / "test_bleu.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for directory, _ in records:
            with (directory / "history.csv").open(encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            epochs = [int(row["epoch"]) for row in history]
            axes[0].plot(
                epochs,
                [float(r["train_nll"]) for r in history],
                "--",
                label=f"{directory.name} train",
            )
            axes[0].plot(
                epochs, [float(r["dev_nll"]) for r in history], label=f"{directory.name} dev"
            )
            axes[1].plot(epochs, [float(r["dev_bleu"]) for r in history], label=directory.name)
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend()
        axes[0].set_ylabel("NLL per target token")
        axes[1].set_ylabel("Dev greedy BLEU")
        fig.tight_layout()
        fig.savefig(output / "training_curves.png", dpi=160)
        plt.close(fig)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", nargs="+", type=Path, default=[ROOT / "runs/transformer", ROOT / "runs/seq2seq"]
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/comparison")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("At least two runs are required")
    compare(args.runs, args.output_dir, args.plots)
