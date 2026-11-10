import csv
import importlib.metadata
import itertools
import math
import platform
import shutil
import time
from pathlib import Path

import torch

from common.config import build_model, config_from_dict
from common.data import TranslationDataset, build_loader, load_split
from common.decoding import beam_search
from common.loss import loss_sums
from common.metrics import corpus_metrics
from common.tokenizer import EOS_ID, PAD_ID, load_tokenizers
from common.utils import (
    atomic_save,
    dump_json,
    get_device,
    restore_rng,
    rng_state,
    set_seed,
    sha256,
    synchronize,
)


class InverseSqrtSchedule:
    def __init__(self, optimizer, d_model, warmup_steps, scale):
        self.optimizer = optimizer
        self.d_model, self.warmup_steps, self.scale = d_model, warmup_steps, scale
        self.updates = 0

    def prepare_step(self):
        step = self.updates + 1
        lr = self.scale * self.d_model**-0.5 * min(step**-0.5, step * self.warmup_steps**-1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def state_dict(self):
        return {"updates": self.updates}

    def load_state_dict(self, state):
        self.updates = state["updates"]


def make_optimizer(model, cfg):
    if cfg.model == "transformer":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-7, betas=(0.9, 0.98), eps=1e-9, weight_decay=cfg.weight_decay
        )
        scheduler = InverseSqrtSchedule(optimizer, cfg.d_model, cfg.warmup_steps, cfg.lr_scale)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = None
    return optimizer, scheduler


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True) if key not in ("src_len", "index") else value
        for key, value in batch.items()
    }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg):
    model.train()
    total_nll, total_loss, total_tokens, updates, skipped = 0.0, 0.0, 0, 0, 0
    synchronize(device)
    started = time.perf_counter()
    iterator = iter(loader)
    planned = math.ceil(len(loader) / cfg.accum_steps)
    while group := list(itertools.islice(iterator, cfg.accum_steps)):
        denominator = sum(int(batch["tgt_out"].ne(PAD_ID).sum()) for batch in group)
        if denominator == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        for raw_batch in group:
            batch = move_batch(raw_batch, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()
            ):
                logits = model(batch["src"], batch["src_len"], batch["tgt_in"])
                loss, nll, count = loss_sums(logits, batch["tgt_out"], cfg.label_smoothing)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss / denominator).backward()
            total_loss += float(loss.detach())
            total_nll += float(nll.detach())
            total_tokens += int(count)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.clip_norm, error_if_nonfinite=not scaler.is_enabled()
        )
        if scheduler:
            scheduler.prepare_step()
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            skipped += 1
        else:
            updates += 1
            if scheduler:
                scheduler.updates += 1
        completed = updates + skipped
        if completed % 100 == 0 or completed == planned:
            elapsed = time.perf_counter() - started
            remaining = elapsed / completed * max(0, planned - completed)
            print(f"  train {completed}/{planned} | NLL={total_nll / total_tokens:.4f} | "
                  f"elapsed={elapsed:.0f}s | train ETA={remaining:.0f}s | "
                  f"skipped={skipped}", flush=True)
    synchronize(device)
    seconds = time.perf_counter() - started
    if total_tokens == 0 or updates == 0:
        raise RuntimeError("No successful optimizer update in this epoch")
    return {
        "train_loss": total_loss / total_tokens,
        "train_nll": total_nll / total_tokens,
        "train_seconds": seconds,
        "train_tokens": total_tokens,
        "target_tokens_per_second": total_tokens / seconds,
        "optimizer_updates": updates,
        "skipped_updates": skipped,
    }


@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = model(batch["src"], batch["src_len"], batch["tgt_in"])
        _, nll, count = loss_sums(logits, batch["tgt_out"])
        total_nll += float(nll)
        total_tokens += int(count)
    if not total_tokens or not math.isfinite(total_nll):
        raise FloatingPointError("Invalid validation NLL")
    return total_nll / total_tokens


@torch.no_grad()
def translate_dataset(
    model,
    dataset,
    tokenizer,
    device,
    max_tokens=2048,
    beam_size=1,
    max_new_tokens=160,
    length_penalty=1.0,
):
    model.eval()
    hypotheses = [None] * len(dataset)
    truncated = 0
    for raw_batch in build_loader(dataset, max_tokens=max_tokens):
        batch = move_batch(raw_batch, device)
        output = (
            beam_search(
                model, batch["src"], batch["src_len"], beam_size, max_new_tokens, length_penalty
            )
            .cpu()
            .tolist()
        )
        for index, tokens in zip(raw_batch["index"].tolist(), output):
            if hypotheses[index] is not None:
                raise RuntimeError("Duplicate prediction index")
            hypotheses[index] = tokenizer.decode(tokens)
            truncated += int(EOS_ID not in tokens and len(tokens) >= max_new_tokens)
    if any(hyp is None for hyp in hypotheses):
        raise RuntimeError("Missing predictions")
    scores = corpus_metrics(hypotheses, dataset.raw_tgt)
    scores.update(
        {
            "beam_size": beam_size,
            "max_new_tokens": max_new_tokens,
            "length_penalty": length_penalty,
            "truncated_sentences": truncated,
            "inference_precision": "float32",
        }
    )
    return scores, hypotheses


def training_fingerprint(cfg):
    data = Path(cfg.data_dir)
    tokenizer = Path(cfg.tokenizer_dir)
    files = {
        f"data/{stem}.{lang}": data / f"{stem}.{lang}"
        for stem in ("train", "tst2012")
        for lang in ("en", "vi")
    }
    files.update(
        {f"tokenizer/spm_{lang}.model": tokenizer / f"spm_{lang}.model" for lang in ("en", "vi")}
    )
    return {name: sha256(path) for name, path in files.items()}


def write_history(directory, history):
    if not history:
        return
    path = Path(directory) / "history.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in history for key in row)))
        writer.writeheader()
        writer.writerows(history)


def environment_info(device):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "sentencepiece", "sacrebleu")
        },
    }


def run_training(cfg, resume=False):
    cfg.validate()
    device = get_device(cfg.device)
    if cfg.amp and device.type != "cuda":
        raise ValueError("AMP is only enabled for CUDA; remove --amp for CPU")
    output = Path(cfg.output_dir)
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {output}. Use another directory or --resume."
        )
    if resume and not (output / "last.pt").exists():
        raise FileNotFoundError(output / "last.pt")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    tok_src, tok_tgt = load_tokenizers(cfg.tokenizer_dir)
    cfg.src_vocab_size, cfg.tgt_vocab_size = tok_src.vocab_size, tok_tgt.vocab_size
    identity = training_fingerprint(cfg)
    datasets = {}
    for split in ("train", "dev"):
        src, tgt = load_split(cfg.data_dir, split)
        if cfg.smoke:
            limit = 128 if split == "train" else 16
            src, tgt = src[:limit], tgt[:limit]
        datasets[split] = TranslationDataset(
            src, tgt, tok_src, tok_tgt, cfg.max_len, filter_long=split == "train"
        )
    model = build_model(cfg).to(device)
    optimizer, scheduler = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    best_bleu, best_epoch, stale, start_epoch, global_updates = -1.0, 0, 0, 1, 0
    history = []
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        allowed = {"device", "data_dir", "tokenizer_dir", "output_dir", "epochs", "num_workers", "amp", "eval_max_tokens"}
        changes = [
            key
            for key, value in cfg.to_dict().items()
            if key not in allowed and checkpoint["config"].get(key) != value
        ]
        if changes or identity != checkpoint["fingerprint"]:
            raise ValueError(
                f"Resume requires matching config and data/tokenizer. Changed: {changes}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler.is_enabled() and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        old_amp = checkpoint["config"].get("amp", False)
        if old_amp != cfg.amp:
            print(f"Precision change at resume: amp={old_amp} -> {cfg.amp}; "
                  "weights, optimizer and scheduler restored.", flush=True)
        best_bleu, best_epoch = checkpoint["best_bleu"], checkpoint["best_epoch"]
        stale, start_epoch = checkpoint["stale"], checkpoint["epoch"] + 1
        history, global_updates = checkpoint["history"], checkpoint["global_updates"]
        for row in history:
            row.setdefault("amp", old_amp)
        restore_rng(checkpoint["rng"])
        del checkpoint
        write_history(output, history)
        if not (output / "best.pt").exists():
            raise FileNotFoundError("best.pt is missing from the resumed run")
    else:
        tokenizer_copy = output / "tokenizer"
        tokenizer_copy.mkdir()
        for path in Path(cfg.tokenizer_dir).glob("spm_*.*"):
            shutil.copy2(path, tokenizer_copy / path.name)
    dump_json(output / "config.json", cfg.to_dict())
    dump_json(output / "fingerprint.json", identity)
    dump_json(output / "environment.json", environment_info(device))
    dump_json(
        output / "data_stats.json",
        {split: {"used": len(ds), "dropped": ds.n_dropped} for split, ds in datasets.items()},
    )
    train_loader = build_loader(datasets["train"], cfg.max_tokens, True, cfg.seed, cfg.num_workers)
    dev_loader = build_loader(
        datasets["dev"], cfg.eval_max_tokens, False, cfg.seed, cfg.num_workers
    )
    if cfg.model == "transformer":
        planned = math.ceil(len(train_loader) / cfg.accum_steps) * cfg.epochs
        if planned <= cfg.warmup_steps:
            print(
                f"Warning: {planned} planned updates <= {cfg.warmup_steps} warmup updates.",
                flush=True,
            )
    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.patience and stale >= cfg.patience:
            break
        print(f"\n>>> Epoch {epoch}/{cfg.epochs} | amp={cfg.amp}", flush=True)
        train_loader.batch_sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        stats = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, cfg)
        global_updates += stats["optimizer_updates"]
        print(f"Train done: {stats['train_seconds']:.1f}s. Computing dev NLL...", flush=True)
        dev_nll = evaluate_loss(model, dev_loader, device)
        print("Computing dev BLEU (greedy, FP32)...", flush=True)
        dev_scores, _ = translate_dataset(
            model,
            datasets["dev"],
            tok_tgt,
            device,
            cfg.eval_max_tokens,
            1,
            cfg.max_new_tokens,
            cfg.length_penalty,
        )
        synchronize(device)
        row = {
            "epoch": epoch,
            "amp": cfg.amp,
            **stats,
            "dev_nll": dev_nll,
            "dev_bleu": dev_scores["bleu"],
            "dev_truncated": dev_scores["truncated_sentences"],
            "lr": optimizer.param_groups[0]["lr"],
            "global_updates": global_updates,
            "epoch_seconds_including_dev": time.perf_counter() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        }
        history.append(row)
        improved = dev_scores["bleu"] > best_bleu
        if improved:
            best_bleu, best_epoch, stale = dev_scores["bleu"], epoch, 0
            atomic_save(
                {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "epoch": epoch,
                    "dev_bleu": best_bleu,
                    "fingerprint": identity,
                },
                output / "best.pt",
            )
        else:
            stale += 1
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict(),
                "config": cfg.to_dict(),
                "epoch": epoch,
                "best_bleu": best_bleu,
                "best_epoch": best_epoch,
                "stale": stale,
                "history": history,
                "rng": rng_state(),
                "fingerprint": identity,
                "global_updates": global_updates,
            },
            output / "last.pt",
        )
        write_history(output, history)
        dump_json(output / "dev_metrics.json", dev_scores)
        print(
            f"epoch={epoch} train_nll={stats['train_nll']:.4f} dev_nll={dev_nll:.4f} "
            f"dev_bleu={dev_scores['bleu']:.2f} best_epoch={best_epoch}",
            flush=True,
        )
    summary = {
        "model": cfg.model,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_dev_bleu": best_bleu,
        "epochs_completed": len(history),
        "train_seconds": sum(row["train_seconds"] for row in history),
        "total_seconds_including_dev": sum(row["epoch_seconds_including_dev"] for row in history),
        "smoke": cfg.smoke,
        "training_accessed_test": False,
    }
    dump_json(output / "summary.json", summary)
    return summary


def load_run(run_dir, device):
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg = config_from_dict(checkpoint["config"])
    tok_src, tok_tgt = load_tokenizers(run_dir / "tokenizer")
    for lang in ("en", "vi"):
        if (
            sha256(run_dir / "tokenizer" / f"spm_{lang}.model")
            != checkpoint["fingerprint"][f"tokenizer/spm_{lang}.model"]
        ):
            raise ValueError("Tokenizer does not match the checkpoint")
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg, tok_src, tok_tgt, checkpoint
