import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.config import ROOT
from common.data import SPLITS, TranslationDataset, load_split
from common.tokenizer import load_tokenizers, train_tokenizer
from common.utils import dump_json, sha256


def prepare(data_dir, tokenizer_dir, vocab_size=8000, max_len=100, strict=True):
    data_dir, tokenizer_dir = Path(data_dir), Path(tokenizer_dir)
    splits = {name: load_split(data_dir, name, strict) for name in SPLITS}
    settings = {
        "vocab_size_requested": vocab_size,
        "normalization": "identity",
        "training_files": {lang: sha256(data_dir / f"train.{lang}") for lang in ("en", "vi")},
    }
    manifest_path = tokenizer_dir / "manifest.json"
    existing = list(tokenizer_dir.glob("spm_*.*")) if tokenizer_dir.exists() else []
    if existing:
        if not manifest_path.exists():
            raise FileExistsError(
                "Tokenizer files exist without manifest. Use a new tokenizer directory."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["settings"] != settings:
            raise ValueError("Tokenizer training inputs changed. Use a new tokenizer directory.")
        for lang in ("en", "vi"):
            if sha256(tokenizer_dir / f"spm_{lang}.model") != manifest["model_hashes"][lang]:
                raise ValueError("Tokenizer file checksum mismatch")
    else:
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        for lang in ("en", "vi"):
            train_tokenizer(data_dir / f"train.{lang}", tokenizer_dir / f"spm_{lang}", vocab_size)
        dump_json(
            manifest_path,
            {
                "settings": settings,
                "model_hashes": {
                    lang: sha256(tokenizer_dir / f"spm_{lang}.model") for lang in ("en", "vi")
                },
            },
        )
    tok_src, tok_tgt = load_tokenizers(tokenizer_dir)
    stats = {
        "max_len_train_including_special_tokens": max_len,
        "vocabulary_sizes": {"en": tok_src.vocab_size, "vi": tok_tgt.vocab_size},
        "splits": {},
        "overlap": {},
    }
    pairs = {}
    for name, (src, tgt) in splits.items():
        ds = TranslationDataset(src, tgt, tok_src, tok_tgt, max_len, filter_long=name == "train")
        record = {
            "raw_pairs": len(src),
            "used_pairs": len(ds),
            "dropped": ds.n_dropped,
            "empty_source": sum(not s for s in src),
            "empty_target": sum(not t for t in tgt),
        }
        for lang, lines, tok in (("en", src, tok_src), ("vi", tgt, tok_tgt)):
            lengths = np.array([len(ids) for ids in tok.encode_batch(lines)])
            record[lang] = {
                "mean_subwords": float(lengths.mean()),
                "p95_subwords": float(np.percentile(lengths, 95)),
                "max_subwords": int(lengths.max()),
                "sha256": sha256(data_dir / f"{SPLITS[name]}.{lang}"),
            }
        pairs[name] = set(zip(src, tgt))
        record["duplicate_pairs_within_split"] = len(src) - len(pairs[name])
        stats["splits"][name] = record
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        stats["overlap"][f"{left}_{right}_exact_pairs"] = len(pairs[left] & pairs[right])
        stats["overlap"][f"{left}_{right}_source_sentences"] = len(
            set(splits[left][0]) & set(splits[right][0])
        )
    dump_json(tokenizer_dir / "data_stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/iwslt15_en_vi")
    parser.add_argument("--tokenizer-dir", type=Path, default=ROOT / "data/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-len", type=int, default=100)
    args = parser.parse_args()
    prepare(args.data_dir, args.tokenizer_dir, args.vocab_size, args.max_len)


if __name__ == "__main__":
    main()
