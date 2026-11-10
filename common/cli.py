import argparse
from dataclasses import fields
from pathlib import Path

from common.config import ROOT
from common.engine import run_training


def train_main(config_class):
    defaults = config_class()
    parser = argparse.ArgumentParser()
    for field in fields(defaults):
        if field.name in {"model", "src_vocab_size", "tgt_vocab_size"}:
            continue
        value = getattr(defaults, field.name)
        option = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=value)
        else:
            parser.add_argument(option, type=type(value), default=value)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--small", action="store_true")
    args = vars(parser.parse_args())
    resume, small = args.pop("resume"), args.pop("small")
    config = config_class(**args)
    if small or config.smoke:
        dim = 32 if config.smoke else 256
        if config.model == "transformer":
            config.d_model, config.n_heads = dim, 4
            config.n_enc_layers = config.n_dec_layers = 1 if config.smoke else 4
            config.d_ff = 64 if config.smoke else 1024
            if config.smoke:
                config.warmup_steps = 10
        else:
            config.emb_dim = config.hidden_dim = dim
        if config.smoke:
            config.epochs = 1
            config.max_tokens = config.eval_max_tokens = 256
            config.max_new_tokens = 12
    if not config.output_dir:
        suffix = "_smoke" if config.smoke else "_small" if small else ""
        config.output_dir = str(ROOT / "runs" / (config.model + suffix))
    config.output_dir = str(Path(config.output_dir).resolve())
    run_training(config, resume)
