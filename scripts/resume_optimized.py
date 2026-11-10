import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.config import config_from_dict
from common.engine import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    checkpoint = torch.load(args.run_dir / 'last.pt', map_location='cpu', weights_only=False)
    cfg = config_from_dict(checkpoint['config'])
    completed = checkpoint['epoch']
    del checkpoint
    if args.epochs <= completed:
        raise ValueError(f'Checkpoint already completed epoch {completed}; increase --epochs')
    cfg.output_dir = str(args.run_dir.resolve())
    cfg.epochs = args.epochs
    if args.amp is not None:
        cfg.amp = args.amp
    print(f'Resuming epoch {completed + 1}/{cfg.epochs} from {cfg.output_dir}', flush=True)
    run_training(cfg, resume=True)


if __name__ == '__main__':
    main()
