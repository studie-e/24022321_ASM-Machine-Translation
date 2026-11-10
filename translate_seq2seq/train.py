import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cli import train_main
from translate_seq2seq.config import Seq2SeqConfig

if __name__ == "__main__":
    train_main(Seq2SeqConfig)
