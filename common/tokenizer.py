from pathlib import Path

import sentencepiece as spm

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class Tokenizer:
    def __init__(self, path):
        self.path = Path(path)
        self.processor = spm.SentencePieceProcessor(model_file=str(self.path))
        ids = (
            self.processor.pad_id(),
            self.processor.unk_id(),
            self.processor.bos_id(),
            self.processor.eos_id(),
        )
        if ids != (PAD_ID, UNK_ID, BOS_ID, EOS_ID):
            raise ValueError(f"Unexpected special token IDs: {ids}")

    @property
    def vocab_size(self):
        return self.processor.vocab_size()

    def encode(self, text):
        return self.processor.encode(text, out_type=int)

    def encode_batch(self, lines):
        return self.processor.encode(lines, out_type=int)

    def decode(self, ids):
        tokens = []
        for token in ids:
            token = int(token)
            if token == EOS_ID:
                break
            if token not in (PAD_ID, BOS_ID):
                tokens.append(token)
        return self.processor.decode(tokens)


def load_tokenizers(directory):
    directory = Path(directory)
    return Tokenizer(directory / "spm_en.model"), Tokenizer(directory / "spm_vi.model")


def train_tokenizer(input_path, prefix, vocab_size=8000):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(prefix),
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        normalization_rule_name="identity",
        remove_extra_whitespaces=True,
        shuffle_input_sentence=False,
        num_threads=1,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    return Tokenizer(prefix.with_suffix(".model"))
