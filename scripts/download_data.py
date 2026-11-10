import argparse
import io
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ROOT
from common.data import SPLITS, load_split

ARCHIVES = {"train": "train-en-vi", "dev": "dev-2012-en-vi", "test": "test-2013-en-vi"}
DEFAULT_MIRROR = "https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data"


def download(directory, mirror):
    directory.mkdir(parents=True, exist_ok=True)
    for split, archive in ARCHIVES.items():
        stem = SPLITS[split]
        targets = [directory / f"{stem}.{lang}" for lang in ("en", "vi")]
        if all(path.exists() for path in targets):
            load_split(directory, split)
            continue
        request = urllib.request.Request(
            f"{mirror.rstrip('/')}/{archive}.tgz", headers={"User-Agent": "nmt-en-vi/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as package:
            with tempfile.TemporaryDirectory(dir=directory) as staging:
                staging = Path(staging)
                for target in targets:
                    matches = [
                        m
                        for m in package.getmembers()
                        if Path(m.name).name == target.name and m.isfile()
                    ]
                    if len(matches) != 1:
                        raise ValueError(f"Archive must contain exactly one {target.name}")
                    with (
                        package.extractfile(matches[0]) as source,
                        (staging / target.name).open("wb") as dest,
                    ):
                        shutil.copyfileobj(source, dest)
                load_split(staging, split)
                for target in targets:
                    shutil.move(staging / target.name, target)
        print(f"Ready: {split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/iwslt15_en_vi")
    parser.add_argument("--mirror", default=DEFAULT_MIRROR)
    args = parser.parse_args()
    download(args.data_dir, args.mirror)
