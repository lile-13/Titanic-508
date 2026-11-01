"""Fetch one public, pinned author checkpoint without uploading competition data."""
import argparse
import hashlib
import urllib.request

from src.paths import PROJECT_ROOT as P, EXPERIMENTS_DIR
from src.train import _write_json

REPO="Prior-Labs/TabPFN-v2-clf"
REVISION="275f19d2add9c2c0de38f9ebaded3856599fa2e7"
CHECKPOINT="tabpfn-v2-classifier-v2_default.ckpt"
SHA256="cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
SIZE=29009539


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--proxy",default=None)
    args=parser.parse_args()
    path=P/"models/tabpfn_pretrained"/CHECKPOINT
    path.parent.mkdir(parents=True,exist_ok=True)
    url=f"https://huggingface.co/{REPO}/resolve/{REVISION}/{CHECKPOINT}"
    if not path.exists():
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({"https":args.proxy}) if args.proxy else urllib.request.ProxyHandler())
        with opener.open(url,timeout=60) as response, path.open("wb") as target:
            while chunk:=response.read(1024*1024):
                target.write(chunk)
    assert path.stat().st_size==SIZE
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SHA256
    (EXPERIMENTS_DIR/"tabpfn_v2_20261031").mkdir(parents=True,exist_ok=True)
    _write_json(EXPERIMENTS_DIR/"tabpfn_v2_20261031/pretrained_source.json",dict(
        repository=REPO,revision=REVISION,filename=CHECKPOINT,sha256=SHA256,bytes=SIZE,
        public=True,gated=False,source_url=url,
        author_model_card=f"https://huggingface.co/{REPO}",
        attribution="TabPFN v2 by Prior Labs; Hollmann et al., Nature 2025, doi:10.1038/s41586-024-08328-6",
        license="Prior Labs License (Apache 2.0 with additional attribution requirement); model weights remain local and untracked",
        usage="Explicit V2 base checkpoint. No competition data uploaded, no account terms accepted, no gated checkpoint requested."))
    print(f"Verified public author checkpoint: {path.name}, {SIZE} bytes")


if __name__=="__main__":
    main()
