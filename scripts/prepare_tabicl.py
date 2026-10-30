"""Download one author checkpoint pinned by revision, size, and SHA256."""
import argparse
import hashlib
import json
import urllib.request

from src.paths import PROJECT_ROOT, EXPERIMENTS_DIR

REPO = "jingang/TabICL"
REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
FILENAME = "tabicl-classifier-v2-20260212.ckpt"
EXPECTED_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
EXPECTED_SIZE = 110368038


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(proxy=None):
    directory = PROJECT_ROOT / "models/tabicl_pretrained"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"
    if path.exists():
        if sha(path) != EXPECTED_SHA256 or path.stat().st_size != EXPECTED_SIZE:
            raise ValueError("Existing checkpoint differs; preserve it and investigate")
    else:
        handler = urllib.request.ProxyHandler({"https": proxy, "http": proxy}) if proxy else urllib.request.ProxyHandler()
        opener = urllib.request.build_opener(handler)
        temporary = directory / (FILENAME + ".partial")
        with opener.open(url, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if sha(temporary) != EXPECTED_SHA256 or temporary.stat().st_size != EXPECTED_SIZE:
            raise ValueError("Checkpoint download integrity failed; partial file preserved")
        temporary.replace(path)
    report = EXPERIMENTS_DIR / "tabicl_20261030"
    report.mkdir(parents=True, exist_ok=True)
    metadata = dict(repository=REPO, revision=REVISION, filename=FILENAME, size=EXPECTED_SIZE,
                    sha256=EXPECTED_SHA256, url=url, model_card=f"https://huggingface.co/{REPO}",
                    license="bsd-3-clause per author model card",
                    pretraining_reference="https://github.com/soda-inria/tabicl", local_only_inference=True,
                    data_transfer="Only checkpoint downloaded; competition feature rows and labels not uploaded")
    (report / "pretrained_source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Pinned author checkpoint verified:", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy")
    main(parser.parse_args().proxy)
