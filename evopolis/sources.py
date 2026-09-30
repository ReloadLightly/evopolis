"""Small, atomic, streaming downloads of the public research sources."""

import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen

REVISION = "4f1a99a9d150f9fa6bad1a0f70f11c0673d46763"
SOURCES = {
    "sustainable_behavior.csv": "https://storage.googleapis.com/sustainable_behavior/sustainable_behavior.csv",
    "sustainable_behavior.ipynb": f"https://raw.githubusercontent.com/google-deepmind/sustainable_behavior/{REVISION}/notebooks/sustainable_behavior.ipynb",
    "paper.xml": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11929920/fullTextXML",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(directory: Path, expected: dict) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, url in SOURCES.items():
        target = directory / name
        if not target.exists():
            print(f"Downloading {url}", flush=True)
            partial = target.with_suffix(target.suffix + ".partial")
            with urlopen(url, timeout=120) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            partial.replace(target)
        digest = sha256(target)
        if name in expected and digest != expected[name]:
            raise ValueError(f"Source checksum changed: {name}; inspect before updating the manifest")
        manifest.append({"file": name, "url": url, "sha256": digest, "bytes": target.stat().st_size})
    return manifest
