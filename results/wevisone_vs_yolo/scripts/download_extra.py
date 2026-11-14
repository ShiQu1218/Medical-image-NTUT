"""Download BUS-BRA (Zenodo 8231412, md5-verified) and BUS_UC (Mendeley 3ksd7w7jkx, sha256-verified)."""
import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1] / "external"


def session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1)))
    return s


def busbra(s):
    root = ROOT / "BUS-BRA"
    root.mkdir(parents=True, exist_ok=True)
    meta = s.get("https://zenodo.org/api/records/8231412", timeout=60).json()
    (root / "source.json").write_text(json.dumps(meta, indent=2))
    f = meta["files"][0]
    zpath = root / f["key"]
    md5 = f["checksum"].split(":")[1]
    if not zpath.exists() or hashlib.md5(zpath.read_bytes()).hexdigest() != md5:
        r = s.get(f["links"]["self"], timeout=600)
        r.raise_for_status()
        if hashlib.md5(r.content).hexdigest() != md5:
            raise ValueError("BUS-BRA md5 mismatch")
        zpath.write_bytes(r.content)
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():  # extract data files only, refuse path traversal
            if ".." in Path(name).parts or Path(name).is_absolute():
                raise ValueError(name)
        z.extractall(root / "data")
    print("BUS-BRA ok")


def busuc(s):
    root = ROOT / "BUS_UC"
    root.mkdir(parents=True, exist_ok=True)
    meta = s.get("https://data.mendeley.com/public-api/datasets/3ksd7w7jkx", timeout=60).json()
    (root / "source.json").write_text(json.dumps(meta, indent=2))

    def fetch(rec):
        target = root / rec["folder_id"] / rec["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = rec["content_details"]["sha256_hash"]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return
        r = s.get(rec["content_details"]["download_url"], timeout=120)
        r.raise_for_status()
        if hashlib.sha256(r.content).hexdigest() != expected:
            raise ValueError(f"hash mismatch {target}")
        target.write_bytes(r.content)

    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, _ in enumerate(pool.map(fetch, meta["files"]), 1):
            if i % 500 == 0:
                print(f"BUS_UC {i}/{len(meta['files'])}", flush=True)
    print("BUS_UC ok", len(meta["files"]), "files")


if __name__ == "__main__":
    s = session()
    busbra(s)
    busuc(s)
