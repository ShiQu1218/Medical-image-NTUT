"""Download BUS-UCLM v3 from its public Mendeley record and verify hashes."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def main():
    root = Path('external/BUS-UCLM')
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1)))
    metadata = session.get('https://data.mendeley.com/public-api/datasets/7fvgj4jsp7', timeout=60).json()
    if metadata['version'] != 3:
        raise ValueError('Review dataset version before downloading')
    (root/'source.json').write_text(json.dumps(metadata, indent=2))

    def fetch(record):
        target = root/record['folder_id']/record['filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = record['content_details']['sha256_hash']
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return
        response = session.get(record['content_details']['download_url'], timeout=120)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != expected:
            raise ValueError(f'Hash mismatch: {target}')
        temporary = target.with_suffix(target.suffix+'.part')
        temporary.write_bytes(response.content)
        temporary.replace(target)

    with ThreadPoolExecutor(max_workers=12) as pool:
        for index, _ in enumerate(pool.map(fetch, metadata['files']), 1):
            if index % 100 == 0:
                print(f'{index}/{len(metadata["files"])} verified', flush=True)


if __name__ == '__main__':
    main()
