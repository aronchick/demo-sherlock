"""Read stored Expanso core incidents and report verified crop receipts.

Run on the core, after C1 writes its JSONL. No receipt is inferred from HTTP
acceptance or output scheduling: the JPEG bytes are read back from disk.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request

MAX_LINE = 24000000


def receipt(record: dict) -> dict | None:
    if not isinstance(record, dict):
        raise TypeError('stored record must be an object')
    core = record.get('core', {})
    if not isinstance(core, dict) or not isinstance(record.get('crops', {}), dict):
        raise TypeError('invalid stored record')
    if core.get('node') != 'sherlock-core' or core.get('stream') != 'incident':
        return None
    crops = {}
    for name, crop in record.get('crops', {}).items():
        if name not in {'weapon_crop', 'object_crop'}:
            continue
        if not isinstance(crop, dict):
            raise TypeError('invalid crop')
        raw = base64.b64decode(crop['b64'], validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != crop.get('sha256') or digest != crop.get('sha256_edge') or len(raw) != crop.get('bytes'):
            raise ValueError('stored crop integrity mismatch')
        crops[name] = {'sha256': digest, 'bytes': len(raw)}
    if not crops:
        return None
    return {'incident': record['incident'], 'core_node': core['node'],
            'stored_at': core['received_at'], 'proof': 'disk_readback', 'crops': crops}


def scan(directory: pathlib.Path, url: str, sent: set[str]) -> tuple[int, int]:
    delivered = failed = 0
    for path in sorted((directory / 'incident').glob('*.jsonl')):
        with path.open('rb') as stream:
            while line := stream.readline(MAX_LINE + 1):
                if len(line) > MAX_LINE or not line.endswith(b'\n'):
                    # Skip oversized records and an in-progress final append.
                    while len(line) > MAX_LINE and not line.endswith(b'\n'):
                        line = stream.readline(MAX_LINE + 1)
                    continue
                try:
                    item = receipt(json.loads(line))
                    if item is None:
                        continue
                    data = json.dumps(item).encode()
                    key = hashlib.sha256(data).hexdigest()
                    if key in sent:
                        continue
                    request = urllib.request.Request(url, data=data,
                        headers={'Content-Type': 'application/json'}, method='POST')
                    with urllib.request.urlopen(request, timeout=5) as response:
                        if response.status != 200:
                            raise ValueError('receipt rejected')
                    sent.add(key)
                    if len(sent) > 10000:
                        sent.clear()
                    delivered += 1
                except (ValueError, KeyError, TypeError, OSError, urllib.error.URLError):
                    failed += 1
    return delivered, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-dir', type=pathlib.Path, default=pathlib.Path.home() / 'sherlock-core')
    parser.add_argument('--url', default='http://127.0.0.1:23100/api/receipts')
    parser.add_argument('--watch', action='store_true', help='Recheck stored records every two seconds')
    args = parser.parse_args()
    if not args.url.startswith('http://127.0.0.1:'):
        parser.error('receipt destination must use the loopback SSH tunnel')
    sent: set[str] = set()
    while True:
        delivered, failed = scan(args.core_dir, args.url, sent)
        if delivered or failed or not args.watch:
            print(f'Core disk readback: {delivered} receipts delivered; {failed} rejected', flush=True)
        if not args.watch:
            break
        time.sleep(2)


if __name__ == '__main__':
    main()
