"""Core disk-readback receipts, separate from signed acquisition records.

Loopback callers are trusted transport peers, not an independent attestation.
Only crops matching the local signed artifact and current bytes are displayed.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from datetime import datetime

from .incidents import ID, IncidentStore

CROPS = {'weapon_crop', 'object_crop'}
LIMIT = 100


class ReceiptStore:
    def __init__(self, incidents: IncidentStore):
        self.incidents = incidents
        self.path = incidents.root / '.core-receipts.json'
        self.lock = threading.RLock()
        self.records: dict[str, dict] = {}
        try:
            if self.path.stat().st_size <= 262144:
                records = json.loads(self.path.read_text())
                if isinstance(records, list):
                    for record in records[-LIMIT:]:
                        try:
                            valid = self.validate(record)
                            self.records[valid['incident']] = valid
                        except (ValueError, OSError, KeyError, TypeError):
                            continue
        except (OSError, ValueError):
            pass

    def validate(self, receipt: object) -> dict:
        if not isinstance(receipt, dict):
            raise TypeError('receipt must be an object')
        iid = receipt.get('incident')
        if not isinstance(iid, str) or not ID.fullmatch(iid):
            raise ValueError('invalid incident')
        if receipt.get('core_node') != 'sherlock-core' or receipt.get('proof') != 'disk_readback':
            raise ValueError('core disk readback required')
        at = receipt.get('stored_at')
        if not isinstance(at, str) or len(at) > 64:
            raise ValueError('invalid storage timestamp')
        parsed = datetime.fromisoformat(at)
        if parsed.tzinfo is None or not 0 < parsed.timestamp() <= time.time() + 300:
            raise ValueError('invalid storage timestamp')
        crops = receipt.get('crops')
        if not isinstance(crops, dict) or not 1 <= len(crops) <= 2 or not set(crops) <= CROPS:
            raise ValueError('only bounded object crops accepted')
        record = self.incidents.read(iid)
        if any(record['verification'].get(key) != 'verified' for key in ('signature', 'artifacts', 'history')):
            raise ValueError('local signed evidence does not verify')
        accepted = {}
        for name, crop in crops.items():
            if not isinstance(crop, dict):
                raise TypeError('invalid crop')
            sha = crop.get('sha256')
            size = crop.get('bytes')
            if not isinstance(sha, str) or not re.fullmatch('[a-f0-9]{64}', sha) or type(size) is not int or not 0 < size <= 16000000:
                raise ValueError('invalid crop hash or size')
            artifact = next((a for a in record['artifacts'] if a['name'] == name and a['sha256'] == sha), None)
            if not artifact:
                raise ValueError('crop does not match local evidence')
            raw = self.incidents.path(iid, artifact['filename']).read_bytes()
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('crop bytes do not match local evidence')
            accepted[name] = {'sha256': sha, 'bytes': size, 'url': artifact['url']}
        return {'incident': iid, 'core_node': 'sherlock-core', 'proof': 'disk_readback',
                'stored_at': at, 'crops': accepted}

    def accept(self, receipt: object) -> dict:
        with self.lock:
            record = self.validate(receipt)
            self.records.pop(record['incident'], None)
            self.records[record['incident']] = record
            while len(self.records) > LIMIT:
                self.records.pop(next(iter(self.records)))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            temporary.write_text(json.dumps(list(self.records.values())))
            temporary.replace(self.path)
            return record

    def list(self) -> list[dict]:
        with self.lock:
            # A changed/deleted local artifact must no longer appear confirmed.
            records = []
            for item in reversed(list(self.records.values())):
                try:
                    records.append(self.validate(item))
                except (ValueError, OSError, KeyError, TypeError):
                    continue
            return records
