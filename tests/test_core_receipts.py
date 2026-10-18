"""A remote acceptance is insufficient; persisted bytes must match signed evidence."""
import base64
import hashlib
import importlib.util
import json
import pathlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sherlock.dashboard.server import make_app
from sherlock.incidents import IncidentStore
from sherlock.receipts import ReceiptStore

IID = '20261017-120000-001'
RAW = b'actual stored crop'
SHA = hashlib.sha256(RAW).hexdigest()
AT = '2026-10-17T12:00:00Z'
SPEC = importlib.util.spec_from_file_location('core_receipts', pathlib.Path(__file__).parents[1] / 'scripts/core_receipts.py')
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def setup(tmp_path):
    store = IncidentStore(tmp_path)
    path = store.path(IID, 'weapon_crop.jpg')
    path.parent.mkdir()
    path.write_bytes(RAW)
    store.save({'incident': IID, 'action': 'alert', 'verified': True, 'claim': 'knife',
                't': 10, 'evidence': ['weapon_crop']}, [], {'mock': False})
    app = make_app(SimpleNamespace(incidents=store, edge=None, cfg=SimpleNamespace(mock=False)))
    return store, TestClient(app, client=('127.0.0.1', 5000)), app


def sample():
    return {'incident': IID, 'core_node': 'sherlock-core', 'stored_at': AT,
            'proof': 'disk_readback', 'crops': {'weapon_crop': {'sha256': SHA, 'bytes': len(RAW)}}}


def test_verified_receipt_persists_without_resigning(tmp_path):
    store, client, _ = setup(tmp_path)
    before = store.path(IID).read_bytes()
    assert client.post('/api/receipts', json=sample()).status_code == 200
    records = client.get('/expanso').json()['receipts']
    assert records[0]['crops']['weapon_crop']['url'].endswith(SHA + '.jpg')
    assert store.path(IID).read_bytes() == before
    assert ReceiptStore(store).list() == records
    assert client.post('/api/receipts', json=sample()).status_code == 200
    assert len(client.get('/expanso').json()['receipts']) == 1
    store.path(IID, SHA + '.jpg').write_bytes(b'tampered')
    assert client.get('/expanso').json()['receipts'] == []


@pytest.mark.parametrize('change', [
    {'incident': '20261017-120000-999'}, {'incident': '../secret'},
    {'core_node': 'other'}, {'proof': 'http_accepted'}, {'stored_at': 'not a date'},
    {'crops': {}}, {'crops': {'face_crop': {'sha256': SHA, 'bytes': len(RAW)}}},
    {'crops': {'weapon_crop': {'sha256': '0' * 64, 'bytes': len(RAW)}}},
    {'crops': {'weapon_crop': {'sha256': SHA, 'bytes': 1}}},
    {'crops': {'weapon_crop': {'sha256': SHA, 'bytes': True}}},
])
def test_receipt_rejects_unknown_or_mismatched(tmp_path, change):
    _, client, _ = setup(tmp_path)
    assert client.post('/api/receipts', json={**sample(), **change}).status_code == 400
    assert client.get('/expanso').json()['receipts'] == []


def test_receipt_input_limits(tmp_path):
    _, client, app = setup(tmp_path)
    assert client.post('/api/receipts', content=b'not json').status_code == 400
    assert client.post('/api/receipts', json=[]).status_code == 400
    assert client.post('/api/receipts', content=b'x' * 4097).status_code == 413
    remote = TestClient(app, client=('203.0.113.9', 5000))
    assert remote.post('/api/receipts', json=sample()).status_code == 403


def stored():
    return {'incident': IID, 'core': {'node': 'sherlock-core', 'stream': 'incident', 'received_at': AT},
            'crops': {'weapon_crop': {'b64': base64.b64encode(RAW).decode(),
                      'sha256': SHA, 'sha256_edge': SHA, 'bytes': len(RAW)}}}


def test_core_helper_checks_stored_bytes():
    assert helper.receipt(stored()) == sample()
    record = stored()
    record['crops']['weapon_crop']['sha256_edge'] = '0' * 64
    with pytest.raises(ValueError, match='integrity mismatch'):
        helper.receipt(record)


def test_core_readback_waits_for_complete_line(tmp_path, monkeypatch):
    folder = tmp_path / 'incident'
    folder.mkdir()
    path = folder / '2026-10-17.jsonl'
    path.write_text(json.dumps(stored()))
    sent = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def post(request, **kwargs):
        sent.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(helper.urllib.request, 'urlopen', post)
    seen = set()
    assert helper.scan(tmp_path, 'http://127.0.0.1:23100/api/receipts', seen) == (0, 0)
    assert not sent
    path.write_text(path.read_text() + '\n')
    assert helper.scan(tmp_path, 'http://127.0.0.1:23100/api/receipts', seen) == (1, 0)
    assert sent == [sample()]
    assert helper.scan(tmp_path, 'http://127.0.0.1:23100/api/receipts', seen) == (0, 0)


def test_receipt_map_is_bounded(tmp_path, monkeypatch):
    store, _, _ = setup(tmp_path)
    receipts = ReceiptStore(store)
    monkeypatch.setattr('sherlock.receipts.LIMIT', 2)
    monkeypatch.setattr(receipts, 'validate', lambda item: item)
    for number in range(3):
        receipts.accept({'incident': str(number)})
    assert list(receipts.records) == ['1', '2']
    assert len(json.loads(receipts.path.read_text())) == 2


def test_helper_retries_after_dashboard_returns(tmp_path, monkeypatch):
    folder = tmp_path / 'incident'
    folder.mkdir()
    (folder / '2026-10-17.jsonl').write_text(json.dumps(stored()) + '\n')

    def offline(*args, **kwargs):
        raise OSError('dashboard unavailable')

    monkeypatch.setattr(helper.urllib.request, 'urlopen', offline)
    sent = set()
    assert helper.scan(tmp_path, 'http://127.0.0.1:23100/api/receipts', sent) == (0, 1)
    assert not sent
    monkeypatch.setattr(helper.urllib.request, 'urlopen', lambda *a, **k: nullcontext(SimpleNamespace(status=200)))
    assert helper.scan(tmp_path, 'http://127.0.0.1:23100/api/receipts', sent) == (1, 0)
    assert len(sent) == 1
