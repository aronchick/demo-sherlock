"""Remote bootstrap transfers only the minimum credential through stdin."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('expanso_brev', SCRIPTS / 'expanso_brev.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_bootstrap_does_not_forward_cloud_admin_key(tmp_path, monkeypatch):
    credentials = tmp_path / '.env'
    credentials.write_text('EXPANSO_CLI_API_KEY=admin-private\nEXPANSO_EDGE_BOOTSTRAP_TOKEN=exp_bk_test\n')
    credentials.chmod(0o600)
    monkeypatch.setenv('EXPANSO_WORKSPACE_ENV', str(credentials))
    assert core.bootstrap_token() == 'exp_bk_test'
    credentials.chmod(0o644)
    with pytest.raises(ValueError, match='owner-only'):
        core.bootstrap_token()


def test_registered_core_reuses_identity_without_bootstrap_secret(monkeypatch):
    calls = []
    monkeypatch.setattr(core, 'find_instance', lambda _: {'status': 'RUNNING'})
    monkeypatch.setattr(core.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(core, 'bootstrap_token', lambda: pytest.fail('already registered'))

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout=b'archive')

    monkeypatch.setattr(core, 'run', run)
    core.core_up('sherlock-nebius-h100')
    assert calls[-1][1]['input'] == b''
    assert 'bootstrap.env' in calls[-1][0][-1]
    assert 'scripts/expanso.sh' in calls[0][0]
    assert 'jobs' in calls[0][0]
    assert 'edge' in calls[0][0]
