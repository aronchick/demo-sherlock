"""Process ownership and Cloud reconciliation never rely on a PID alone."""
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('expanso_process', ROOT / 'scripts/expanso_process.py')
process = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(process)


def test_owned_process_reuses_and_restarts_changed_revision(tmp_path):
    script = tmp_path / 'watcher.py'
    script.write_text('import time\ntime.sleep(120)\n')
    args = [sys.executable, str(script)]
    try:
        process.ensure(tmp_path, 'receipts', args, [script])
        first = process.owned(tmp_path, 'receipts')
        process.ensure(tmp_path, 'receipts', args, [script])
        assert process.owned(tmp_path, 'receipts')['pid'] == first['pid']
        script.write_text('import time\ntime.sleep(119)\n')
        process.ensure(tmp_path, 'receipts', args, [script])
        assert process.owned(tmp_path, 'receipts')['pid'] != first['pid']
        assert not process.identity(first['pid'])
    finally:
        process.stop(tmp_path, 'receipts')
    assert process.owned(tmp_path, 'receipts') is None


def test_recycled_pid_is_never_signaled(tmp_path, monkeypatch):
    folder = tmp_path / '.runtime/expanso'
    folder.mkdir(parents=True)
    (folder / 'core.process.json').write_text(json.dumps({'pid': 1234, 'identity': 'old'}))
    monkeypatch.setattr(process, 'identity', lambda _: 'new unrelated process')
    monkeypatch.setattr(process.os, 'killpg', lambda *args: pytest.fail('must not signal'))
    with pytest.raises(ValueError, match='identity changed'):
        process.stop(tmp_path, 'core')


def test_legacy_generic_node_pid_is_not_adopted(tmp_path, monkeypatch):
    folder = tmp_path / '.runtime/expanso'
    folder.mkdir(parents=True)
    (folder / 'edge.pid').write_text('1234')
    monkeypatch.setattr(process, 'identity', lambda _: 'start expanso-edge')
    monkeypatch.setattr(process, 'command', lambda _: ['expanso-edge', 'run', '--data-dir', '/other-project'])
    with pytest.raises(ValueError, match='cannot be proven'):
        process.owned(tmp_path, 'edge')


@pytest.mark.parametrize('terminal', [False, True])
def test_unchanged_deploy_preserves_healthy_current_version(tmp_path, terminal):
    text = (ROOT / 'scripts/expanso.sh').read_text()
    functions = text[text.index('deploy() {'):text.index('\nbootstrap()')]
    marker = tmp_path / 'rerun'
    observed = 'failed' if terminal else 'running'
    script = '''set -euo pipefail
say() { :; }
die() { echo "$*" >&2; exit 1; }
sleep() { :; }
'''+functions+'''
cli() {
  case "$2" in
    deploy) echo NO_CHANGES_DETECTED; return 1 ;;
    describe) echo '{"status":{"version":3,"state":{"state_type":"running"}}}' ;;
    executions)
      state="'''+observed+'''"
      [[ ! -f "$MARKER" ]] || state=running
      printf '[{"job_version":3,"status":{"desired_state":{"state_type":"running"},"observed_state":{"state_type":"%s"}}}]' "$state"
      ;;
    rerun) touch "$MARKER" ;;
    *) exit 9 ;;
  esac
}
deploy cli -- "$JOB"
'''
    result = subprocess.run(['bash', '-c', script], env={'PATH': '/opt/homebrew/bin:/usr/bin:/bin',
                            'MARKER': str(marker), 'JOB': str(ROOT / 'jobs/sherlock-incidents.yaml')},
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert marker.exists() == terminal
