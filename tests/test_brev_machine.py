"""The Mac workflow must not retire a machine or touch a busy local service blindly."""
import importlib.util
import json
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('brev_machine', ROOT / 'scripts/brev_machine.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


@pytest.mark.parametrize('value', ['--all', 'one; touch bad', 'name/../../other', 'name\nsecond', ''])
def test_instance_name_rejects_options_and_shell_syntax(value):
    with pytest.raises(ValueError):
        workflow.instance_name(value)


def test_busy_port_is_left_alone():
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        server.listen()
        with pytest.raises(ValueError, match='occupied'):
            workflow.require_free_ports([server.getsockname()[1]])
        assert server.fileno() >= 0


def test_model_environment_is_exact_and_does_not_replace_camera_settings():
    env = workflow.model_env('sherlock-h100')
    assert env['DIRECTOR_MODEL'] == workflow.DIRECTOR
    assert env['ANALYST_MODEL'] == env['VERIFIER_MODEL'] == workflow.VLM
    assert env['BREV_HOST'] == 'sherlock-h100'
    assert env['SHERLOCK_MOCK'] == '0'
    assert env['DIRECTOR_BASE_URL'] == 'http://127.0.0.1:18001/v1'
    assert not {'CAMERAS', 'SHERLOCK_EDGE'} & env.keys()


@pytest.fixture
def retirement(monkeypatch):
    calls = []
    items = {'old': {'id': 'old-id', 'name': 'old', 'status': 'STOPPED'},
             'new': {'id': 'new-id', 'name': 'new', 'status': 'RUNNING'}}
    monkeypatch.setattr(workflow, 'find_instance', lambda name: items[name])
    monkeypatch.setattr(workflow, 'connection', lambda: {'id': 'new-id', 'name': 'new'})
    monkeypatch.setattr(workflow, 'smoke', lambda name: calls.append(('smoke', name)))
    monkeypatch.setattr(workflow, 'run', lambda args, **kw: calls.append(args))
    return calls, items


def test_retirement_refuses_same_instance_id(retirement):
    calls, items = retirement
    items['old']['id'] = 'new-id'
    with pytest.raises(ValueError, match='different'):
        workflow.retire('old', 'new')
    assert calls == []


def test_retirement_does_not_stop_if_replacement_smoke_fails(retirement, monkeypatch):
    calls, _ = retirement

    def fail(name):
        raise subprocess.CalledProcessError(1, ['smoke'])

    monkeypatch.setattr(workflow, 'smoke', fail)
    with pytest.raises(subprocess.CalledProcessError):
        workflow.retire('old', 'new')
    assert calls == []


def test_retirement_only_stops_named_old_after_smoke_and_reads_status(retirement):
    calls, _ = retirement
    workflow.retire('old', 'new')
    assert calls == [('smoke', 'new'), ['brev', 'stop', 'old']]


def test_retirement_reports_unconfirmed_stop(retirement):
    calls, items = retirement
    items['old']['status'] = 'STOPPING'
    with pytest.raises(ValueError, match='not confirmed'):
        workflow.retire('old', 'new')
    assert calls[-1] == ['brev', 'stop', 'old']


def test_stale_pid_cannot_authorize_retirement(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'RUNTIME', tmp_path)
    (tmp_path / 'connection.json').write_text(json.dumps({'name': 'new', 'pid': 123}))
    monkeypatch.setattr(workflow.os, 'kill', lambda *_: None)
    monkeypatch.setattr(workflow, 'run', lambda *a, **kw: SimpleNamespace(stdout='unrelated process'))
    with pytest.raises(ValueError, match='no longer belongs'):
        workflow.connection()


def test_setup_refuses_unrelated_container_before_stopping_anything(tmp_path):
    fake = tmp_path / 'bin'
    fake.mkdir()
    calls = tmp_path / 'calls'
    docker = fake / 'docker'
    docker.write_text('''#!/bin/bash
printf '%s\\n' "$*" >> "$CALLS"
case "$*" in
  *Config.Image*) echo unrelated/image ;;
  *Config.Cmd*) printf '%s\\n' --model unrelated/model ;;
esac
''')
    docker.chmod(0o755)
    for name in ['nvidia-smi', 'uv', 'curl']:
        tool = fake / name
        tool.write_text('#!/bin/sh\nexit 0\n')
        tool.chmod(0o755)
    import os
    result = subprocess.run(['bash', str(ROOT / 'brev/setup.sh')],
                            env={**os.environ, 'PATH': f'{fake}:{os.environ["PATH"]}', 'CALLS': str(calls)},
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert 'unrecognized director' in result.stderr
    assert 'stop' not in calls.read_text()
    assert 'rm' not in calls.read_text().split()


def test_ssh_disables_inherited_background_multiplexing():
    args = workflow.ssh('sherlock-h100', '-N')
    assert 'ControlMaster=no' in args
    assert 'ControlPath=none' in args
    assert 'ForkAfterAuthentication=no' in args
    assert args[-1] == 'sherlock-h100'


def test_live_checks_explicit_dashboard_port(monkeypatch, tmp_path):
    ports, commands = [], []
    monkeypatch.setattr(workflow, 'connection', lambda: {'name': 'new'})
    monkeypatch.setattr(workflow, 'require_free_ports', lambda values: ports.extend(values))
    monkeypatch.setattr(workflow, 'smoke', lambda *_: None)
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)

    class Presenter:
        pid = 123

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def wait(self):
            assert (tmp_path / '.runtime/sherlock.pid').read_text() == '123\n'
            return 0

    def launch(args, **kw):
        commands.append(args)
        return Presenter()

    monkeypatch.setattr(workflow.subprocess, 'Popen', launch)
    workflow.live(['--port', '8099'])
    assert ports == [8099]
    assert commands[0][-2:] == ['--port', '8099']
    assert not (tmp_path / '.runtime/sherlock.pid').exists()


def test_failed_tunnel_smoke_preserves_previous_receipt_and_stops_own_process(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'RUNTIME', tmp_path)
    receipt = tmp_path / 'connection.json'
    previous = {'name': 'old', 'pid': 99}
    receipt.write_text(json.dumps(previous))
    monkeypatch.setattr(workflow, 'find_instance', lambda _: {'status': 'RUNNING', 'id': 'new-id'})
    monkeypatch.setattr(workflow, 'require_free_ports', lambda _: None)
    monkeypatch.setattr(workflow, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)

    class Tunnel:
        pid = 100
        stopped = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def poll(self):
            return 0 if self.stopped else None

        def terminate(self):
            self.stopped = True

        def wait(self, **_):
            return 0

    tunnel = Tunnel()
    monkeypatch.setattr(workflow.subprocess, 'Popen', lambda *_: tunnel)

    def fail(_):
        raise ValueError('wrong model')

    monkeypatch.setattr(workflow, 'smoke', fail)
    with pytest.raises(ValueError, match='wrong model'):
        workflow.connect('new')
    assert tunnel.stopped
    assert json.loads(receipt.read_text()) == previous


def test_down_waits_for_stopped_readback(monkeypatch):
    states = iter(['RUNNING', 'STOPPING', 'STOPPED'])
    calls = []
    monkeypatch.setattr(workflow, 'find_instance',
                        lambda _: {'id': 'owned', 'status': next(states)})
    monkeypatch.setattr(workflow, 'run', lambda args: calls.append(args))
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    workflow.down('sherlock-h100')
    assert calls == [['brev', 'stop', 'sherlock-h100']]


def test_down_is_idempotent(monkeypatch):
    monkeypatch.setattr(workflow, 'find_instance',
                        lambda _: {'id': 'owned', 'status': 'STOPPED'})
    monkeypatch.setattr(workflow, 'run', lambda _: pytest.fail('already stopped'))
    workflow.down('sherlock-h100')


def test_down_does_not_claim_success_when_still_running(monkeypatch):
    monkeypatch.setattr(workflow, 'find_instance',
                        lambda _: {'id': 'owned', 'status': 'STOPPING'})
    monkeypatch.setattr(workflow, 'run', lambda _: pytest.fail('already stopping'))
    with pytest.raises(ValueError, match='Stop not confirmed'):
        workflow.down('sherlock-h100', timeout=0)


def test_unsupported_stop_warns_about_billing_without_deleting(monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, 'find_instance',
                        lambda _: {'id': 'owned', 'status': 'RUNNING'})

    def rejected(args):
        calls.append(args)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(workflow, 'run', rejected)
    with pytest.raises(ValueError, match='GPU billing may continue'):
        workflow.down('sherlock-h100')
    assert calls == [['brev', 'stop', 'sherlock-h100']]
