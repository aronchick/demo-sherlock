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


def test_ssh_readiness_refreshes_alias_after_boot(monkeypatch):
    calls = []
    refreshed = False

    def execute(args, **kwargs):
        nonlocal refreshed
        calls.append(args)
        if args == ['brev', 'refresh']:
            refreshed = True
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=0 if refreshed else 255)

    monkeypatch.setattr(workflow.subprocess, 'run', execute)
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    workflow.wait_for_ssh('new')
    assert len(calls) == 3
    assert calls[1] == ['brev', 'refresh']
    assert calls[0][-2:] == ['new', 'true']
    assert calls[2][-2:] == ['new', 'true']


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
    def launch(command):
        assert '127.0.0.1:23121:127.0.0.1:23121' in command
        assert '127.0.0.1:23100:127.0.0.1:8000' in command
        assert '-R' in command
        return tunnel

    monkeypatch.setattr(workflow.subprocess, 'Popen', launch)

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


def test_start_retries_provider_stopping_race(monkeypatch):
    attempts = []

    def start(args, **kwargs):
        attempts.append(args)
        if len(attempts) == 1:
            raise subprocess.CalledProcessError(1, args, stderr='Instance is not stopped status=STOPPING')
        return SimpleNamespace(stdout='started')

    monkeypatch.setattr(workflow, 'run', start)
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    workflow.start_existing('new')
    assert attempts == [['brev', 'start', 'new', '--detached']] * 2


def test_start_does_not_retry_unrelated_provider_errors(monkeypatch):
    def denied(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr='access denied')

    monkeypatch.setattr(workflow, 'run', denied)
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: pytest.fail('must not retry'))
    with pytest.raises(subprocess.CalledProcessError):
        workflow.start_existing('new')


def test_start_waits_past_three_minutes_but_has_a_ten_minute_limit(monkeypatch):
    attempts = []
    clock = iter([0, 181, 599, 600])

    def stopping(args, **kwargs):
        attempts.append(args)
        raise subprocess.CalledProcessError(1, args, stderr='status=STOPPING')

    monkeypatch.setattr(workflow, 'run', stopping)
    monkeypatch.setattr(workflow.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(workflow.time, 'sleep', lambda _: None)
    with pytest.raises(subprocess.CalledProcessError):
        workflow.start_existing('new')
    assert len(attempts) == 3


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


def test_owned_pid_recycling_never_signals(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'RUNTIME', tmp_path)
    (tmp_path / 'presenter.json').write_text(json.dumps({'pid': 42, 'identity': 'original'}))
    monkeypatch.setattr(workflow, 'process_identity', lambda _: 'unrelated process')
    monkeypatch.setattr(workflow.os, 'killpg', lambda *_: pytest.fail('must not signal'))
    with pytest.raises(ValueError, match='ownership changed'):
        workflow.stop_owned('presenter')


def test_missing_receipts_do_not_prevent_gpu_shutdown(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(workflow, 'RUNTIME', tmp_path)
    calls = []
    monkeypatch.setattr(workflow, 'down', calls.append)
    workflow.demo_down('new')
    assert calls == ['new']


def test_local_stop_failure_still_attempts_gpu_stop(monkeypatch):
    monkeypatch.setattr(workflow, 'run', lambda *a, **kw: None)
    calls = []

    def stop(kind):
        calls.append(kind)
        if kind == 'presenter':
            raise ValueError('ownership changed')

    monkeypatch.setattr(workflow, 'stop_owned', stop)
    monkeypatch.setattr(workflow, 'down', calls.append)
    with pytest.raises(ValueError, match='ownership changed'):
        workflow.demo_down('new')
    assert calls == ['presenter', 'tunnel', 'new']


def test_local_only_does_not_stop_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, 'stop_owned', calls.append)
    monkeypatch.setattr(workflow, 'down', lambda _: pytest.fail('must not stop GPU'))
    workflow.demo_down('new', local_only=True)
    assert calls == ['presenter']


def test_matching_tunnel_is_reused_and_verified(monkeypatch):
    state = {'name': 'new', 'id': 'owned', 'port': 8000}
    monkeypatch.setattr(workflow, 'read_owned', lambda _: state)
    monkeypatch.setattr(workflow, 'launch_owned', lambda *_: pytest.fail('must reuse'))
    calls = []
    monkeypatch.setattr(workflow, 'smoke', calls.append)
    workflow.ensure_tunnel('new', {'id': 'owned'}, 8000)
    assert calls == ['new']


def test_tunnel_target_mismatch_is_not_replaced(monkeypatch):
    monkeypatch.setattr(workflow, 'read_owned', lambda _: {'name': 'other'})
    monkeypatch.setattr(workflow, 'launch_owned', lambda *_: pytest.fail('must not replace'))
    with pytest.raises(ValueError, match='different settings'):
        workflow.ensure_tunnel('new', {'id': 'owned'}, 8000)


def test_presenter_reuses_only_matching_healthy_configuration(monkeypatch, tmp_path):
    settings = {'name': 'new', 'id': 'owned', 'port': 8000, 'config_digest': 'abc'}
    monkeypatch.setattr(workflow, 'read_owned', lambda _: settings)
    monkeypatch.setattr(workflow, 'presenter_settings', lambda *_: settings)
    monkeypatch.setattr(workflow, 'presenter_health', lambda _: True)
    monkeypatch.setattr(workflow, 'launch_owned', lambda *_: pytest.fail('must reuse'))
    workflow.ensure_presenter('new', {'id': 'owned'}, 8000)
    monkeypatch.setattr(workflow, 'presenter_settings', lambda *_: {**settings, 'config_digest': 'changed'})
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    (tmp_path / '.runtime').mkdir()
    actions = []
    monkeypatch.setattr(workflow, 'stop_owned', lambda kind: actions.append(('stop', kind)))
    monkeypatch.setattr(workflow, 'require_free_ports', lambda ports: actions.append(('ports', ports)))
    monkeypatch.setattr(workflow, 'launch_owned', lambda kind, *args, **kwargs:
                        actions.append(('launch', kind)) or {'pid': 123})
    workflow.ensure_presenter('new', {'id': 'owned'}, 8000)
    assert actions == [('stop', 'presenter'), ('ports', [8000]), ('launch', 'presenter')]


def test_new_tunnel_smoke_failure_stops_only_new_tunnel(monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, 'read_owned', lambda _: None)
    monkeypatch.setattr(workflow, 'require_free_ports', lambda _: None)
    monkeypatch.setattr(workflow, 'launch_owned', lambda *a: calls.append(a[0]))
    monkeypatch.setattr(workflow, 'stop_owned', lambda kind: calls.append(('stop', kind)))
    monkeypatch.setattr(workflow, 'smoke', lambda _: (_ for _ in ()).throw(ValueError('failed')))
    with pytest.raises(ValueError, match='failed'):
        workflow.ensure_tunnel('new', {'id': 'owned'}, 8000)
    assert calls == ['tunnel', ('stop', 'tunnel')]


def test_reverse_transport_uses_actual_presenter_port():
    command = workflow.tunnel_command('new', 8099)
    assert '127.0.0.1:23100:127.0.0.1:8099' in command
    assert command[-1] == 'new'


def test_configuration_fingerprint_hides_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    (tmp_path / '.env').write_text('SECRET=private-value\n')
    before = workflow.presenter_settings('new', {'id': 'owned'}, 8000)
    assert 'private-value' not in json.dumps(before)
    (tmp_path / '.env').write_text('SECRET=changed-value\n')
    after = workflow.presenter_settings('new', {'id': 'owned'}, 8000)
    assert before['config_digest'] != after['config_digest']


def test_camera_location_override_changes_presenter_fingerprint(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    monkeypatch.setenv('CAMERA_LOCATIONS', '{"hero":{"area":"West room"}}')
    before = workflow.presenter_settings('new', {'id': 'owned'}, 8000)
    monkeypatch.setenv('CAMERA_LOCATIONS', '{"hero":{"area":"East room"}}')
    after = workflow.presenter_settings('new', {'id': 'owned'}, 8000)
    assert before['config_digest'] != after['config_digest']


def test_demo_up_never_creates_missing_instance(monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, 'read_owned', lambda _: None)
    monkeypatch.setattr(workflow, 'require_free_ports', lambda _: None)
    monkeypatch.setattr(workflow, 'inventory', list)
    monkeypatch.setattr(workflow, 'run', lambda args, **_: calls.append(args))
    with pytest.raises(ValueError, match='existing instance'):
        workflow.demo_up('absent', 8000)
    assert calls == []


def test_configuration_detects_loaded_source_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    source = tmp_path / 'run.py'
    source.write_text('VERSION = 1\n')
    before = workflow.presenter_settings('new', {'id': 'owned'}, 8000)
    source.write_text('VERSION = 2\n')
    after = workflow.presenter_settings('new', {'id': 'owned'}, 8000)
    assert before['config_digest'] != after['config_digest']


def test_sandbox_cannot_inherit_live_transport_or_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    monkeypatch.setenv('SHERLOCK_EDGE', '1')
    monkeypatch.setenv('EVIDENCE_DIR', '/live/evidence')
    monkeypatch.setattr(workflow, 'read_owned', lambda _: None)
    monkeypatch.setattr(workflow, 'require_free_ports', lambda _: None)
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs['env'])
        raise RuntimeError('launch captured')

    monkeypatch.setattr(workflow, 'launch_owned', capture)
    with pytest.raises(RuntimeError, match='launch captured'):
        workflow.sandbox_up(8099)
    assert captured['SHERLOCK_EDGE'] == '0'
    assert captured['EVIDENCE_DIR'] == str(tmp_path / '.runtime' / 'rehearsal-evidence')


def test_demo_up_orders_real_transport_before_presenter(monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, 'read_owned', lambda _: {'owned': True})
    monkeypatch.setattr(workflow, 'up', lambda name: calls.append(('gpu', name)))
    monkeypatch.setattr(workflow, 'find_instance', lambda _: {'id': 'id', 'status': 'RUNNING'})
    monkeypatch.setattr(workflow, 'run', lambda args, **_: calls.append(args))
    monkeypatch.setattr(workflow, 'ensure_tunnel', lambda *a: calls.append('tunnel'))
    monkeypatch.setattr(workflow, 'ensure_presenter', lambda *a: calls.append('presenter'))
    workflow.demo_up('new', 8000)
    assert calls == [('gpu', 'new'),
                     ['uv', 'run', '--no-project', 'scripts/expanso_brev.py', 'up', 'new'],
                     'tunnel', ['bash', 'scripts/expanso.sh', 'edge'], 'presenter']


def test_owned_process_launch_and_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'RUNTIME', tmp_path / 'runtime')
    monkeypatch.setattr(workflow, 'ROOT', tmp_path)
    state = workflow.launch_owned('tunnel', ['sleep', '30'], {'name': 'test'})
    try:
        assert workflow.read_owned('tunnel')['pid'] == state['pid']
        assert 'sleep 30' in state['identity']
        workflow.stop_owned('tunnel')
        assert workflow.read_owned('tunnel') is None
        assert workflow.process_identity(state['pid']) == ''
    finally:
        # Reap the test child if this test process still owns it.
        try:
            workflow.os.waitpid(state['pid'], 0)
        except ChildProcessError:
            pass


def test_expanso_stop_failure_does_not_skip_gpu_stop(monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, 'stop_owned', lambda _: None)

    def fail(args, **_):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(workflow, 'run', fail)
    monkeypatch.setattr(workflow, 'down', calls.append)
    with pytest.raises(ValueError):
        workflow.demo_down('new')
    assert calls == ['new']


@pytest.mark.parametrize('receipt', [{}, [], {'pid': 0, 'identity': 'x'}, {'pid': 42}])
def test_invalid_receipt_cannot_skip_gpu_shutdown(receipt, tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'RUNTIME', tmp_path)
    (tmp_path / 'presenter.json').write_text(json.dumps(receipt))
    calls = []
    monkeypatch.setattr(workflow, 'run', lambda *a, **kw: None)
    monkeypatch.setattr(workflow, 'down', calls.append)
    monkeypatch.setattr(workflow.os, 'killpg', lambda *_: pytest.fail('must not signal'))
    with pytest.raises(ValueError, match='receipt is invalid'):
        workflow.demo_down('new')
    assert calls == ['new']


@pytest.mark.parametrize('strategy,replaced', [
    ('--safetensors-load-strategy prefetch', False),
    ('--safetensors-load-strategy=prefetch', False),
    ('', True),
    ('--safetensors-load-strategy lazy', True),
    ('--safetensors-load-strategy=prefetch --safetensors-load-strategy lazy', True),
])
@pytest.mark.parametrize('running', ['true', 'false'])
def test_setup_reconciles_strategy_and_reuses_matching_caches(tmp_path, strategy, replaced, running):
    import os

    fake = tmp_path / 'bin'
    fake.mkdir()
    calls = tmp_path / 'calls'
    docker = fake / 'docker'
    docker.write_text('''#!/bin/bash
printf '%s\\n' "$*" >> "$CALLS"
case "$*" in
  *Config.Image*) echo vllm/vllm-openai:v0.28.0 ;;
  *Config.Cmd*director) printf '%s\\n' --model RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8 $STRATEGY ;;
  *Config.Cmd*vlm) printf '%s\\n' --model HuggingFaceTB/SmolVLM-500M-Instruct ;;
  *State.Running*) echo "$RUNNING" ;;
esac
''')
    docker.chmod(0o755)
    for name in ['nvidia-smi', 'uv', 'curl']:
        tool = fake / name
        tool.write_text('#!/bin/sh\nexit 0\n')
        tool.chmod(0o755)
    result = subprocess.run(['bash', str(ROOT / 'brev/setup.sh')],
                            env={**os.environ, 'PATH': f'{fake}:{os.environ["PATH"]}',
                                 'CALLS': str(calls), 'STRATEGY': strategy, 'RUNNING': running},
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    operations = calls.read_text().splitlines()
    if replaced:
        assert operations.index('pull vllm/vllm-openai:v0.28.0') < operations.index('container stop --timeout 30 director')
        assert 'container rm director' in operations
        starts = [line for line in operations if line.startswith('run ')]
        assert len(starts) == 2
        assert '--name vlm' in starts[0]
        assert '--name director' in starts[1]
        assert '--safetensors-load-strategy prefetch' in starts[1]
        assert all('/.cache/huggingface:/root/.cache/huggingface' in line for line in starts)
        assert not any('prune' in line or 'volume rm' in line for line in operations)
    else:
        if running == 'false':
            assert operations.index('start vlm') < operations.index('start director')
        else:
            assert not any(line.startswith('start ') for line in operations)
        assert not any(line.startswith(('container rm ', 'container stop ', 'pull ', 'run ')) for line in operations)
