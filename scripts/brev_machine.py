"""Mac entry point for an existing Brev machine and owned, foreground SSH tunnel."""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.runtime' / 'brev'
DIRECTOR = 'RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8'
VLM = 'HuggingFaceTB/SmolVLM-500M-Instruct'
PORTS = (18001, 18002)
TRANSPORT_PORT = 23121


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def instance_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,62}', value):
        raise ValueError('Use an exact Brev instance name, without shell syntax')
    return value


def inventory():
    result = run(['brev', 'ls', '--json'], capture_output=True, text=True)
    return json.loads(result.stdout)['workspaces']


def find_instance(name):
    matches = [item for item in inventory() if item['name'] == name]
    if len(matches) != 1:
        raise ValueError(f'Expected one existing instance named {name}; run: brev ls')
    return matches[0]


def ssh(name, *args):
    return ['ssh', '-T', '-o', 'ControlMaster=no', '-o', 'ControlPath=none',
            '-o', 'ForkAfterAuthentication=no', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3', *args, name]


def model_env(name):
    result = {'BREV_HOST': name, 'SHERLOCK_MOCK': '0', 'DIRECTOR_THINKING': '0',
              'VERIFIER_MARGIN': '-1.0', 'DASHBOARD_HOST': '127.0.0.1'}
    for role, port, model in [('DIRECTOR', PORTS[0], DIRECTOR),
                              ('ANALYST', PORTS[1], VLM), ('VERIFIER', PORTS[1], VLM)]:
        result.update({f'{role}_BASE_URL': f'http://127.0.0.1:{port}/v1',
                       f'{role}_MODEL': model, f'{role}_API_KEY': 'EMPTY'})
    return result


def smoke(name):
    run(['uv', 'run', 'scripts/brev_preflight.py', '--mode', 'smoke',
         '--ready-timeout', '1800', '--instance', name, '--no-instructions',
         '--output-dir', str(RUNTIME / 'checks')],
        cwd=ROOT, env={**os.environ, **model_env(name)})


def require_free_ports(ports):
    for port in ports:
        with socket.socket() as listener:
            # A stopped presenter can leave TIME_WAIT sockets for a minute.
            # Match the server's reuse behavior while still rejecting listeners.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(('127.0.0.1', port))
            except OSError as exc:
                raise ValueError(f'Local port {port} is occupied; no process was stopped') from exc


def wait_for_ssh(name, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = subprocess.run([*ssh(name), 'true'], check=False,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return
        print(f'Waiting for SSH on {name}...', flush=True)
        time.sleep(5)
    raise ValueError('Brev did not become SSH-ready within ten minutes')


def down(name, timeout=180):
    """Stop only the named GPU and require a terminal stopped-state readback."""
    item = find_instance(name)
    stopped = {'STOPPED', 'PAUSED', 'OFF'}
    if item['status'] in stopped:
        print(f"{name}: {item['status']}; already stopped.")
        return
    if item['status'] != 'STOPPING':
        try:
            run(['brev', 'stop', name])
        except subprocess.CalledProcessError as exc:
            raise ValueError(f'Brev could not stop {name}; GPU billing may continue. '
                             'Some providers do not support stop. No machine was deleted.') from exc
    deadline = time.monotonic() + timeout
    while True:
        observed = find_instance(name)
        if observed['id'] != item['id']:
            raise ValueError('Instance identity changed while stopping; inspect brev ls')
        if observed['status'] in stopped:
            print(f"{name}: {observed['status']}. GPU stopped; storage may still incur charges.")
            return
        if time.monotonic() >= deadline:
            raise ValueError(f"Stop not confirmed ({observed['status']}); check brev ls --json")
        print(f"Waiting for {name} to stop ({observed['status']})...", flush=True)
        time.sleep(3)


def up(name):
    item = find_instance(name)
    if 'H100' not in item.get('gpu', '').upper():
        raise ValueError('This setup targets an existing H100; refusing a different GPU')
    if item['status'] in {'STOPPED', 'STOPPING', 'PAUSED', 'OFF'}:
        if item['status'] == 'STOPPING':
            raise ValueError('Instance is still stopping; retry after it has stopped')
        run(['brev', 'start', name])
    elif item['status'] != 'RUNNING':
        raise ValueError(f"Instance is {item['status']}; wait for Brev to finish")
    run(['brev', 'refresh'])
    wait_for_ssh(name)
    # Only committed, selected source files travel. No .env, keys, camera files or results.
    revision = run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                   capture_output=True, text=True).stdout.strip()
    if run(['git', 'status', '--porcelain', '--', 'brev', 'scripts', 'sherlock', 'jobs', 'edge',
            'pyproject.toml', 'uv.lock', 'tests/fixtures/vllm'], cwd=ROOT,
           capture_output=True, text=True).stdout:
        raise ValueError('Commit setup source changes before uploading a release')
    relative = f'code/second-brain/projects/demos/demo-sherlock/releases/{revision}'
    remote = f'"$HOME"/{shlex.quote(relative)}'
    archive = run(['git', 'archive', revision, 'brev', 'scripts', 'sherlock', 'jobs', 'edge',
                   'pyproject.toml', 'uv.lock', 'tests/fixtures/vllm'],
                  cwd=ROOT, capture_output=True).stdout
    run([*ssh(name), f'mkdir -p {remote} && tar -xf - -C {remote}'], input=archive)
    run([*ssh(name), f'cd {remote} && bash brev/setup.sh'])
    print(f'GPU setup verified. Next: just brev-connect {name}', flush=True)


def connect(name):
    item = find_instance(name)
    if item['status'] != 'RUNNING':
        raise ValueError('Instance is not running; use just brev-up first')
    require_free_ports((*PORTS, TRANSPORT_PORT))
    run(['brev', 'refresh'])
    RUNTIME.mkdir(parents=True, exist_ok=True)
    receipt = RUNTIME / 'connection.json'
    command = ssh(name, '-N', '-o', 'ExitOnForwardFailure=yes',
                  '-L', f'127.0.0.1:{PORTS[0]}:127.0.0.1:8001',
                  '-L', f'127.0.0.1:{PORTS[1]}:127.0.0.1:8002',
                  '-L', '127.0.0.1:23121:127.0.0.1:23121',
                  '-R', '127.0.0.1:23100:127.0.0.1:8000')
    with subprocess.Popen(command) as tunnel:
        try:
            time.sleep(1)
            if tunnel.poll() is not None:
                raise ValueError('SSH tunnel exited; no local settings changed')
            smoke(name)
            if tunnel.poll() is not None:
                raise ValueError('SSH tunnel exited during verification')
            state = {'name': name, 'id': item['id'], 'pid': tunnel.pid,
                     'verified_at': time.time(), 'env': model_env(name)}
            temp = receipt.with_suffix('.tmp')
            temp.write_text(json.dumps(state, indent=2) + '\n')
            temp.replace(receipt)
            print('FP8 and VLM smoke passed through the Mac tunnel.\n'
                  'Keep this terminal open. In another: just brev-live\n'
                  'Ctrl-C closes only this tunnel. Your .env is unchanged.', flush=True)
            tunnel.wait()
        finally:
            if tunnel.poll() is None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
                    tunnel.wait()
            if receipt.exists() and json.loads(receipt.read_text()).get('pid') == tunnel.pid:
                receipt.unlink()


def connection():
    path = RUNTIME / 'connection.json'
    if not path.exists():
        raise ValueError('No verified tunnel; run just brev-connect in another terminal')
    state = json.loads(path.read_text())
    instance_name(state['name'])
    os.kill(state['pid'], 0)
    process = run(['ps', '-p', str(state['pid']), '-o', 'command='],
                  capture_output=True, text=True).stdout.strip()
    tokens = shlex.split(process)
    if (not tokens or Path(tokens[0]).name != 'ssh' or '-N' not in tokens
            or tokens[-1] != state['name']
            or f'127.0.0.1:{PORTS[0]}:127.0.0.1:8001' not in tokens
            or f'127.0.0.1:{PORTS[1]}:127.0.0.1:8002' not in tokens):
        raise ValueError('Saved tunnel PID no longer belongs to this connection')
    return state


def live(args):
    state = connection()
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--port', type=int, default=int(os.environ.get('DASHBOARD_PORT', '8000')))
    parsed, _ = options.parse_known_args(args)
    require_free_ports([parsed.port])
    smoke(state['name'])
    env = {**os.environ, **model_env(state['name'])}
    if '--mock' in args:
        raise ValueError('brev-live requires the real model endpoints')
    pidfile = ROOT / '.runtime' / 'sherlock.pid'
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(['uv', 'run', '--extra', 'yolo', 'python', 'run.py', *args],
                          cwd=ROOT, env=env) as presenter:
        pidfile.write_text(f'{presenter.pid}\n')
        try:
            result = presenter.wait()
            if result:
                raise subprocess.CalledProcessError(result, 'run.py')
        finally:
            if pidfile.exists() and pidfile.read_text().strip() == str(presenter.pid):
                pidfile.unlink()


def retire(old, replacement):
    previous, current = find_instance(old), find_instance(replacement)
    if previous['id'] == current['id']:
        raise ValueError('Old and replacement must be different machines')
    state = connection()
    if state['id'] != current['id'] or state['name'] != replacement:
        raise ValueError('Connect and verify the replacement before retiring the old machine')
    smoke(replacement)
    if connection() != state or find_instance(replacement)['status'] != 'RUNNING':
        raise ValueError('Replacement connection changed during verification')
    run(['brev', 'stop', old])
    observed = find_instance(old)
    print(f"{old}: {observed['status']}. No instance was deleted.")
    if observed['status'] not in {'STOPPED', 'PAUSED', 'OFF'}:
        raise ValueError('Stop requested but not confirmed; check: brev ls --json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('up', 'connect', 'down'):
        child = sub.add_parser(command)
        child.add_argument('instance', nargs='?', default='sherlock-h100', type=instance_name)
    sub.add_parser('status')
    child = sub.add_parser('live')
    child.add_argument('args', nargs=argparse.REMAINDER)
    child = sub.add_parser('retire')
    child.add_argument('old', type=instance_name)
    child.add_argument('replacement', type=instance_name)
    args = parser.parse_args()
    try:
        if args.command == 'status':
            print(json.dumps(inventory(), indent=2))
        elif args.command == 'up':
            up(args.instance)
        elif args.command == 'down':
            down(args.instance)
        elif args.command == 'connect':
            connect(args.instance)
        elif args.command == 'live':
            live(args.args[1:] if args.args[:1] == ['--'] else args.args)
        else:
            retire(args.old, args.replacement)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Sherlock Brev: {exc}\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Brev connection closed.\n')


if __name__ == '__main__':
    main()
