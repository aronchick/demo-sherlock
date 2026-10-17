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
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
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
         '--ready-timeout', '1800', '--instance', name,
         '--output-dir', str(RUNTIME / 'checks')],
        cwd=ROOT, env={**os.environ, **model_env(name)})


def require_free_ports(ports):
    for port in ports:
        with socket.socket() as listener:
            try:
                listener.bind(('127.0.0.1', port))
            except OSError as exc:
                raise ValueError(f'Local port {port} is occupied; no process was stopped') from exc


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
    # Only committed, selected source files travel. No .env, keys, camera files or results.
    revision = run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                   capture_output=True, text=True).stdout.strip()
    if run(['git', 'status', '--porcelain', '--', 'brev', 'scripts', 'sherlock',
            'pyproject.toml', 'uv.lock', 'tests/fixtures/vllm'], cwd=ROOT,
           capture_output=True, text=True).stdout:
        raise ValueError('Commit setup source changes before uploading a release')
    relative = f'code/second-brain/projects/demos/demo-sherlock/releases/{revision}'
    remote = f'"$HOME"/{shlex.quote(relative)}'
    archive = run(['git', 'archive', revision, 'brev', 'scripts', 'sherlock',
                   'pyproject.toml', 'uv.lock', 'tests/fixtures/vllm'],
                  cwd=ROOT, capture_output=True).stdout
    run([*ssh(name), f'mkdir -p {remote} && tar -xf - -C {remote}'], input=archive)
    run([*ssh(name), f'cd {remote} && bash brev/setup.sh'])
    print(f'GPU setup verified. Next: just brev-connect {name}', flush=True)


def connect(name):
    item = find_instance(name)
    if item['status'] != 'RUNNING':
        raise ValueError('Instance is not running; use just brev-up first')
    require_free_ports(PORTS)
    run(['brev', 'refresh'])
    RUNTIME.mkdir(parents=True, exist_ok=True)
    receipt = RUNTIME / 'connection.json'
    command = ssh(name, '-N', '-o', 'ExitOnForwardFailure=yes',
                  '-L', f'127.0.0.1:{PORTS[0]}:127.0.0.1:8001',
                  '-L', f'127.0.0.1:{PORTS[1]}:127.0.0.1:8002')
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
    require_free_ports([int(os.environ.get('DASHBOARD_PORT', '8000'))])
    smoke(state['name'])
    env = {**os.environ, **model_env(state['name'])}
    if '--mock' in args:
        raise ValueError('brev-live requires the real model endpoints')
    run(['uv', 'run', '--extra', 'yolo', 'python', 'run.py', *args], cwd=ROOT, env=env)


def retire(old, replacement):
    previous, current = find_instance(old), find_instance(replacement)
    if previous['id'] == current['id']:
        raise ValueError('Old and replacement must be different machines')
    state = connection()
    if state['id'] != current['id'] or state['name'] != replacement:
        raise ValueError('Connect and verify the replacement before retiring the old machine')
    smoke(replacement)
    run(['brev', 'stop', old])
    observed = find_instance(old)
    print(f"{old}: {observed['status']}. No instance was deleted.")
    if observed['status'] not in {'STOPPED', 'PAUSED', 'OFF'}:
        raise ValueError('Stop requested but not confirmed; check: brev ls --json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('up', 'connect'):
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
