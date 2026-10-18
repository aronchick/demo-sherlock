"""Own Expanso processes by exact command, start time and process group."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time


def identity(pid: int) -> str:
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'lstart=', '-o', 'command='],
                            capture_output=True, text=True, check=False)
    text = result.stdout.strip()
    if result.returncode or not text or text.split()[0].startswith('Z'):
        return ''
    return text.split(maxsplit=1)[1]


def command(pid: int) -> list[str]:
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                            capture_output=True, text=True, check=False)
    return shlex.split(result.stdout.strip()) if result.returncode == 0 else []


def owned(root: pathlib.Path, role: str) -> dict | None:
    folder = root / '.runtime/expanso'
    path = folder / f'{role}.process.json'
    if path.exists():
        state = json.loads(path.read_text())
        pid = state.get('pid')
        if type(pid) is not int or pid <= 1 or not state.get('identity'):
            raise ValueError('invalid ownership receipt')
        current = identity(pid)
        if not current:
            path.unlink()
            (folder / f'{role}.pid').unlink(missing_ok=True)
            return None
        if current != state['identity'] or os.getpgid(pid) != pid:
            raise ValueError(f'{role} process identity changed; no process touched')
        return state
    legacy = folder / f'{role}.pid'
    if not legacy.exists():
        return None
    pid = int(legacy.read_text().strip())
    if pid <= 1:
        raise ValueError('invalid legacy PID')
    current = identity(pid)
    if not current:
        legacy.unlink()
        return None
    tokens = command(pid)
    # Legacy nodes carried an absolute, project-specific data directory; legacy
    # watchers carry the absolute core directory. Require a detached session too.
    if role in {'edge', 'core'}:
        expected = ['--data-dir', str(root / '.cloud-state' / f'sherlock-{role}')]
        match = (tokens and pathlib.Path(tokens[0]).name == 'expanso-edge'
                 and any(tokens[i:i + 2] == expected for i in range(len(tokens)))
                 and f'edge/sherlock-{role}.yaml' in tokens)
    else:
        expected = ['--core-dir', str(root / '.runtime/expanso/core')]
        match = ('scripts/core_receipts.py' in tokens
                 and any(tokens[i:i + 2] == expected for i in range(len(tokens))))
    if not match or os.getpgid(pid) != pid or identity(pid) != current:
        raise ValueError(f'{role} legacy PID cannot be proven project-owned; no process touched')
    state = {'pid': pid, 'identity': current, 'fingerprint': 'legacy'}
    path.write_text(json.dumps(state))
    return state


def stop(root: pathlib.Path, role: str) -> None:
    state = owned(root, role)
    if state is None:
        return
    pid = state['pid']
    if identity(pid) != state['identity'] or os.getpgid(pid) != pid:
        raise ValueError('identity changed before stop')
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while identity(pid) == state['identity']:
        if time.monotonic() >= deadline:
            if os.getpgid(pid) == pid and identity(pid) == state['identity']:
                os.killpg(pid, signal.SIGKILL)
            break
        time.sleep(.1)
    folder = root / '.runtime/expanso'
    (folder / f'{role}.process.json').unlink(missing_ok=True)
    (folder / f'{role}.pid').unlink(missing_ok=True)


def ensure(root: pathlib.Path, role: str, args: list[str], files: list[pathlib.Path]) -> None:
    digest = hashlib.sha256(json.dumps(args).encode())
    environment = {name: value for name, value in os.environ.items()
                   if name.startswith('SHERLOCK_') or name == 'EDGE_NODE_NAME'}
    digest.update(json.dumps(environment, sort_keys=True).encode())
    for path in files:
        digest.update(path.read_bytes())
    fingerprint = digest.hexdigest()
    state = owned(root, role)
    if state and state.get('fingerprint') == fingerprint:
        print(f'{role}: reusing owned process')
        return
    if state:
        stop(root, role)
    folder = root / '.runtime/expanso'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f'{role}.log').open('ab') as log:
        process = subprocess.Popen(args, cwd=root, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=log, start_new_session=True)
    time.sleep(.2)
    observed = identity(process.pid)
    if process.poll() is not None or not observed:
        raise ValueError(f'{role} exited during startup; inspect its log')
    state = {'pid': process.pid, 'identity': observed, 'fingerprint': fingerprint}
    (folder / f'{role}.process.json').write_text(json.dumps(state))
    (folder / f'{role}.pid').write_text(str(process.pid) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['status', 'stop', 'ensure'])
    parser.add_argument('role', choices=['edge', 'core', 'receipts'])
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--fingerprint-file', type=pathlib.Path, action='append', default=[])
    args, command_args = parser.parse_known_args()
    root = args.root.resolve()
    try:
        if args.action == 'status':
            raise SystemExit(0 if owned(root, args.role) else 1)
        if args.action == 'stop':
            stop(root, args.role)
        else:
            if command_args[:1] == ['--']:
                command_args = command_args[1:]
            if not command_args:
                raise ValueError('ensure requires an executable command')
            ensure(root, args.role, command_args, args.fingerprint_file)
    except (OSError, ValueError) as error:
        print(f'Expanso ownership: {error}', file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == '__main__':
    main()
