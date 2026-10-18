"""Provision the persistent Expanso Cloud core beside the Brev model servers."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from brev_machine import ROOT, find_instance, instance_name, run, ssh

REMOTE = 'code/second-brain/projects/demos/demo-sherlock/expanso-core'
VERSION = 'v2.1.23'


def bootstrap_token() -> str:
    """Read only the bootstrap token; never forward Cloud administrator keys."""
    override = os.environ.get('EXPANSO_WORKSPACE_ENV')
    paths = [Path(override)] if override else [ROOT.parent / '_demo-kit/.env', ROOT / '.env']
    for path in paths:
        if not path.is_absolute():
            raise ValueError('EXPANSO_WORKSPACE_ENV must be an absolute path')
        if not path.exists():
            continue
        if path.stat().st_mode & 0o077:
            raise ValueError(f'Bootstrap credentials must be owner-only: {path}')
        for line in path.read_text().splitlines():
            key, sep, value = line.partition('=')
            if sep and key.strip() == 'EXPANSO_EDGE_BOOTSTRAP_TOKEN':
                token = value.strip().strip('\"').strip("'")
                if token.startswith('exp_bk_') and not any(c.isspace() for c in token):
                    return token
    raise ValueError('No project-scoped Expanso bootstrap token is configured')


def core_up(name: str):
    item = find_instance(name)
    if item['status'] != 'RUNNING':
        raise ValueError('Start the Brev machine before provisioning its Expanso core')
    remote = f'"$HOME"/{REMOTE}'
    # The core directory survives code releases and GPU stop/start.
    archive = run(['git', 'archive', 'HEAD', 'scripts/expanso.sh', 'scripts/core_receipts.py', 'scripts/expanso_process.py', 'edge', 'jobs'],
                  cwd=ROOT, capture_output=True).stdout
    run([*ssh(name), f'umask 077; mkdir -p {remote}; tar -xf - -C {remote}'], input=archive)
    installed = subprocess.run([*ssh(name), f'test -x {remote}/bin/expanso-edge'],
                               check=False, capture_output=True)
    if installed.returncode:
        run([*ssh(name), (f'cd {remote} && mkdir -p bin .runtime && '
             'curl -fsSL https://get.expanso.io/edge/install.sh -o .runtime/install-edge.sh && '
             f'EXPANSO_VERSION={VERSION} EXPANSO_INSTALL_DIR="$PWD/bin" '
             'EXPANSO_DISABLEANALYTICS=true bash .runtime/install-edge.sh')])
    run([*ssh(name), (f'set -e; cd {remote}; export PATH="$PWD/bin:$HOME/.local/bin:$PATH"; '
         'if ! command -v uv >/dev/null; then '
         'curl -fsSL https://astral.sh/uv/0.12.17/install.sh -o .runtime/install-uv.sh; '
         'UV_INSTALL_DIR="$PWD/bin" UV_NO_MODIFY_PATH=1 sh .runtime/install-uv.sh; fi')])
    registered = subprocess.run(
        [*ssh(name), f'test -f {remote}/.cloud-state/sherlock-core/auth/credentials.creds'],
        check=False, capture_output=True)
    token = '' if registered.returncode == 0 else bootstrap_token()
    # stdin keeps the short-lived bootstrap credential out of argv and logs.
    command = (f'set -e; cd {remote} && umask 077 && mkdir -p .runtime && '
               'cat > .runtime/bootstrap.env && '
               'export PATH="$PWD/bin:$HOME/.local/bin:$PATH" && '
               'export EXPANSO_WORKSPACE_ENV="$PWD/.runtime/bootstrap.env" && '
               'export SHERLOCK_CORE_DIR="$PWD/.runtime/expanso/core" && '
               'trap \'rm -f .runtime/bootstrap.env\' EXIT; '
               'bash scripts/expanso.sh core')
    payload = f'EXPANSO_EDGE_BOOTSTRAP_TOKEN={token}\n' if token else ''
    run([*ssh(name), command], input=payload.encode())
    print(f'Expanso core started on {name}; verify Cloud execution and downstream receipt separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['up'])
    parser.add_argument('instance', type=instance_name)
    args = parser.parse_args()
    try:
        core_up(args.instance)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'Expanso core: {exc}\n')


if __name__ == '__main__':
    main()
