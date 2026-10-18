"""Check disconnected-board motion with an owned headless agent-browser session."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'sherlock/sherlock/dashboard/static'


class BoardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = '/pipeline.html'
        elif self.path.startswith('/pipeline/assets/'):
            self.path = self.path.removeprefix('/pipeline/assets')
        super().do_GET()

    def log_message(self, *_):
        pass


def probe(output):
    output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(BoardHandler, directory=str(STATIC)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    results = {}
    try:
        for reduced in (False, True):
            name = f'sherlock-motion-{os.getpid()}-{int(reduced)}'

            def browser(*args, session=name):
                subprocess.run(['agent-browser', '--session', session, *args],
                               check=True, capture_output=True, text=True, timeout=60)

            try:
                browser('set', 'viewport', '1920', '1080')
                if reduced:
                    browser('set', 'media', 'light', 'reduced-motion')
                browser('open', f'http://127.0.0.1:{server.server_port}')
                time.sleep(2.5)
                frames = []
                for index in range(4):
                    if index:
                        time.sleep(1.2)
                    path = output / f'{name}-{index}.png'
                    browser('screenshot', str(path))
                    frames.append(Image.open(path).convert('RGB'))
                changed = sum(sum(1 for pixel in ImageChops.difference(a, b).getdata() if max(pixel) > 12)
                              for a, b in pairwise(frames))
                minimum = 300 if reduced else 800
                results['reduced' if reduced else 'normal'] = {
                    'changed_pixels': changed, 'minimum': minimum, 'passed': changed >= minimum}
            finally:
                browser('close')
    finally:
        server.shutdown()
        server.server_close()
    (output / 'motion.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))
    return all(result['passed'] for result in results.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.runtime/motion-check')
    options = parser.parse_args()
    raise SystemExit(0 if probe(options.output) else 1)
