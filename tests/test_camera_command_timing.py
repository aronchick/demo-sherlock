"""Only an attributable successful device command yields a camera latency."""
import json
import pathlib
import shutil
import subprocess

import pytest

STATIC = pathlib.Path(__file__).resolve().parents[1] / 'sherlock/sherlock/dashboard/static'
pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='requires node')


def test_measured_command_completion_and_missing_or_ambiguous_events():
    script = r'''
const fs = require('node:fs'), vm = require('node:vm');
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8') + ';globalThis.Timing = CameraCommandTiming;');
const outputs = [];
const timing = new Timing(seconds => outputs.push(seconds));
const scout = (t, camera = 'hero', extra = {}) => timing.receive({stage:'scout', t,
  data:{camera, label:'knife', bbox:[0,0,1,1], ...extra}});
const tool = (t, incident, extra = {}) => timing.receive({stage:'tool', t,
  data:{incident, camera:'hero', label:'knife', tool:'look_at', result:{ok:true}, ...extra}});
scout(100);
timing.receive({stage:'narration', t:101, data:{incident:'one', camera:'hero', label:'knife'}});
tool(101.2, 'one', {error:true});
tool(101.3, 'one', {result:{}});
tool(101.4, 'one', {tool:'snapshot'});
tool(101.5, 'one');
tool(105, 'one'); // Only first success, never the second move.
scout(110);
tool(111, 'two', {error:true}); // New investigation clears prior result.
tool(112, 'two', {result:undefined}); // Corruption narration is not completion.
scout(120);
tool(119, 'three'); // Reject clock reversal.
tool(NaN, 'three');
tool(121, 'three', {tool:'move'});
tool(130, 'missing'); // No scout means no measurement.
scout(140);
scout(141); // Duplicate/unattributed scout events: no guessing.
tool(142, 'ambiguous');
scout(150, 'scout1');
tool(151, 'wrong-camera');
scout(160, 'hero', {handled_via:'scout1'});
tool(161, 'duplicate');
console.log(JSON.stringify(outputs));
'''
    result = subprocess.run(['node', '-e', script, str(STATIC / 'pipeline-models.js')],
                            capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(result.stdout) == [None, 1.5, None, None, 1, None, None, None, None]


def test_timing_is_wired_before_animated_stage_dispatch():
    html = (STATIC / 'pipeline.html').read_text()
    handler = html[html.index('es.onmessage ='):html.index('hydration.load();')]
    assert handler.index('cameraTiming.receive(ev)') < handler.index('switch (ev.stage)')
    assert 'First camera command:' in html
    assert 'Camera settling and the full investigation are separate' in html
