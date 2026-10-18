"""Saved incidents restore static presentation without replaying stages or counters."""
import json
import pathlib
import shutil
import subprocess

import pytest

STATIC = pathlib.Path(__file__).resolve().parents[1] / 'sherlock/sherlock/dashboard/static'
pytestmark = pytest.mark.skipif(shutil.which('node') is None, reason='requires node')


def test_snapshot_races_and_quiet_render():
    script = r'''
const fs = require('node:fs'), vm = require('node:vm');
vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8') + ';globalThis.Hydration = IncidentHydration;');
(async () => {
  const renders = [];
  const hydration = new Hydration(record => renders.push(record.incident));
  await hydration.load(async () => ({ok: true, json: async () => [{incident: 'first'}]}));
  let response;
  const pending = hydration.load(() => new Promise(resolve => { response = resolve; }));
  hydration.live();
  response({ok: true, json: async () => [{incident: 'stale'}]});
  await pending;
  await hydration.load(async () => { throw new Error('offline'); });
  await hydration.load(async () => ({ok: true, json: async () => []}));
  let finish;
  const decoding = hydration.load(async () => ({ok: true, json: () => new Promise(resolve => { finish = resolve; })}));
  await Promise.resolve();
  hydration.live();
  finish([{incident: 'stale-during-json'}]);
  await decoding;
  const elements = {};
  const get = id => elements[id] ??= {textContent: ''};
  globalThis.document = {querySelector: get, getElementById: get};
  const cells = [], previews = [];
  hydrateIncident({incident: 'first', t: 1, action: 'alert', verified: true, total_ms: 2200,
    trigger: {label: 'knife'}, claim: 'a knife', description: 'steel knife',
    comparison: {reason: 'confirmed'}, artifacts: [
      {name:'original_frame', integrity:'verified', url:'/scene'},
      {name:'weapon_crop', integrity:'verified', url:'/crop'},
      {name:'face_crop', integrity:'verified', url:'/face'}]},
    {restore: (...args) => previews.push(args)}, (...args) => cells.push(args));
  console.log(JSON.stringify({renders, cells, previews, elements}));
})().catch(error => { console.error(error); process.exit(1); });
'''
    result = subprocess.run(['node', '-e', script, str(STATIC / 'board-hydration.js')],
                            capture_output=True, text=True, check=True, timeout=10)
    output = json.loads(result.stdout)
    assert output['renders'] == ['first']
    assert len(output['cells']) == 5
    assert output['cells'][-1] == ['decision', 'ALERT, evidence kept', '2.2 s · saved result']
    assert output['previews'] == [['director', '/scene', 'Saved scene · ticket context'],
                                  ['analyst', '/crop', 'Saved object crop'],
                                  ['verifier', '/crop', 'Saved object crop']]
    assert '/face' not in json.dumps(output)
    assert output['elements']['#record .lede small']['textContent'].startswith('Saved incident')
