// @ts-check
'use strict';

/** Restore saved state quietly; any arriving event takes precedence. */
class IncidentHydration {
  /** @param {(record:any) => void} render */
  constructor(render) {
    this.render = render;
    this.generation = 0;
  }

  live() {
    this.generation++;
  }

  /** @param {typeof fetch} [request] */
  async load(request = fetch) {
    const generation = this.generation;

    try {
      const response = await request('/incidents?limit=1', {cache: 'no-store'});

      if (!response.ok) return;

      const records = await response.json();

      if (generation !== this.generation || !Array.isArray(records) || !records[0]) return;

      this.render(records[0]);
    } catch {
      // Existing live state remains authoritative when persisted state is unavailable.
    }
  }
}

/** @param {any} record @param {ModelCards} cards
 * @param {(role:string, value:string, detail:string) => void} cell
 */
function hydrateIncident(record, cards, cell) {
  const artifacts = record.artifacts || [];
  const scene = artifacts.find(/** @param {any} item */ item => item.name === 'original_frame' && item.integrity === 'verified');
  const crop = artifacts.find(/** @param {any} item */ item => ['weapon_crop', 'object_crop'].includes(item.name) && item.integrity === 'verified');
  const compared = !!record.comparison;
  const verdict = record.verified === true ? 'Confirmed' : record.verified === false ? 'Not confirmed' : 'Not requested';
  const timestamp = new Date(record.t * 1000).toLocaleTimeString();
  const origin = document.querySelector('#record .lede small');

  if (origin) origin.textContent = `Saved incident · ${timestamp}`;

  cell('scout', record.trigger?.label || record.label || 'Not recorded', 'Saved detection');
  cell('director', record.claim || 'Not recorded', record.action || '');
  cell('analyst', record.analyst_mismatch ? 'Description withheld' : record.description || 'Not requested', compared ? 'Saved assessment' : '');
  cell('verifier', record.verifier_reason || verdict, compared ? 'Saved assessment' : '');
  cell('decision', record.state === 'investigating' ? 'Saved incomplete investigation' : record.action === 'alert' ? 'ALERT, evidence kept' : record.action === 'ignore' ? 'Archived' : 'Review, evidence kept', Number.isFinite(record.total_ms) ? `${(record.total_ms / 1000).toFixed(1)} s · saved result` : 'Saved result');
  cards.restore('director', scene?.url, 'Saved scene · ticket context');
  cards.restore('analyst', compared ? crop?.url : null, compared ? 'Saved object crop' : 'Not requested');
  cards.restore('verifier', compared ? crop?.url : null, compared ? 'Saved object crop' : 'Not requested');
  const status = document.getElementById('comparison-status');
  const reason = document.getElementById('comparison-reason');

  if (status) status.textContent = compared ? `Saved decision: ${verdict.toLowerCase()}` : 'No model comparison requested';

  if (reason) reason.textContent = record.comparison?.reason || 'Archived without a vision check';
}
