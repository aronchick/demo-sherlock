// @ts-check
'use strict';

/** @typedef {{name:string, sha256:string, url:string, integrity:string, camera?:string,captured_at?:number|null,source_artifact?:string|null,relationship?:string,sources?:string[],missing_sources?:string[],bbox?:number[]|null}} Artifact */
/** @typedef {{stage:string, t:number, detail:string}} CustodyEvent */
/** @typedef {{incident:string, action:string, state?:string, claim:string, t:number, verified?:boolean, comparison?:{status:string,reason:string,claim:string,verifier_supports:boolean|null,analyst_names_claim?:boolean|null}, evidence_camera?:string, trigger?:{camera:string}, artifacts:Artifact[], custody:CustodyEvent[], models?:{mock?:boolean,scout?:string,director?:string,analyst?:string,verifier?:string}, notification:{simulated:boolean,status:string,delivered_at?:number,acknowledged_at?:number}}} Incident */

(() => {
  const phoneError = document.getElementById('phone-error');
  const shelf = document.getElementById('evidence-links');
  const cloudShelf = document.getElementById('cloud-evidence-links');
  const cloudStatus = document.getElementById('cloud-evidence-status');
  const custody = document.getElementById('custody-content');
  const custodyError = document.getElementById('custody-error');
  const selected = new URLSearchParams(location.search).get('incident');
  const isPhone = location.pathname === '/phone';
  const isCustody = location.pathname === '/custody';
  window.name = isPhone ? 'sherlock-phone' : isCustody ? 'sherlock-evidence' : 'sherlock-flow';
  let page = 0;
  let shelfRevision = '';
  const pageSize = 8;

  /** @param {string} tag @param {string} [text] @param {string} [className] */
  function element(tag, text = '', className = '') {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = className;

    return node;
  }

  /** @param {number|undefined|null} value */
  function timestamp(value) {
    return value == null ? 'Not recorded' : new Date(value * 1000).toLocaleString();
  }

  /** @param {string} url @param {RequestInit} [options] @returns {Promise<any>} */
  async function read(url, options) {
    const response = await fetch(url, { ...options, cache: 'no-store' });

    if (!response.ok) throw new Error(`Request failed (${response.status})`);

    return response.json();
  }

  /** @param {string} incident */
  function custodyUrl(incident) {
    return `/custody?incident=${encodeURIComponent(incident)}`;
  }

  /** @param {Incident} record @param {Artifact} artifact */
  function pictureLink(record, artifact) {
    const link = document.createElement('a');
    link.href = custodyUrl(record.incident);
    link.target = 'sherlock-evidence';
    link.title = `Chain of custody: ${artifact.name}, ${record.incident}`;

    if (artifact.integrity === 'missing') {
      link.textContent = 'Image unavailable · view record';

      return link;
    }

    const img = document.createElement('img');
    img.src = artifact.url;
    img.alt = `${artifact.name.replaceAll('_', ' ')} from ${record.incident}. View chain of custody`;

    if (artifact.name === 'face_crop') img.className = 'face-preview';
    link.append(img);

    return link;
  }

  /** @param {Incident|undefined} record */
  function renderPhone(record) {
    const indicator = document.getElementById('phone-indicator');
    const summary = document.getElementById('phone-summary');

    if (indicator && summary) {
      indicator.dataset.status = record?.notification.status || 'waiting';
      summary.textContent = record ? (record.notification.status === 'acknowledged' ? 'Investigating' : 'Alert received') : 'Waiting for an alert';
    }
  }

  /** @param {Incident[]} records */
  function renderShelf(records) {
    if (!shelf) return;
    const entries = records.flatMap(record => record.artifacts.filter(artifact => artifact.name !== 'model_context').map(artifact => ({ record, artifact }))).slice(0, 8);
    const revision = entries.map(({ record, artifact }) => `${record.incident}:${artifact.name}`).join(',');

    if (revision === shelfRevision) return;
    shelfRevision = revision;

    shelf.replaceChildren(...entries.map(({ record, artifact }) => pictureLink(record, artifact)));

    if (!entries.length) shelf.append(element('span', 'No saved evidence yet'));
  }

  /** @param {string} label @param {string} value */
  function fact(label, value) {
    const paragraph = element('p');
    paragraph.append(element('strong', label), document.createTextNode(value));

    return paragraph;
  }

  /** @param {Incident} record @param {HTMLElement} [custody] */
  function renderCustody(record, custody = document.createElement('div')) {
    custody.replaceChildren();
    const summary = element('div', '', 'incident-summary');
    summary.append(fact('Incident', record.incident), fact('Trigger camera', record.trigger?.camera || 'Not recorded'), fact('Evidence camera', record.evidence_camera || 'Not recorded'), fact('Analysis', record.models?.mock ? 'Mock run' : 'Model inference'), fact('Decision', record.state === 'investigating' ? 'Investigating' : record.action), fact('Recorded', timestamp(record.t)));
    custody.append(summary);

    const grid = element('div', '', 'artifact-grid');

    for (const artifact of record.artifacts) {
      const figure = element('figure', '', `artifact${artifact.name === 'original_frame' ? ' original-scene' : ''}`);
      figure.id = `artifact-${record.incident}-${artifact.name}`;
      const img = document.createElement('img');
      img.src = artifact.url;
      img.alt = artifact.name.replaceAll('_', ' ');

      if (artifact.integrity === 'missing') {
        figure.append(element('p', 'Image unavailable. The recorded hash is retained below.'));
      } else if (artifact.name === 'face_crop') {
        const disclosure = document.createElement('details');
        disclosure.append(element('summary', 'Face evidence · reveal for human review'), img);
        figure.append(disclosure);
      } else figure.append(img);
      figure.append(element('figcaption', `${artifact.name.replaceAll('_', ' ')} · ${artifact.camera || record.evidence_camera || 'camera not recorded'}`));
      figure.append(element('p', `Captured ${timestamp(artifact.captured_at)}`));

      if (artifact.relationship === 'composite') {
        figure.append(element('p', `Model input combines: ${(artifact.sources || []).join(', ')}`));

        if (artifact.missing_sources?.length) figure.append(element('p', `Unavailable: ${artifact.missing_sources.join(', ')}`));
      }

      if (artifact.source_artifact) {
        const association = element('p', `${artifact.relationship === 'subsequent_capture' ? 'Later camera capture, associated with' : 'Source image:'} `);
        const source = document.createElement('a');
        source.href = `#artifact-${record.incident}-${artifact.source_artifact}`;
        source.textContent = artifact.source_artifact.replaceAll('_', ' ');
        association.append(source);
        figure.append(association);
      }

      if (artifact.relationship === 'composite') figure.append(element('p', 'Model context from an earlier version, which sent this composite of the scene and crops to both checks.'));

      if (artifact.bbox) figure.append(element('p', `Target box in source coordinates: ${artifact.bbox.join(', ')}`));
      const good = artifact.integrity === 'verified';
      figure.append(element('p', good ? 'SHA-256 matches saved record' : `Integrity check: ${artifact.integrity}`, good ? 'integrity-ok' : 'integrity-error'));
      figure.append(element('code', `SHA-256 ${artifact.sha256}`));
      const original = document.createElement('a');
      original.href = artifact.url;
      original.textContent = 'Open original image';
      original.target = '_blank';
      original.rel = 'noopener';
      figure.append(original);
      grid.append(figure);
    }

    if (!record.artifacts.length) grid.append(element('p', 'No image was saved for this incident.'));
    custody.append(grid);

    if (record.comparison) {
      const comparison = element('section', '', 'comparison-record');
      comparison.append(element('h3', `Comparison: ${record.comparison.status}`), element('p', record.comparison.reason), element('p', `Verifier supports claim: ${verdictLabel(record.comparison.verifier_supports)} · Analyst description names it: ${verdictLabel(record.comparison.analyst_names_claim)} (advisory)`));
      custody.append(comparison);
    }

    custody.append(custodyCryptography(record), element('h2', 'Chain of custody'));
    const timeline = element('ol', '', 'custody-timeline');

    for (const event of [...record.custody].sort((first, second) => first.t - second.t)) {
      const item = element('li');
      const time = document.createElement('time');
      time.textContent = timestamp(event.t);
      time.dateTime = new Date(event.t * 1000).toISOString();
      item.append(time, element('h3', event.stage.replaceAll('_', ' ')), element('p', event.detail));
      timeline.append(item);
    }

    custody.append(timeline);
    const metadata = document.createElement('details');
    metadata.append(element('summary', 'Configured models and run mode'), element('p', JSON.stringify(record.models, null, 2), 'model-record'));
    custody.append(metadata);
    const download = document.createElement('a');
    download.href = `/incidents/${encodeURIComponent(record.incident)}`;
    download.download = `${record.incident}.json`;
    download.textContent = 'Download incident record (JSON)';
    const downloadParagraph = element('p');
    downloadParagraph.append(download);
    custody.append(downloadParagraph);
  }

  /** @param {boolean|null|undefined} verdict */
  function verdictLabel(verdict) {
    return verdict == null ? 'uncertain' : verdict ? 'yes' : 'no';
  }

  async function loadCustody() {
    if (!custody) return;

    try {
      /** @type {Incident[]} */
      const records = selected ? [await read(`/incidents/${encodeURIComponent(selected)}`)] : await read(`/incidents?limit=${pageSize + 1}&offset=${page * pageSize}`);
      const list = element('div', '', 'incident-rows');

      for (const record of records.slice(0, pageSize)) {
        const row = document.createElement('details');
        row.className = 'incident-row';
        const summary = document.createElement('summary');
        summary.append(element('span', timestamp(record.t), 'row-time'), element('strong', record.claim || record.action), element('span', custodyLocation(record), 'row-location'), element('span', record.state === 'investigating' ? 'Investigating' : record.action, 'row-outcome'), element('span', custodyIntegrity(record), 'row-integrity'));
        const detail = element('div', '', 'incident-expanded');
        row.append(summary, detail);
        let loaded = false;
        row.addEventListener('toggle', async () => {
          if (!row.open || loaded) return;
          loaded = true;
          detail.textContent = 'Loading full record…';

          try {
            renderCustody(await read(`/incidents/${encodeURIComponent(record.incident)}`), detail);
          } catch {
            detail.textContent = 'Record unavailable. Collapse and expand to retry.';
            loaded = false;
          }
        });
        list.append(row);

        if (selected) row.open = true;
      }

      if (!records.length) list.append(element('p', 'No incident records yet. New investigations will appear here.'));
      custody.replaceChildren(list);
      const navigation = document.getElementById('incident-pagination');

      if (navigation) navigation.hidden = Boolean(selected);
      const previous = document.getElementById('previous-incidents');
      const next = document.getElementById('next-incidents');

      if (previous instanceof HTMLButtonElement) previous.disabled = page === 0;

      if (next instanceof HTMLButtonElement) next.disabled = records.length <= pageSize;
      const label = document.getElementById('incident-page-label');

      if (label) label.textContent = `Page ${page + 1}`;

      if (custodyError) custodyError.textContent = '';
    } catch {
      if (custodyError) custodyError.textContent = 'Could not load this incident history. The record may be unavailable. Reload to retry.';
      custody.replaceChildren();
    }
  }

  async function refresh() {
    try {
      /** @type {Incident[]} */
      const alerts = await read('/incidents?limit=1&action=alert');

      renderPhone(alerts.find(record => record.notification.simulated && ['delivered', 'acknowledged'].includes(record.notification.status)));

      if (phoneError) phoneError.textContent = '';

      if (shelf) renderShelf(await read('/incidents?limit=8&evidence_only=true'));

      if (cloudStatus) {
        const transport = await read('/expanso');
        const receipts = transport.receipts || [];

        if (cloudShelf) {
          const links = receipts.flatMap(/** @param {any} receipt */ receipt => Object.entries(receipt.crops).map(([name, crop]) => {
            const item = /** @type {{url:string}} */ (crop);
            const link = document.createElement('a');
            link.href = `/custody?incident=${encodeURIComponent(receipt.incident)}`;
            link.target = 'sherlock-evidence';
            const image = document.createElement('img');
            image.src = item.url;
            image.alt = `${name.replaceAll('_', ' ')} · core storage verified`;
            link.append(image);

            return link;
          }));

          cloudShelf.replaceChildren(...links.slice(0, 3));
        }

        cloudStatus.textContent = receipts.length ? `${receipts.length} core receipts · crop hashes verified` : transport.edge ? 'Edge accepted · awaiting core readback' : 'Awaiting verified core storage';
      }
    } catch {
      if (phoneError) phoneError.textContent = 'Connection unavailable. Displayed incident may be out of date.';

      if (cloudStatus) cloudStatus.textContent = 'Connection unavailable · cloud unconfirmed';
    }
  }

  document.addEventListener('click', event => {
    const target = event.target;

    if (!(target instanceof Element) || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || event.button !== 0) return;
    const link = target.closest('a');

    if (!(link instanceof HTMLAnchorElement) || !['sherlock-phone', 'sherlock-evidence', 'sherlock-flow'].includes(link.target)) return;

    if (!isPhone && !isCustody && link.pathname === '/custody') {
      const incident = new URL(link.href).searchParams.get('incident');

      if (incident && !link.closest('dialog')) {
        event.preventDefault();
        void showCustodySummary(incident);

        return;
      }
    }

    const tab = window.open(link.href, link.target);

    if (tab) {
      event.preventDefault();
      tab.focus();
    }
  });

  const phoneView = document.getElementById('phone-view');
  const custodyView = document.getElementById('custody-view');

  if (phoneView) phoneView.hidden = !isPhone;

  if (custodyView) custodyView.hidden = !isCustody;

  if (isCustody) {
    document.getElementById('previous-incidents')?.addEventListener('click', () => { page = Math.max(0, page - 1); void loadCustody(); });
    document.getElementById('next-incidents')?.addEventListener('click', () => { page += 1; void loadCustody(); });
    void loadCustody();
  } else if (!isPhone) {
    void refresh();
    setInterval(() => { void refresh(); }, 2000);
  }
})();
