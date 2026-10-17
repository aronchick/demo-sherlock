// @ts-check
'use strict';

/** @typedef {{name:string, sha256:string, url:string, integrity:string}} Artifact */
/** @typedef {{stage:string, t:number, detail:string}} CustodyEvent */
/** @typedef {{incident:string, action:string, claim:string, t:number, verified?:boolean, trigger?:{camera:string}, artifacts:Artifact[], custody:CustodyEvent[], models?:unknown, mode?:unknown, notification:{simulated:boolean,status:string,delivered_at?:number,acknowledged_at?:number}}} Incident */

(() => {
  const phone = document.getElementById('phone-content');
  const phoneError = document.getElementById('phone-error');
  const shelf = document.getElementById('evidence-links');
  const custody = document.getElementById('custody-content');
  const custodyError = document.getElementById('custody-error');
  const selected = new URLSearchParams(location.search).get('incident');
  const isPhone = location.pathname === '/phone';
  const isCustody = location.pathname === '/custody';
  let page = 0;
  let phoneRevision = '';
  let shelfRevision = '';
  let acknowledgePending = false;
  const pageSize = 8;

  /** @param {string} tag @param {string} [text] @param {string} [className] */
  function element(tag, text = '', className = '') {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = className;

    return node;
  }

  /** @param {number|undefined} value */
  function timestamp(value) {
    return value === undefined ? 'Not recorded' : new Date(value * 1000).toLocaleString();
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
    link.title = `Chain of custody: ${artifact.name}, ${record.incident}`;
    const img = document.createElement('img');
    img.src = artifact.url;
    img.alt = `${artifact.name.replaceAll('_', ' ')} from ${record.incident}. View chain of custody`;

    if (artifact.name === 'face_crop') img.className = 'face-preview';
    link.append(img);

    return link;
  }

  /** @param {Incident} record */
  async function acknowledge(record) {
    if (acknowledgePending) return;
    acknowledgePending = true;

    try {
      const updated = await read(`/incidents/${encodeURIComponent(record.incident)}/acknowledge`, { method: 'POST' });
      phoneRevision = '';
      renderPhone(updated);

      if (phoneError) phoneError.textContent = '';
    } catch {
      if (phoneError) phoneError.textContent = 'Acknowledgment was not saved. Try again.';
    } finally {
      acknowledgePending = false;
    }
  }

  /** @param {Incident|undefined} record */
  function renderPhone(record) {
    if (!phone) return;
    const revision = record ? `${record.incident}:${record.notification.status}` : 'empty';

    if (revision === phoneRevision) return;
    phoneRevision = revision;
    phone.replaceChildren();

    if (!record) {
      phone.append(element('h2', 'Ready for an alert'), element('p', 'Verified incidents will appear here. Routine detections do not notify the responder.'));

      return;
    }

    const acknowledged = record.notification.status === 'acknowledged';
    phone.append(element('p', acknowledged ? 'ACKNOWLEDGED · SIMULATED' : 'DELIVERED · SIMULATED', `phone-status${acknowledged ? ' acknowledged' : ''}`));
    phone.append(element('h2', record.claim || 'Verified incident'));
    phone.append(element('p', `${record.trigger?.camera || 'Camera not recorded'} · ${timestamp(record.t)}`));
    const artifact = record.artifacts.find(item => item.name !== 'face_crop');

    if (artifact) {
      const link = pictureLink(record, artifact);
      link.className = 'phone-image';
      phone.append(link);
    }

    phone.append(element('p', `Incident ${record.incident}`));
    const link = document.createElement('a');
    link.href = custodyUrl(record.incident);
    link.textContent = 'View evidence and chain of custody';
    phone.append(link);
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = acknowledged ? 'Investigating · acknowledgment saved' : 'Acknowledge · Investigating';
    button.disabled = acknowledged;
    button.addEventListener('click', () => { void acknowledge(record); });
    phone.append(button);

    if (acknowledged) phone.append(element('p', `Acknowledged ${timestamp(record.notification.acknowledged_at)}`));
  }

  /** @param {Incident[]} records */
  function renderShelf(records) {
    if (!shelf) return;
    const entries = records.flatMap(record => record.artifacts.map(artifact => ({ record, artifact }))).slice(0, 8);
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

  /** @param {Incident} record */
  function renderCustody(record) {
    if (!custody) return;
    custody.replaceChildren();
    const summary = element('div', '', 'incident-summary');
    summary.append(fact('Incident', record.incident), fact('Camera', record.trigger?.camera || 'Not recorded'), fact('Decision', record.action), fact('Recorded', timestamp(record.t)));
    custody.append(summary, element('h2', record.claim || 'Incident evidence'));
    const grid = element('div', '', 'artifact-grid');

    for (const artifact of record.artifacts) {
      const figure = element('figure', '', 'artifact');
      const img = document.createElement('img');
      img.src = artifact.url;
      img.alt = artifact.name.replaceAll('_', ' ');

      if (artifact.name === 'face_crop') {
        const disclosure = document.createElement('details');
        disclosure.append(element('summary', 'Face evidence · reveal for human review'), img);
        figure.append(disclosure);
      } else figure.append(img);
      figure.append(element('figcaption', artifact.name.replaceAll('_', ' ')));
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
    custody.append(grid, element('h2', 'Chain of custody'));
    const timeline = element('ol', '', 'custody-timeline');

    for (const event of record.custody) {
      const item = element('li');
      const time = document.createElement('time');
      time.textContent = timestamp(event.t);
      time.dateTime = new Date(event.t * 1000).toISOString();
      item.append(time, element('h3', event.stage.replaceAll('_', ' ')), element('p', event.detail));
      timeline.append(item);
    }

    custody.append(timeline);
    const metadata = document.createElement('details');
    metadata.append(element('summary', 'Configured models and run mode'), element('p', JSON.stringify({ models: record.models, mode: record.mode }, null, 2), 'model-record'));
    custody.append(metadata);
    const download = document.createElement('a');
    download.href = `/incidents/${encodeURIComponent(record.incident)}`;
    download.download = `${record.incident}.json`;
    download.textContent = 'Download incident record (JSON)';
    const downloadParagraph = element('p');
    downloadParagraph.append(download);
    custody.append(downloadParagraph);
  }

  async function loadCustody() {
    if (!custody) return;

    try {
      if (selected) {
        renderCustody(await read(`/incidents/${encodeURIComponent(selected)}`));
      } else {
        /** @type {Incident[]} */
        const records = await read(`/incidents?limit=${pageSize + 1}&offset=${page * pageSize}`);
        const list = element('div', '', 'incident-list');

        for (const record of records.slice(0, pageSize)) {
          const link = document.createElement('a');
          link.href = custodyUrl(record.incident);
          const artifact = record.artifacts.find(item => item.name !== 'face_crop');

          if (artifact) {
            const img = document.createElement('img');
            img.src = artifact.url;
            img.alt = artifact.name.replaceAll('_', ' ');
            link.append(img);
          }

          const copy = element('div');
          copy.append(element('h2', record.claim || record.action), element('p', `${record.incident} · ${record.trigger?.camera || 'Camera not recorded'} · ${timestamp(record.t)}`), element('p', `${record.action} · ${record.artifacts.length} saved images · View chain of custody`));
          link.append(copy);
          list.append(link);
        }

        if (!records.length) list.append(element('p', 'No incident records yet. New investigations will appear here.'));
        custody.replaceChildren(list);
        const navigation = document.getElementById('incident-pagination');

        if (navigation) navigation.hidden = false;
        const previous = document.getElementById('previous-incidents');
        const next = document.getElementById('next-incidents');

        if (previous instanceof HTMLButtonElement) previous.disabled = page === 0;

        if (next instanceof HTMLButtonElement) next.disabled = records.length <= pageSize;
        const label = document.getElementById('incident-page-label');

        if (label) label.textContent = `Page ${page + 1}`;
      }

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

      if (!acknowledgePending) renderPhone(alerts.find(record => record.notification.simulated && ['delivered', 'acknowledged'].includes(record.notification.status)));

      if (phoneError) phoneError.textContent = '';

      if (shelf) renderShelf(await read('/incidents?limit=8&evidence_only=true'));
    } catch {
      if (phoneError) phoneError.textContent = 'Connection unavailable. Displayed incident may be out of date.';
    }
  }

  const phoneView = document.getElementById('phone-view');
  const custodyView = document.getElementById('custody-view');

  if (phoneView) phoneView.hidden = !isPhone;

  if (custodyView) custodyView.hidden = !isCustody;

  if (isCustody) {
    document.getElementById('previous-incidents')?.addEventListener('click', () => { page = Math.max(0, page - 1); void loadCustody(); });
    document.getElementById('next-incidents')?.addEventListener('click', () => { page += 1; void loadCustody(); });
    void loadCustody();
  } else {
    void refresh();
    setInterval(() => { void refresh(); }, 2000);
  }
})();
