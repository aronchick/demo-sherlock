// @ts-check
'use strict';

/** @param {string} tag @param {string} [text] @param {string} [className] */
function custodyNode(tag, text = '', className = '') {
  const node = document.createElement(tag);
  node.textContent = text;
  node.className = className;

  return node;
}

/** @param {any} record */
function custodyIntegrity(record) {
  const v = record.verification;

  if (!v || v.signature === 'unsigned') return 'Unsigned legacy record';

  if (v.signature !== 'verified' || v.artifacts !== 'verified' || v.history !== 'verified') return 'Integrity needs attention';

  return 'Signature, files and history verified';
}

/** @param {any} record */
function custodyLocation(record) {
  return [record.location?.building, record.location?.floor, record.location?.area, record.trigger?.camera].filter(Boolean).join(' · ') || 'Location not recorded';
}

/** @param {string} text */
function coloredJson(text) {
  const fragment = document.createDocumentFragment();
  const pattern = /"(?:\\.|[^"\\])*"(?:\s*:)?|\b(?:true|false|null)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g;
  let offset = 0;

  for (const match of text.matchAll(pattern)) {
    const at = match.index || 0;
    fragment.append(document.createTextNode(text.slice(offset, at)));
    const token = match[0];
    const kind = token.startsWith('"') ? (token.endsWith(':') ? 'key' : 'string') : /^[\d-]/.test(token) ? 'number' : 'literal';
    fragment.append(custodyNode('span', token, `json-${kind}`));
    offset = at + token.length;
  }

  fragment.append(document.createTextNode(text.slice(offset)));

  return fragment;
}

/** @param {unknown} data @param {string} title */
function custodyJson(data, title) {
  const section = custodyNode('section', '', 'json-viewer');
  const lines = JSON.stringify(data, null, 2).split('\n');
  let page = 0;
  const pageSize = 32;
  const heading = custodyNode('h3', title);
  const pre = document.createElement('pre');
  pre.tabIndex = 0;
  pre.setAttribute('aria-label', title);
  const nav = custodyNode('nav', '', 'json-navigation');
  nav.setAttribute('aria-label', `${title} pages`);
  const previous = document.createElement('button');
  const next = document.createElement('button');
  const copy = document.createElement('button');
  const label = custodyNode('span');
  previous.textContent = 'Previous lines';
  next.textContent = 'Next lines';
  copy.textContent = 'Copy full JSON';

  function render() {
    const start = page * pageSize;
    pre.replaceChildren(coloredJson(lines.slice(start, start + pageSize).join('\n')));
    label.textContent = `Lines ${start + 1}–${Math.min(start + pageSize, lines.length)} of ${lines.length}`;
    previous.disabled = page === 0;
    next.disabled = start + pageSize >= lines.length;
  }

  previous.addEventListener('click', () => { page--; render(); });
  next.addEventListener('click', () => { page++; render(); });
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(lines.join('\n'));
      copy.textContent = 'Copied full JSON';
    } catch {
      copy.textContent = 'Copy unavailable; use download';
    }
  });
  nav.append(previous, label, next, copy);
  section.append(heading, nav, pre);
  render();

  return section;
}

/** @param {any} record */
function custodyCryptography(record) {
  const section = custodyNode('section', '', 'cryptography-record');
  section.append(custodyNode('h2', 'Cryptographic verification'), custodyNode('p', custodyIntegrity(record), 'integrity-status'));
  section.append(custodyNode('p', 'A local application key signs each revision and its evidence hashes. This verifies consistency with that key; it does not independently attest the camera, operator or time.', 'integrity-scope'));
  const signature = record.signature;

  if (signature) {
    const fields = document.createElement('dl');

    for (const [label, value] of Object.entries({ Algorithm: signature.algorithm, Revision: signature.revision, 'Record SHA-256': signature.payload_sha256, 'Previous revision SHA-256': signature.previous_sha256 || 'First revision', 'Signer fingerprint (SHA-256)': signature.key_fingerprint, 'Public key (base64)': signature.public_key, 'Signature (base64)': signature.signature })) {
      fields.append(custodyNode('dt', label), custodyNode('dd', String(value)));
    }

    section.append(fields);
  }

  const disclosure = document.createElement('details');
  disclosure.className = 'record-disclosure';
  disclosure.append(custodyNode('summary', 'Formatted record JSON'));
  disclosure.append(custodyJson(record, 'Current record'));
  const history = document.createElement('details');
  history.className = 'record-disclosure';
  history.append(custodyNode('summary', 'Signed revision history and JSON'));
  let loaded = false;
  history.addEventListener('toggle', async () => {
    if (!history.open || loaded) return;
    loaded = true;
    const state = custodyNode('p', 'Loading signed revisions…');
    history.append(state);

    try {
      const response = await fetch(`/incidents/${encodeURIComponent(record.incident)}/signed-history`, {cache: 'no-store'});

      if (!response.ok) throw new Error('history unavailable');
      const data = await response.json();
      state.remove();
      history.append(custodyJson(data, 'Complete signed history'));
      const download = document.createElement('a');
      download.href = response.url;
      download.download = `${record.incident}-signed-history.json`;
      download.textContent = 'Download full signed history';
      history.append(download);
    } catch {
      state.textContent = 'Signed history unavailable for this record.';
      loaded = false;
    }
  });
  section.append(disclosure, history);

  return section;
}

/** @param {string} iid */
async function showCustodySummary(iid) {
  document.getElementById('custody-summary-modal')?.remove();
  const dialog = document.createElement('dialog');
  dialog.id = 'custody-summary-modal';
  dialog.className = 'custody-modal';
  dialog.setAttribute('aria-labelledby', 'custody-modal-title');
  const close = document.createElement('button');
  close.type = 'button';
  close.textContent = 'Close';
  close.className = 'modal-close';
  close.addEventListener('click', () => dialog.close());
  const heading = custodyNode('h2', 'Incident summary');
  heading.id = 'custody-modal-title';
  const content = custodyNode('div', 'Loading record…', 'custody-modal-content');
  dialog.append(close, heading, content);
  dialog.addEventListener('close', () => dialog.remove());
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
  document.body.append(dialog);
  dialog.showModal();

  try {
    const response = await fetch(`/incidents/${encodeURIComponent(iid)}`, {cache: 'no-store'});

    if (!response.ok) throw new Error('record unavailable');
    const record = await response.json();
    heading.textContent = record.claim || 'Incident summary';
    content.replaceChildren();
    content.append(custodyNode('p', `${new Date(record.t * 1000).toLocaleString()} · ${custodyLocation(record)}`, 'modal-meta'));
    const scene = record.artifacts.find(/** @param {any} artifact */ artifact => artifact.name === 'original_frame' && artifact.integrity !== 'missing');

    if (scene) {
      const image = document.createElement('img');
      image.src = scene.url;
      image.alt = 'Original scene associated with this incident';
      content.append(image);
    }

    const decision = record.state === 'investigating' ? 'Investigation in progress' : record.action === 'alert' ? 'Alert' : record.action === 'ignore' ? 'Archived' : 'Review required';
    content.append(custodyNode('p', `${decision} · ${record.artifacts.length} evidence images`, 'modal-decision'), custodyNode('p', record.comparison?.reason || 'No model comparison requested.', 'modal-meta'), custodyNode('p', custodyIntegrity(record), 'integrity-status'));
    const link = document.createElement('a');
    link.href = `/custody?incident=${encodeURIComponent(iid)}`;
    link.target = 'sherlock-evidence';
    link.textContent = 'Open full evidence, signatures and history';
    content.append(link);
  } catch {
    content.textContent = 'This record is unavailable. Close and try again.';
  }
}
