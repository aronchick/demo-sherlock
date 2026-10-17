/* global document, window, fetch */
'use strict';

(() => {
  if (window.location.pathname !== '/phone') return;
  document.body.classList.add('responder-page');
  window.name = 'sherlock-phone';
  const view = document.getElementById('phone-view');

  if (view) view.hidden = false;
  const root = document.getElementById('phone-app');

  if (!root) return;
  const app = root;
  /** @type {any[]} */
  let records = [];
  /** @type {any} */
  let selected = null;
  let page = 0;
  let imageIndex = 0;
  let pending = false;
  let connected = false;
  let loaded = false;
  let message = '';
  let revision = '';
  const pageSize = 4;

  /** @param {string} tag @param {string} [text] @param {string} [className] */
  function node(tag, text = '', className = '') {
    const item = document.createElement(tag);
    item.textContent = text;
    item.className = className;

    return item;
  }

  /** @param {string} text @param {() => void} action @param {boolean} [disabled] */
  function button(text, action, disabled = false) {
    const item = document.createElement('button');
    item.type = 'button';
    item.textContent = text;
    item.disabled = disabled;
    item.addEventListener('click', action);

    return item;
  }

  /** @param {string} url @param {RequestInit} [options] */
  async function read(url, options) {
    const response = await fetch(url, { ...options, cache: 'no-store' });

    if (!response.ok) throw new Error(`Request failed (${response.status})`);

    return response.json();
  }

  /** @param {any} record */
  function location(record) {
    const place = record.location || {};

    return [place.building || 'Building unknown', place.floor || 'Floor unknown', place.area || 'Area unknown'].join(' · ');
  }

  /** @param {number} t */
  function time(t) {
    return t ? new Date(t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : 'Time unavailable';
  }

  /** @param {any} record */
  function status(record) {
    if (record.response?.status === 'resolved') return 'Resolved';

    if (record.response?.team_notified_at) return 'Team notified · demo';

    return record.response?.status === 'investigating' ? 'Investigating' : 'New notice';
  }

  /** @param {string} action */
  async function respond(action) {
    if (!selected || pending || !connected) return;
    pending = true;
    message = 'Saving response…';
    render();

    try {
      selected = await read(`/incidents/${encodeURIComponent(selected.incident)}/respond`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action }),
      });
      message = action === 'notify_team' ? 'Team notification saved in this demo.' : action === 'resolve' ? 'Incident marked resolved.' : 'You are investigating this incident.';
    } catch {
      message = 'Response was not confirmed. Reconnect and try again.';
    } finally {
      pending = false;
      render();
    }
  }

  function inbox() {
    const heading = node('div', '', 'responder-heading');
    heading.append(node('h1', 'Notices'), node('span', `${records.length}${records.length === 100 ? '+' : ''} received`));
    app.append(heading);
    const list = node('div', '', 'responder-notices');

    if (!records.length) list.append(node('p', loaded ? 'Standing by. Verified alerts will appear here.' : 'Connecting to security feed…', 'responder-empty'));

    for (const record of records.slice(page * pageSize, (page + 1) * pageSize)) {
      const row = button('', () => { selected = record; imageIndex = 0; message = ''; render(); });
      row.className = 'responder-notice';
      row.append(node('span', `${time(record.t)} · ${status(record)}`, 'responder-meta'), node('strong', record.claim || 'Security incident'), node('span', location(record), 'responder-place'));
      list.append(row);
    }

    app.append(list);
    const pages = node('nav', '', 'responder-pagination');
    pages.setAttribute('aria-label', 'Notice pages');
    pages.append(button('Previous', () => { page--; render(); }, page === 0), node('span', `${page + 1} / ${Math.max(1, Math.ceil(records.length / pageSize))}`), button('Next', () => { page++; render(); }, (page + 1) * pageSize >= records.length));
    app.append(pages);
    const hint = node('p', 'Select a notice to inspect evidence and respond.', 'responder-hint');
    app.append(hint);
  }

  function detail() {
    const record = selected;
    const back = button('‹ Notices', () => { selected = null; message = ''; render(); });
    back.className = 'responder-back';
    app.append(back);
    const heading = node('div', '', 'responder-detail-heading');
    heading.append(node('p', `${time(record.t)} · ${status(record)}`, 'responder-meta'), node('h1', record.claim || 'Security incident'), node('p', location(record), 'responder-location'), node('p', `Camera ${record.trigger?.camera || 'unknown'}${record.location?.simulated ? ' · Demo location' : ''}`, 'responder-meta'));
    app.append(heading);
    const artifacts = (record.artifacts || []).filter(/** @param {any} item */ item => item.url && ['original_frame', 'object_crop', 'weapon_crop', 'face_crop', 'model_context'].includes(item.name));
    imageIndex = Math.max(0, Math.min(imageIndex, artifacts.length - 1));
    const gallery = node('figure', '', 'responder-gallery');

    if (artifacts.length) {
      const artifact = artifacts[imageIndex];
      const img = document.createElement('img');
      img.src = artifact.url;
      img.alt = artifact.name.replaceAll('_', ' ');
      img.addEventListener('error', () => { img.replaceWith(node('p', 'Image unavailable. Evidence record remains accessible.')); });
      gallery.append(img);
      const controls = node('figcaption', '', 'responder-pagination');
      controls.append(button('‹', () => { imageIndex--; render(); }, imageIndex === 0), node('span', `${imageIndex + 1}/${artifacts.length} · ${img.alt}`), button('›', () => { imageIndex++; render(); }, imageIndex === artifacts.length - 1));
      controls.firstElementChild?.setAttribute('aria-label', 'Previous evidence image');
      controls.lastElementChild?.setAttribute('aria-label', 'Next evidence image');
      gallery.append(controls);
    } else gallery.append(node('p', 'Evidence images are not available yet.'));
    app.append(gallery);
    const actions = node('div', '', 'responder-actions');
    const disabled = pending || !connected;
    const resolved = record.response?.status === 'resolved';
    actions.append(button(record.response?.status === 'investigating' ? 'Investigating' : 'Investigate', () => { void respond('investigate'); }, disabled || resolved || record.response?.status === 'investigating'), button(record.response?.team_notified_at ? 'Team notified' : 'Notify team', () => { void respond('notify_team'); }, disabled || resolved || Boolean(record.response?.team_notified_at)));
    const footer = node('div', '', 'responder-detail-footer');
    const evidence = document.createElement('a');
    evidence.href = `/custody?incident=${encodeURIComponent(record.incident)}`;
    evidence.target = 'sherlock-evidence';
    evidence.textContent = 'Full evidence';
    footer.append(evidence, button(resolved ? 'Resolved' : 'Resolve', () => { void respond('resolve'); }, disabled || resolved));
    app.append(actions, footer);
  }

  function render() {
    const active = document.activeElement;
    const focusText = active instanceof HTMLButtonElement && app.contains(active) ? active.textContent : null;
    app.replaceChildren();
    const chrome = node('div', '', 'responder-chrome');
    chrome.append(node('span', new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })), node('span', 'Sherlock · Demo'), node('span', connected ? 'Live' : 'Offline', connected ? 'responder-live' : 'responder-offline'));
    app.append(chrome);

    if (selected) detail(); else inbox();
    const feedback = node('p', message || (connected ? 'Feed connected · simulated delivery' : 'Feed disconnected · retrying automatically'), 'responder-feedback');
    feedback.setAttribute('role', 'status');
    app.append(feedback);

    if (focusText) Array.from(app.querySelectorAll('button')).find(item => item.textContent === focusText)?.focus();
  }

  async function poll() {
    try {
      const next = await read('/incidents?limit=100&action=alert');

      if (pending) return;
      const current = selected ? await read(`/incidents/${encodeURIComponent(selected.incident)}`) : null;

      if (pending) return;
      records = next;
      loaded = true;
      connected = true;

      if (current) selected = current;
      page = Math.min(page, Math.max(0, Math.ceil(records.length / pageSize) - 1));
      const nextRevision = JSON.stringify([records, selected]);

      if (revision !== nextRevision) { revision = nextRevision; render(); }
      else if (app.querySelector('.responder-offline')) render();
    } catch {
      connected = false;
      render();
    } finally {
      window.setTimeout(() => { void poll(); }, 2000);
    }
  }

  render();
  void poll();
})();
