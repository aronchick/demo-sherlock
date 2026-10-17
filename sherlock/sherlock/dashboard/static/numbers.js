/* Saved measurements are separate from the live event counters. */
function measurementText(id, value) {
  const element = document.getElementById(id);

  if (element) element.textContent = value;
}

function measuredNumber(value, suffix = '') {
  return Number.isFinite(value)
    ? value.toLocaleString(undefined, {maximumFractionDigits: 1}) + suffix
    : 'TBD';
}

function measuredPair(before, after, suffix) {
  if (before == null && after == null) return 'TBD';

  return measuredNumber(before, suffix) + ' → ' + measuredNumber(after, suffix);
}

function measuredPercent(value) {
  return Number.isFinite(value) ? value * 100 : null;
}

function resultLabel(id, label, source) {
  const element = document.getElementById(id);

  if (!element) return;

  element.textContent = label;
  element.title = source || '';
}

function renderMeasurements(data) {
  const director = data.director;
  const ablation = data.ablation;
  const scout = data.scout;

  resultLabel('number-director', director
    ? `Director · C${director.concurrency} · ${director.succeeded}/${director.attempted} OK`
    : 'Director', director?.file);
  measurementText('number-latency', director
    ? measuredNumber(director.p50_ms) + ' / ' + measuredNumber(director.p95_ms, ' ms')
    : 'TBD');
  measurementText('number-throughput', director
    ? measuredNumber(director.output_tokens_per_s, ' tok/s') + ' · ' + measuredNumber(director.requests_per_s, ' req/s')
    : 'TBD');

  const delta = director?.prefix_cache_p95_change_pct;

  measurementText('number-cache', Number.isFinite(delta)
    ? (delta > 0 ? '+' : '') + measuredNumber(delta, '%') : 'TBD');
  resultLabel('number-ablation', ablation
    ? `NeMo Gym · ${ablation.tasks} tasks` : 'NeMo Gym', ablation
    ? `${ablation.config} · ${ablation.file}` : '');
  measurementText('number-ablation-config', ablation?.config
    ? ablation.config.slice(0, 32) : 'TBD');
  measurementText('number-success', measuredNumber(measuredPercent(ablation?.task_success), '%'));
  measurementText('number-validity', measuredNumber(measuredPercent(ablation?.tool_call_validity), '%'));
  resultLabel('number-scout', scout
    ? `Scout · ${scout.source} ${scout.split} · ${scout.boxes} weapon boxes`
    : 'Scout · weapon', scout?.file);
  measurementText('number-precision', measuredPair(
    measuredPercent(scout?.before_precision), measuredPercent(scout?.after_precision), '%'));
  measurementText('number-recall', measuredPair(
    measuredPercent(scout?.before_recall), measuredPercent(scout?.after_recall), '%'));
  measurementText('numbers-status', director || ablation || scout
    ? 'Latest saved measurements' : 'No measured results yet');
}

async function refreshMeasurements() {
  try {
    const response = await fetch('/numbers', {cache: 'no-store', signal: AbortSignal.timeout(5000)});

    if (!response.ok) throw new Error('Measurements request failed');

    renderMeasurements(await response.json());
  } catch {
    renderMeasurements({director: null, ablation: null, scout: null});
    measurementText('numbers-status', 'Measurements unavailable');
  }
}

refreshMeasurements();

setInterval(refreshMeasurements, 10000);
