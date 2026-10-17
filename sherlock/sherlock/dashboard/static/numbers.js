/* Saved measurements are separate from the live event counters. */
const NOT_MEASURED = 'not measured';

function measurementText(id, value) {
  const element = document.getElementById(id);

  if (element) element.textContent = value;
}

function measuredNumber(value, suffix = '') {
  return Number.isFinite(value)
    ? value.toLocaleString(undefined, {maximumFractionDigits: 1}) + suffix
    : NOT_MEASURED;
}

function measuredPair(before, after, suffix) {
  if (before == null && after == null) return NOT_MEASURED;

  return measuredNumber(before, suffix) + ' → ' + measuredNumber(after, suffix);
}

function measuredPercent(value) {
  return Number.isFinite(value) ? value * 100 : null;
}

/* A result heading names where it was measured, as the artifact recorded it, so a
   local run never reads as a vLLM one. */
function resultHeading(id, name, detail, result) {
  const element = document.getElementById(id);

  if (!element) return;

  const label = document.createElement('b');
  const more = document.createElement('span');
  const where = document.createElement('small');

  label.textContent = name;
  more.textContent = detail ? ' · ' + detail : '';
  where.textContent = result ? result.measured_on || 'runtime not recorded' : '';
  element.replaceChildren(label, more, where);
  element.title = result?.file || '';
}

function renderMeasurements(data) {
  const director = data.director;
  const ablation = data.ablation;
  const scout = data.scout;

  resultHeading('number-director', 'Director', director
    ? `C${director.concurrency} · ${director.succeeded}/${director.attempted} OK` : '', director);
  measurementText('number-latency', director
    ? measuredNumber(director.p50_ms) + ' / ' + measuredNumber(director.p95_ms, ' ms')
    : NOT_MEASURED);
  measurementText('number-throughput', director
    ? measuredNumber(director.output_tokens_per_s, ' tok/s') + ' · ' + measuredNumber(director.requests_per_s, ' req/s')
    : NOT_MEASURED);

  const delta = director?.prefix_cache_p95_change_pct;

  measurementText('number-cache', Number.isFinite(delta)
    ? (delta > 0 ? '+' : '') + measuredNumber(delta, '%') : NOT_MEASURED);
  resultHeading('number-ablation', 'NeMo Gym', ablation ? `${ablation.tasks} tasks` : '', ablation);
  measurementText('number-ablation-config', ablation?.config
    ? ablation.config.slice(0, 32) : NOT_MEASURED);
  measurementText('number-success', measuredNumber(measuredPercent(ablation?.task_success), '%'));
  measurementText('number-validity', measuredNumber(measuredPercent(ablation?.tool_call_validity), '%'));
  resultHeading('number-scout', 'Scout', scout
    ? `weapon · ${scout.source} ${scout.split} n=${scout.boxes}` : 'weapon', scout);
  measurementText('number-precision', measuredPair(
    measuredPercent(scout?.before_precision), measuredPercent(scout?.after_precision), '%'));
  measurementText('number-recall', measuredPair(
    measuredPercent(scout?.before_recall), measuredPercent(scout?.after_recall), '%'));
  measurementText('numbers-status', director || ablation || scout
    ? 'Latest measurements' : 'No measured results yet');
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
