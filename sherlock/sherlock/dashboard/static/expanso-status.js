// @ts-check
'use strict';

/** Local acceptance is transport evidence, not a current Cloud health check.
 * @param {any} status
 */
function expansoStatusText(status) {
  if (!status) return 'Expanso status unavailable';

  if (!status.configured) return 'Expanso transport not enabled in this run';

  const age = Number.isFinite(status.last_delivery_s)
    ? `last accepted delivery ${Math.round(status.last_delivery_s)}s ago`
    : 'no accepted delivery yet';

  const state = status.delivery_state;

  const activity = state === 'active' ? 'accepting events' : state === 'failed'
    ? 'last send failed' : state === 'idle' ? 'idle' : 'waiting for delivery';

  const mode = status.mode === 'cloud' ? 'Cloud-managed pipelines configured' : 'local pipelines configured';

  const receipts = status.receipts?.length || 0;

  return `Expanso Edge ${activity} · ${age} · ${mode} · ${receipts} verified core receipt${receipts === 1 ? '' : 's'}`;
}
