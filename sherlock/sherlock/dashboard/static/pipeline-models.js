// @ts-check

/** Timestamp measurement, independent of the board's animation queue. */
class CameraCommandTiming {
  /** @param {(seconds:number|null) => void} render */
  constructor(render) {
    this.render = render;
    /** @type {Map<string, number[]>} */
    this.detections = new Map();
    this.incident = '';
    /** @type {number|null} */
    this.started = null;
    this.measured = false;
  }

  /** @param {any} event */
  receive(event) {
    const data = event.data || {};
    const key = `${data.camera}:${data.label}`;

    if (event.stage === 'scout' && data.bbox && Number.isFinite(event.t)) {
      if (data.handled_via) return;

      const previous = this.detections.get(key) || [];

      // Multiple unmatched detections cannot be attributed to an incident safely.
      this.detections.set(key, [...previous, event.t].slice(-2));

      if (this.detections.size > 64) this.detections.clear();

      return;
    }

    if (!['narration', 'tool'].includes(event.stage) || !data.incident) return;

    if (data.incident !== this.incident) {
      this.incident = data.incident;
      this.measured = false;
      this.started = null;
      this.render(null);
      const detections = this.detections.get(key);

      if (detections?.length === 1) this.started = detections[0];

      this.detections.delete(key);
    }

    if (this.measured || this.started === null || event.stage !== 'tool') return;

    if (!['look_at', 'move'].includes(data.tool) || data.error || data.result?.ok !== true) return;

    if (!Number.isFinite(event.t) || event.t < this.started) return;

    this.measured = true;
    this.render(event.t - this.started);
  }
}

/** @typedef {{model?: string, where?: string}} Model */
/** @typedef {{mock: boolean, models: Record<string, Model>}} ModelConfig */

/** @param {string} id */
function modelElement(id) {
  const element = document.getElementById(id);

  if (!element) throw new Error(`Missing model card element: ${id}`);

  return element;
}

/** @param {string} model */
function modelIdentity(model) {
  const shortName = model.split('/').pop() || '';

  if (/nemotron/i.test(model)) {
    return {
      name: shortName.replace(/^NVIDIA-/, '').replace(/-NVFP4$/, '').replaceAll('-', ' '),
      brand: 'nvidia', vendor: 'NVIDIA',
    };
  }

  if (/smolvlm/i.test(model)) {
    return {name: shortName.replace(/-Instruct$/, '').replaceAll('-', ' '), brand: 'huggingface', vendor: 'Hugging Face'};
  }

  if (shortName === 'scout-r2-960.pt') {
    return {name: 'YOLOv8n · fine-tuned', brand: 'ultralytics', vendor: 'Ultralytics'};
  }

  if (/yolo/i.test(model)) {
    return {name: shortName.replace(/\.pt$/, '').replace(/^yolo/i, 'YOLO'), brand: 'ultralytics', vendor: 'Ultralytics'};
  }

  return {name: shortName || 'Model not configured', brand: '', vendor: ''};
}

/**
 * The Scout as /mode reports it (scout_backend), never from config alone: a
 * synthetic run reads scene truth, and "mock" or unknown backends fall back to
 * motion. The configured YOLO weights name the card only when YOLO really runs.
 * @param {string | null | undefined} backend @param {string} model
 */
function scoutIdentity(backend, model) {
  if (backend === 'yolo') {
    const identity = modelIdentity(model || 'yolo');

    const runtime = model.split('/').pop() === 'scout-r2-960.pt' ? 'scout-r2-960.pt · revision 2' : 'Configured model';

    return {...identity, tag: identity.name, runtime, sub: 'YOLO, every frame'};
  }

  const none = {brand: '', vendor: ''};

  if (backend === 'scene') {
    return {...none, name: 'Scene input', tag: 'camera input', runtime: 'Object + person context', sub: 'scene analysis, every frame'};
  }

  if (backend === 'motion') {
    return {...none, name: 'Motion detector', tag: 'motion', runtime: 'Background subtraction · no model', sub: 'motion, every frame'};
  }

  return {...none, name: 'Scout', tag: '', runtime: 'Detector not reported', sub: 'detector not reported'};
}

/** @param {string} role @param {{name: string, brand: string, vendor: string}} identity @param {string} title */
function showIdentity(role, identity, title) {
  const name = modelElement(`${role}Model`);
  const logo = /** @type {HTMLImageElement} */ (modelElement(`${role}Logo`));

  name.textContent = identity.name;
  name.title = title;
  logo.hidden = !identity.brand;

  if (identity.brand) {
    logo.src = `/pipeline/assets/brand-${identity.brand}.svg`;
    logo.alt = identity.vendor;
  }
}

class ModelCards {
  /** @param {ModelConfig} config */
  constructor(config) {
    this.roles = ['director', 'analyst', 'verifier'];
    this.scoutModel = config.models.scout?.model || '';

    for (const role of this.roles) {
      const model = config.models[role]?.model || '';

      showIdentity(role, modelIdentity(model), model);
      modelElement(`${role}Runtime`).textContent = 'Configured model';
    }

    this.scout(null);
    this.reset();
  }

  /** @param {string | null | undefined} backend what /mode reports as scout_backend */
  scout(backend) {
    const identity = scoutIdentity(backend, this.scoutModel);

    showIdentity('scout', identity, backend === 'yolo' ? this.scoutModel : '');
    modelElement('scoutRuntime').textContent = identity.runtime;
    modelElement('scoutSub').textContent = identity.sub;

    return identity;
  }

  reset() {
    for (const role of this.roles) {
      const image = /** @type {HTMLImageElement} */ (modelElement(`${role}Preview`));

      image.onload = null;
      image.hidden = true;
      modelElement(`${role}Empty`).textContent = role === 'director' ? 'No detection yet' : 'No crop yet';
      image.removeAttribute('src');
      modelElement(`${role}Empty`).hidden = false;
      modelElement(`${role}PreviewState`).textContent = role === 'director' ? 'Detection context' : 'Waiting for the crop';
      modelElement(`st-${role}`).dataset.evaluating = 'false';
    }
  }

  /** @param {string} role @param {string|null|undefined} source @param {string} label */
  restore(role, source, label) {
    const image = /** @type {HTMLImageElement} */ (modelElement(`${role}Preview`));

    image.onload = null;
    image.hidden = true;
    modelElement(`${role}Empty`).hidden = true;
    modelElement(`${role}PreviewState`).textContent = label;
    modelElement(`st-${role}`).dataset.evaluating = 'false';

    if (source) {
      image.onload = () => { image.hidden = false; };

      image.src = source;
    } else {
      image.removeAttribute('src');
      modelElement(`${role}Empty`).textContent = 'No saved crop';
      modelElement(`${role}Empty`).hidden = false;
    }
  }

  /** @param {string} role @param {string} source */
  show(role, source) {
    if (!this.roles.includes(role)) return;

    const image = /** @type {HTMLImageElement} */ (modelElement(`${role}Preview`));

    image.onload = () => {
      image.hidden = false;
      modelElement(`${role}Empty`).hidden = true;
    };

    image.src = source;

    modelElement(`${role}PreviewState`).textContent = role === 'director' ? 'Ticket only' : 'Reading the crop';
    modelElement(`st-${role}`).dataset.evaluating = 'true';
  }

  /** @param {string} role */
  complete(role) {
    modelElement(`st-${role}`).dataset.evaluating = 'false';
    modelElement(`${role}PreviewState`).textContent = role === 'director' ? 'Ticket only' : 'Read the crop';
  }
}
