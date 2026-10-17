// @ts-check

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

    return {...identity, tag: identity.name, runtime: 'Configured model', sub: 'YOLO, every frame'};
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

      image.hidden = true;
      image.removeAttribute('src');
      modelElement(`${role}Empty`).hidden = false;
      modelElement(`${role}PreviewState`).textContent = role === 'director' ? 'Detection context' : 'Waiting for the crop';
      modelElement(`st-${role}`).dataset.evaluating = 'false';
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
