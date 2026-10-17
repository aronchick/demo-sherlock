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

class ModelCards {
  /** @param {ModelConfig} config */
  constructor(config) {
    this.roles = ['director', 'analyst', 'verifier'];

    for (const role of ['scout', ...this.roles]) {
      const model = config.models[role]?.model || '';
      const identity = modelIdentity(model);
      const name = modelElement(`${role}Model`);
      const logo = /** @type {HTMLImageElement} */ (modelElement(`${role}Logo`));

      name.textContent = identity.name;
      name.title = model;
      logo.hidden = !identity.brand;

      if (identity.brand) {
        logo.src = `/pipeline/assets/brand-${identity.brand}.svg`;
        logo.alt = identity.vendor;
      }

      modelElement(`${role}Runtime`).textContent = config.mock ? 'Configured model · mock run' : 'Configured model';
    }

    this.reset();
  }

  reset() {
    for (const role of this.roles) {
      const image = /** @type {HTMLImageElement} */ (modelElement(`${role}Preview`));

      image.hidden = true;
      image.removeAttribute('src');
      modelElement(`${role}Empty`).hidden = false;
      modelElement(`${role}PreviewState`).textContent = role === 'director' ? 'Detection context' : 'Waiting for context';
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

    modelElement(`${role}PreviewState`).textContent = role === 'director' ? 'Ticket only' : 'Evaluating context';
    modelElement(`st-${role}`).dataset.evaluating = 'true';
  }

  /** @param {string} role */
  complete(role) {
    modelElement(`st-${role}`).dataset.evaluating = 'false';
    modelElement(`${role}PreviewState`).textContent = role === 'director' ? 'Ticket only' : 'Evaluated context';
  }
}
