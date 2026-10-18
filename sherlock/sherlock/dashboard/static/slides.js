(() => {
  const slides = Array.from(document.querySelectorAll('main > section'));
  const previous = document.querySelector('#previous');
  const next = document.querySelector('#next');
  const status = document.querySelector('#slide-status');

  if (!(previous instanceof HTMLButtonElement) || !(next instanceof HTMLButtonElement) || !status) return;
  let current = 0;

  const showSlide = () => {
    const selected = slides.findIndex(slide => `#${slide.id}` === location.hash);
    current = selected < 0 ? 0 : selected;
    slides.forEach((slide, index) => { if (slide instanceof HTMLElement) slide.hidden = index !== current; });
    previous.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    status.textContent = `${String(current + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}`;
  };

  /** @param {number} delta */
  function moveSlide(delta) {
    const index = Math.max(0, Math.min(slides.length - 1, current + delta));
    const slide = slides[index];

    if (slide) location.hash = slide.id;
  }

  previous.addEventListener('click', () => moveSlide(-1));
  next.addEventListener('click', () => moveSlide(1));
  document.addEventListener('keydown', event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;

    if (event.target instanceof Element && event.target.closest('a, button, input, textarea, select, [contenteditable]')) return;

    if (event.key === 'ArrowRight' || event.key === ' ') { event.preventDefault(); moveSlide(1); }
    else if (event.key === 'ArrowLeft') { event.preventDefault(); moveSlide(-1); }
  });
  window.addEventListener('hashchange', showSlide);
  showSlide();
})();
