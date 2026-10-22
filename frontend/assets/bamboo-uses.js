/* Educational use guidance; growth labels are not measurements of culm age. */
(function (root) {
  'use strict';
  const guidance = {
    young: {
      name: 'Young', title: 'Flexible crafts & weaving',
      uses: ['Baskets and woven mats', 'Light handicrafts and decorative weaving'],
      next: 'Confirm the culm is developed enough for weaving. Leave new or tender culms growing when the goal is stronger, mature material.'
    },
    matured: {
      name: 'Matured', title: 'Furniture & finished products',
      uses: ['Furniture components and household crafts', 'Panels, screens and decorative products'],
      next: 'Confirm species, recorded age and condition; dry and treat the material appropriately. Any load-bearing use needs proper grading and qualified design review.'
    },
    old: {
      name: 'Old', title: 'Inspect, then repurpose',
      uses: ['Sound pieces for non-load-bearing crafts and decor', 'Pulp or biomass processing where the material is suitable'],
      next: 'Check for decay, insect damage and deep splits first. Separate damaged culms; an Old prediction alone cannot determine whether a piece is usable.'
    }
  };
  function recommendations(data) {
    const counts = {young: 0, matured: 0, old: 0};
    // Use the current frame detections, never a previous frame or the largest count only.
    for (const detection of Array.isArray(data?.detections) ? data.detections : []) {
      let stage = String(detection.class_name || '').trim().toLowerCase();
      if (stage === 'mature') stage = 'matured';
      if (Object.hasOwn(counts, stage)) counts[stage]++;
    }
    return Object.keys(counts).filter(stage => counts[stage] > 0)
      .map(stage => ({stage, count: counts[stage], ...guidance[stage]}));
  }
  function render(container, data, context = 'This image') {
    if (!container) return;
    const cards = container.querySelector('[data-use-cards]');
    cards.replaceChildren();
    const items = recommendations(data);
    container.hidden = items.length === 0;
    if (!items.length) return;
    container.querySelector('[data-use-context]').textContent =
      context + ' · ' + items.map(item => `${item.count} ${item.name}`).join(' / ');
    const doc = container.ownerDocument;
    function node(tag, className, text) {
      const el = doc.createElement(tag); el.className = className;
      if (text) el.textContent = text;
      return el;
    }
    for (const item of items) {
      const card = node('article', 'bu-card'); card.dataset.stage = item.stage;
      card.append(node('span', 'bu-stage', item.name), node('h4', '', item.title));
      const uses = node('ul', 'bu-use-list');
      for (const use of item.uses) uses.append(node('li', '', use));
      card.append(uses, node('p', 'bu-next', item.next)); cards.append(card);
    }
  }
  const api = Object.freeze({recommendations, render});
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.BambooUses = api;
})(typeof window === 'object' ? window : globalThis);
