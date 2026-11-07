'use strict';
const search = document.getElementById('search');
const level = document.getElementById('level');
if (search && level) {
  const rows = Array.from(document.querySelectorAll('#questions li'));
  const count = document.getElementById('result-count');
  const empty = document.getElementById('no-results');
  function filter() {
    const terms = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    for (const row of rows) {
      const matches = (!level.value || row.dataset.level === level.value) && terms.every(t => row.textContent.toLowerCase().includes(t));
      row.hidden = !matches;
      if (matches) shown++;
    }
    count.textContent = `${shown} of ${rows.length} questions`;
    empty.hidden = shown !== 0;
  }
  search.addEventListener('input', filter);
  level.addEventListener('change', filter);
}
