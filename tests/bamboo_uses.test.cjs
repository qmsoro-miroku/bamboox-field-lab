const {test} = require('node:test');
const assert = require('node:assert/strict');
const {recommendations} = require('../frontend/assets/bamboo-uses.js');
const result = (...stages) => ({detections: stages.map(class_name => ({class_name}))});

test('A mixed scan keeps every detected stage and its object count', () => {
  const items = recommendations(result('young', 'old', 'matured', 'young'));
  assert.deepEqual(items.map(({stage,count}) => ({stage,count})), [
    {stage:'young',count:2}, {stage:'matured',count:1}, {stage:'old',count:1}
  ]);
  assert.match(items[0].uses.join(' '), /woven mats/);
  assert.match(items[1].uses.join(' '), /Furniture/);
  assert.match(items[2].next, /decay/);
});
test('No detection produces no recommendation, even with stale aggregate counts', () => {
  recommendations(result('young'));
  for (const data of [null, {}, {detections: []}, {class_counts:{young:8}}, {detections:'young'}]) {
    assert.deepEqual(recommendations(data), []);
  }
});
test('Unknown labels never turn into material-use advice', () => {
  assert.deepEqual(recommendations(result('unknown', '__proto__', 'constructor')), []);
});
test('Names normalize without changing input or inventing an age measurement', () => {
  const data = result(' Mature ', 'YOUNG');
  const before = JSON.stringify(data);
  const items = recommendations(data);
  assert.deepEqual(items.map(item => item.stage), ['young','matured']);
  assert.equal(JSON.stringify(data), before);
  for (const item of items) assert.equal(Object.hasOwn(item, 'age'), false);
});
