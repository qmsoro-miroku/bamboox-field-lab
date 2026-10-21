# Publication validation

- All eight Jinja pages render into static HTML and use the GitHub Pages project subpath.
- 48 page/viewport/theme combinations passed: desktop, short desktop, and phone widths, in dark and light modes.
- Home and Mission Hub fit 1536×740 and 1093×500 CSS viewports without clipping content or hiding overflow.
- Original trained models exported to ONNX; native numerical parity was verified against PyTorch.
- Real browser growth inference detected one object in a validation sample.
- Real browser species inference produced a nine-class model result and saved it to IndexedDB.
- Saved photo, sample ID, origin/location, notes, and prediction persist after reloading the browser page.
- Another isolated browser context sees an empty journal, confirming per-browser separation.
- Field Journal CSV generation and the actual browser download both passed.
- Dashboard and Settings read the visitor's own journal.
- All 7,050 public growth dataset image paths exist; real images, boxes, class filters, and split filters passed browser checks.
- Camera preview, capture, and stop passed using a synthetic browser test camera.
- No mocked inference API responses were used in the public-demo functional checks.
- No browser script errors or failed asset requests occurred in the final functional check.
- Original development project, inspection database, uploads, and trained checkpoints were kept unchanged.

These checks verify the deployment and application behavior; they do not establish new model accuracy or biological ground truth.
