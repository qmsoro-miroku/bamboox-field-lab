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

## Camera and rectangular preprocessing correction

- The original camera detected a captured still only. Live mode now repeatedly scans new frames with one inference request in flight, explicit start/stop, and view/generation checks that reject late results.
- Controller regression tests cover serial scheduling, blank-frame box removal, stopping during inference, capture changes, capture scans without a running camera, recording, hidden tabs, and camera shutdown. These tests use DOM/API fixtures; model parity tests use the trained weights.
- Five validation photos were compared with the original checkpoint. With the previous fixed square browser input, one photo returned zero objects while the native rectangular pipeline detected Old at 43.96%. Dynamic rectangular ONNX inference recovers its 43.96% raw score without changing the 25% confidence threshold.
- The reproducible export passed numerical parity on all six square/rectangular input shapes, including landscape 320×640.
- A crop of the user-supplied screen capture returned no detections in both native YOLO (rectangular and square) and the previous ONNX export. Its strongest square-input score was approximately 0.8%. Screenshot pixels are not the original captured image; this is diagnostic evidence, not a ground-truth evaluation.
- Model weights and class mappings are unchanged. These corrections do not establish new model accuracy on field photographs or re-filmed screens.

- Real inference on the deployed GitHub Pages website repeatedly detected an Old object in explicitly labeled dataset-sample camera mode (50.3% in the browser). Stop cleared the overlay, and a captured-frame scan also detected one Old object. No physical webcam was accessed for this test.
- Returning from captured-frame detection to live mode is covered by a regression test after an observer timing issue was found and fixed during public testing.

- The final public version also passed captured-frame detection followed by resuming multiple automatic live frames. The reported screenshot crop returned zero objects in the deployed browser detector as well as native YOLO.

## Image Scanner use recommendations (22 October 2026)

- Suggested uses appear only in Image Scanner after a growth-stage detection; camera and species flows remain unchanged.
- Four unit tests cover all stages and mixed counts, zero/unknown labels, normalization, stale aggregates, and absence of invented age measurements. The three existing camera controller tests also pass.
- Browser UI fixtures verified Young/Matured/Old content, mixed stages, three viewport sizes in light/dark themes, species-tab separation, upload replacement, no detections, API errors, and rejection of a late response after a new upload.
- A separate clean browser ran the real ONNX growth model on a public validation image: one Young object was detected and the Young guidance appeared. This checks the detection-to-recommendation flow, not model accuracy or material suitability.
- Both local installations render all eight Jinja templates with StrictUndefined. Recommendations are educational stage-based rules, not a second AI model or an exact-age estimate, and are not saved to the journal.
