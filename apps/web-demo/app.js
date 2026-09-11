import init, { create_renderer, generate_scene } from './web-demo.js';

let canvas = document.getElementById('triage-canvas');
let renderer;
let frame = 0;
let failed = false;
let previousTime = 0;
const pointers = new Map();
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const modes = ['RGB', 'relative depth', 'instance IDs', 'comparison'];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const recordingInput = document.getElementById('trajectory-file');
const recordingStatus = document.getElementById('trajectory-status');
const recordingError = document.getElementById('trajectory-error');
let recordingName = '';
let loadGeneration = 0;
let lastStatusTime = 0;

const sceneForm = document.getElementById('scene-form');
const seedInput = document.getElementById('seed');
const sampleInput = document.getElementById('sample-index');
const sceneStatus = document.getElementById('scene-status');
const sceneError = document.getElementById('scene-error');
const downloadButton = document.getElementById('download-scene');
const outputMode = document.getElementById('output-mode');
const fallback = document.getElementById('fallback');
const backendStatus = document.getElementById('backend-status');
let generatedJson = '';

function updateDetails(info) {
  outputMode.value = String(info.view.mode);
  downloadButton.disabled = info.playback.active;
  document.getElementById('calibration').textContent = JSON.stringify({
    generation: info.generation,
    ontology_version: info.ontology_version,
    camera: info.camera,
  }, null, 2);
  document.getElementById('camera-status').textContent = info.playback.active
    ? 'Trajectory camera active. Return to the generated scene to download its realized geometry and camera.'
    : `${info.camera.width} × ${info.camera.height} sensor pixels · ${info.camera.fov_vertical_degrees.toFixed(2)}° vertical field of view · ${info.camera.modified ? 'interacted camera (included in download)' : 'original calibrated recipe camera'}. Canvas resizing does not change sensor resolution.`;
  const legend = document.getElementById('instance-legend');
  legend.replaceChildren(...info.objects.map(object => {
    const item = document.createElement('li');
    if (object.color) {
      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.backgroundColor = object.color;
      swatch.setAttribute('aria-hidden', 'true');
      item.append(swatch);
    }
    item.append(document.createTextNode(`${object.id} · ${object.name}${object.class ? ` · ${object.class} (class ${object.semantic_id})` : ''}`));
    return item;
  }));
}

function generateSelected() {
  if (!sceneForm.reportValidity()) return;
  const seed = Number(seedInput.value), sample = Number(sampleInput.value);
  if (![seed, sample].every(value => Number.isInteger(value) && value >= 0 && value <= 4294967295)) {
    sceneError.textContent = 'Seed and sample index must be unsigned 32-bit integers.';
    return;
  }
  try {
    // Geometry, variation and camera are always generated in shared Rust, never JavaScript.
    const json = generate_scene(seed, sample);
    if (renderer && !failed) {
      renderer.load_generated_scene(seed, sample);
      updateDetails(JSON.parse(renderer.info()));
    } else {
      const scene = JSON.parse(json);
      document.getElementById('calibration').textContent = JSON.stringify({
        generation: scene.generation, sensor: scene.sensor,
      }, null, 2);
      document.getElementById('camera-status').textContent = 'GPU-independent generation: the download uses the original calibrated recipe camera. The static image remains seed 42 / sample 0.';
      document.getElementById('instance-legend').replaceChildren(...scene.objects.map(object => {
        const item = document.createElement('li');
        item.textContent = `${object.id} · ${object.name} · ${object.class}`;
        return item;
      }));
    }
    generatedJson = json;
    downloadButton.disabled = false;
    sceneStatus.textContent = `Shared recipe · seed ${seed} · sample ${sample}. ${renderer && !failed ? 'Scene and calibrated camera loaded.' : 'Realized JSON is ready; static preview remains seed 42 / sample 0.'}`;
    sceneError.textContent = '';
    previousTime = 0;
    requestRender();
  } catch (error) {
    sceneError.textContent = `Generation failed; previous scene retained. ${String(error?.message || error)}`;
  }
}

sceneForm.addEventListener('submit', event => { event.preventDefault(); generateSelected(); });
document.getElementById('next-sample').addEventListener('click', () => {
  if (!sceneForm.reportValidity()) return;
  const sample = Number(sampleInput.value);
  if (sample >= 4294967295) {
    sceneError.textContent = 'The maximum sample index is 4294967295. Choose another index or seed.';
    return;
  }
  sampleInput.value = String(sample + 1);
  generateSelected();
});
downloadButton.addEventListener('click', () => {
  try {
    const json = renderer && !failed ? renderer.scene_json() : generatedJson;
    if (!json) return;
    const { generation } = JSON.parse(json);
    const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `seed-${generation.recipe.seed}-sample-${generation.sample_index}.scene.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { sceneError.textContent = String(error?.message || error); }
});
outputMode.addEventListener('change', () => {
  if (!renderer || failed) return;
  renderer.set_mode(Number(outputMode.value));
  requestRender();
});
document.getElementById('reset-camera').addEventListener('click', () => {
  if (!renderer || failed) return;
  renderer.reset_view();
  requestRender();
});
document.getElementById('show-generated').addEventListener('click', () => {
  if (!renderer || failed) return;
  renderer.show_generated_scene();
  requestRender();
});
recordingInput.addEventListener('change', async () => {
  const file = recordingInput.files[0];
  const generation = ++loadGeneration;
  if (!file || !renderer || failed) return;
  try {
    const text = await file.text();
    if (generation !== loadGeneration) return;
    renderer.load_trajectory(text);
    recordingName = file.name;
    recordingError.textContent = '';
    previousTime = 0;
    canvas.focus();
    requestRender();
  } catch (error) {
    if (generation === loadGeneration) {
      recordingError.textContent = `Could not load ${file.name}: ${String(error?.message || error)}. Previous scene retained.`;
    }
  } finally {
    recordingInput.value = '';
  }
});

function fail(error) {
  failed = true;
  cancelAnimationFrame(frame);
  frame = 0;
  console.error(error);
  const message = String(error?.message || error);
  recordingInput.disabled = true;
  document.getElementById('view-controls').disabled = true;
  recordingStatus.textContent = `Interactive rendering unavailable: ${message}`;
  backendStatus.textContent = 'Static native preview · seed 42 / sample 0';
  canvas.hidden = true;
  fallback.hidden = false;
  document.getElementById('fallback-message').textContent = `WebGPU rendering is unavailable: ${message}. This is the native seed 42 / sample 0 preview, not a rendering of changed controls. The native capture link remains usable.`;
  if (generatedJson) generateSelected();
}

function requestRender() {
  if (renderer && !failed && !document.hidden && !frame) frame = requestAnimationFrame(render);
}

function render(time) {
  frame = 0;
  if (failed || document.hidden) return;
  const { width, height } = canvas.getBoundingClientRect();
  if (width < 1 || height < 1) { previousTime = 0; return; }
  const scale = Math.min(devicePixelRatio || 1, 2, 2048 / width, 2048 / height);
  const delta = previousTime ? (time - previousTime) / 1000 : 0;
  try {
    renderer.render(Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale)), scale, delta);
    if (time - lastStatusTime >= 250 || !previousTime || !renderer.is_animating()) {
      const info = JSON.parse(renderer.info());
      updateDetails(info);
      const playback = info.playback;
      const scene = playback.active ? `trajectory ${recordingName}` : info.scene_name;
      canvas.setAttribute('aria-label', `Triage — ${modes[renderer.mode()]} — ${scene}. ${canvas.title}`);
      recordingStatus.textContent = playback.active
        ? `${recordingName} · ${playback.playing ? 'Playing' : 'Paused'} · ${playback.time_seconds.toFixed(2)} / ${playback.end_seconds.toFixed(2)} s · ${playback.camera_mode} · environment ${playback.environment_id}`
        : 'Generated showcase · Load a sampled trajectory recording to replay. Space toggles auto-orbit.';
      lastStatusTime = time;
    }
  } catch (error) { fail(error); return; }
  previousTime = renderer.is_animating() ? time : 0;
  if (renderer.is_animating()) requestRender();
}

function backing(event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) * canvas.width / rect.width,
    (event.clientY - rect.top) * canvas.height / rect.height];
}

function blockGesture() {
  renderer.pointer_cancel();
  for (const pointer of pointers.values()) pointer.kind = 'blocked';
}

function separation() {
  const values = pointers.values();
  const a = values.next().value;
  const b = values.next().value;
  return b ? Math.hypot(a.x - b.x, a.y - b.y) : 0;
}

canvas.addEventListener('pointerdown', event => {
  if (!renderer || failed || event.button !== 0) return;
  event.preventDefault();
  canvas.focus();
  canvas.setPointerCapture(event.pointerId);
  const [x, y] = backing(event);
  let kind;
  if (!pointers.size) {
    kind = renderer.pointer_down(x, y) ? 'gui' : 'camera';
    if (kind === 'camera') renderer.orbit_by(0, 0);
  } else {
    const first = pointers.values().next().value;
    // Only two contacts that both began on the scene may pinch. Additional
    // contacts poison the whole gesture until every contact has lifted.
    kind = pointers.size === 1 && first.kind === 'camera' &&
      event.pointerType === 'touch' && first.touch && !renderer.pointer_move(x, y)
      ? 'camera' : 'blocked';
    if (kind === 'blocked') blockGesture();
  }
  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY, kind, touch: event.pointerType === 'touch' });
  requestRender();
});

canvas.addEventListener('pointermove', event => {
  const pointer = pointers.get(event.pointerId);
  if (!pointer || failed) return;
  const before = separation();
  const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
  pointer.x = event.clientX;
  pointer.y = event.clientY;
  if (pointer.kind === 'gui') renderer.pointer_move(...backing(event));
  else if (pointer.kind === 'camera') {
    if (pointers.size === 1) renderer.orbit_by(dx, dy);
    else {
      const after = separation();
      if (before > 0 && after > 0) renderer.zoom(before / after);
    }
  }
  requestRender();
});

for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  canvas.addEventListener(name, event => {
    const pointer = pointers.get(event.pointerId);
    if (!pointer || failed) return;
    if (name === 'pointerup' && pointer.kind === 'gui') renderer.pointer_up(...backing(event));
    else if (name !== 'pointerup') blockGesture();
    pointers.delete(event.pointerId);
    requestRender();
  });
}

canvas.addEventListener('wheel', event => {
  if (!renderer || failed) return;
  event.preventDefault();
  if (pointers.size) return;
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1;
  renderer.zoom(Math.exp(clamp(event.deltaY * unit * .001, -1, 1)));
  requestRender();
}, { passive: false });
canvas.addEventListener('contextmenu', event => {
  event.preventDefault();
  if (!renderer || failed || pointers.size) return;
  renderer.set_mode((renderer.mode() + 1) % 4);
  requestRender();
});
canvas.addEventListener('dblclick', event => {
  if (!renderer || failed || pointers.size || renderer.pointer_move(...backing(event))) return;
  renderer.reset_view();
  requestRender();
});
canvas.addEventListener('keydown', event => {
  if (!renderer || failed || !renderer.key(event.key)) return;
  event.preventDefault();
  requestRender();
});
new ResizeObserver(requestRender).observe(canvas);
window.addEventListener('resize', requestRender);
document.addEventListener('visibilitychange', () => {
  cancelAnimationFrame(frame);
  frame = 0;
  previousTime = 0;
  if (renderer && !failed) blockGesture();
  pointers.clear();
  requestRender();
});
reducedMotion.addEventListener('change', event => {
  if (event.matches && renderer && !failed && renderer.is_animating()) {
    renderer.pause();
    if (JSON.parse(renderer.info()).view.auto_orbit) renderer.key(' ');
    requestRender();
  }
});
try {
  await init();
  document.getElementById('controls').disabled = false;
  generateSelected();
  try {
    renderer = await create_renderer(canvas);
    const { generation } = JSON.parse(generatedJson);
    renderer.load_generated_scene(generation.recipe.seed, generation.sample_index);
    recordingInput.disabled = false;
    document.getElementById('view-controls').disabled = false;
    canvas.hidden = false;
    fallback.hidden = true;
    backendStatus.textContent = 'Live WebGPU · shared Rust generator';
    if (reducedMotion.matches && renderer.is_animating()) renderer.key(' ');
    requestRender();
  } catch (error) { fail(error); }
} catch (error) {
  fail(error);
  sceneError.textContent = 'The WebAssembly module could not load. Use the static native capture; scene generation and downloads require WebAssembly.';
}
