import init, { create_renderer } from './web-demo.js';

let canvas = document.getElementById('triage-canvas');
let renderer;
let frame = 0;
let failed = false;
let previousTime = 0;
const pointers = new Map();
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const modes = ['RGB', 'relative depth', 'instance IDs', 'comparison'];
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function fail(error) {
  failed = true;
  cancelAnimationFrame(frame);
  console.error(error);
  const message = String(error?.message || error);
  const replacement = canvas.cloneNode(false);
  canvas.replaceWith(replacement);
  canvas = replacement;
  canvas.width = Math.max(320, canvas.clientWidth);
  canvas.height = Math.max(160, canvas.clientHeight);
  canvas.setAttribute('aria-label', `Renderer unavailable: ${message}`);
  canvas.title = message;
  const context = canvas.getContext('2d');
  context.fillStyle = '#141c24';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#f3f1ea';
  context.font = '16px sans-serif';
  context.fillText('Triage requires WebGPU on HTTPS or localhost.', 24, 42);
  context.font = '13px sans-serif';
  let line = '', y = 72;
  for (const word of message.split(/\s+/)) {
    if (context.measureText(`${line} ${word}`).width > canvas.width - 48) {
      context.fillText(line, 24, y);
      y += 20;
      line = word;
    } else line += `${line ? ' ' : ''}${word}`;
  }
  context.fillText(line, 24, y);
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
    canvas.setAttribute('aria-label', `Triage — ${modes[renderer.mode()]} — ${renderer.scene_index() ? 'calibration cubes' : 'courtyard'}. Canvas controls; keys 1–4 select representation, S switches scene, R resets, Space toggles orbit, arrows rotate, plus and minus zoom.`);
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
    renderer.key(' ');
    requestRender();
  }
});
try {
  await init();
  renderer = await create_renderer(canvas);
  if (reducedMotion.matches && renderer.is_animating()) renderer.key(' ');
  requestRender();
} catch (error) { fail(error); }
