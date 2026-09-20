// Visual encoder web demo — runs update() per frame via ONNX Runtime Web.
// Assets: model.onnx, frames.jpg + meta.json (sim), real.jpg + real.json (real FPV).

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.webgpu.min.mjs";

const status = document.getElementById("status");
window.onerror = (m) => { status.textContent = "error: " + m; };

// --- load both datasets ---
async function loadDataset(imgUrl, metaUrl) {
  const meta = await (await fetch(metaUrl)).json();
  const img = new Image();
  img.src = imgUrl;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return { meta, ctx };
}

const sim = await loadDataset("frames.jpg", "meta.json");
let real = null;
try { real = await loadDataset("real.jpg", "real.json"); } catch (_) {}

const PCA = sim.meta.pca; // shared projection so colors are comparable
const [LH, LW] = sim.meta.latent_hw, LC = sim.meta.latent_channels;
const TW = sim.meta.tile_w, TH = sim.meta.tile_h;

// --- ORT session ---
ort.env.wasm.numThreads = 1;
const session = await ort.InferenceSession.create("model.onnx", {
  executionProviders: ["webgpu", "wasm"],
});

// --- canvases ---
const inputCtx = document.getElementById("input").getContext("2d");
const latentCtx = document.getElementById("latent").getContext("2d");
const occCtx = document.getElementById("occ").getContext("2d");
const probeCtx = document.getElementById("probe").getContext("2d");
const latentImg = latentCtx.createImageData(LW, LH);
const occImg = occCtx.createImageData(256, 96);

// --- state ---
let h = new Float32Array(LC * LH * LW);
const frameTensor = new ort.Tensor("float32", new Float32Array(3 * TH * TW), [1, 3, TH, TW]);
const motionTensor = new ort.Tensor("float32", new Float32Array(5), [1, 5]);
const actionTensor = new ort.Tensor("float32", new Float32Array(2), [1, 2]);

let ds = sim, T = ds.meta.frames;
function framePixels(i) {
  const x = (i % ds.meta.cols) * TW, y = Math.floor(i / ds.meta.cols) * TH;
  return ds.ctx.getImageData(x, y, TW, TH);
}

async function runStep(i) {
  const px = framePixels(i).data;
  const fd = frameTensor.data;
  for (let c = 0; c < 3; c++)
    for (let p = 0; p < TH * TW; p++)
      fd[c * TH * TW + p] = px[p * 4 + c] / 255;
  motionTensor.data.set(ds.meta.motion[i]);
  actionTensor.data.set(ds.meta.actions[i]);

  const out = await session.run({
    frame: frameTensor,
    h: new ort.Tensor("float32", h, [1, LC, LH, LW]),
    motion: motionTensor,
    action: actionTensor,
  });
  h = out.h_new.data;
  draw(i, out);
}

function draw(i, out) {
  inputCtx.putImageData(framePixels(i), 0, 0);

  const f = out.f.data;
  const d = latentImg.data;
  for (let y = 0; y < LH; y++) {
    for (let x = 0; x < LW; x++) {
      let r = 0, g = 0, b = 0;
      for (let c = 0; c < LC; c++) {
        const v = f[c * LH * LW + y * LW + x];
        r += v * PCA[c][0]; g += v * PCA[c][1]; b += v * PCA[c][2];
      }
      const k = (y * LW + x) * 4;
      d[k] = Math.max(0, Math.min(255, r * 40 + 128));
      d[k + 1] = Math.max(0, Math.min(255, g * 40 + 128));
      d[k + 2] = Math.max(0, Math.min(255, b * 40 + 128));
      d[k + 3] = 255;
    }
  }
  latentCtx.putImageData(latentImg, 0, 0);

  // occupancy: top = pred, bottom = target (or black if none)
  const occ = out.occupancy.data, tgt = ds.meta.occ_target && ds.meta.occ_target[i];
  const od = occImg.data;
  for (let y = 0; y < 96; y++) {
    for (let x = 0; x < 256; x++) {
      const col = Math.floor(x / 8);
      const v = y < 48 ? 1 / (1 + Math.exp(-occ[col])) : (tgt ? tgt[col] : 0);
      const k = (y * 256 + x) * 4;
      od[k] = v * 255; od[k + 1] = v * 200; od[k + 2] = v * 80; od[k + 3] = 255;
    }
  }
  occCtx.putImageData(occImg, 0, 0);

  const p = out.probe.data, scale = [4.5, 10, 0.75, 0.75];
  probeCtx.fillStyle = "#000"; probeCtx.fillRect(0, 0, 384, 60);
  probeCtx.fillStyle = "#8f8";
  probeCtx.fillText(
    `gap_x ${(p[0] * scale[0]).toFixed(2)}  wall_d ${(p[1] * scale[1]).toFixed(2)}  vx ${(p[2] * scale[2]).toFixed(2)}  vz ${(p[3] * scale[3]).toFixed(2)}`,
    8, 34
  );
}

// --- playback + dataset toggle ---
let cur = 0, playing = true, busy = false;
const scrub = document.getElementById("scrub");
scrub.max = T - 1;
document.getElementById("play").onclick = (e) => {
  playing = !playing;
  e.target.textContent = playing ? "pause" : "play";
};
scrub.oninput = () => { cur = +scrub.value; };

const toggle = document.getElementById("dataset");
if (real) {
  toggle.style.display = "";
  toggle.onclick = () => {
    ds = ds === sim ? real : sim;
    T = ds.meta.frames;
    scrub.max = T - 1;
    cur = 0;
    h = new Float32Array(LC * LH * LW); // reset recurrent state
    toggle.textContent = ds === sim ? "sim → real" : "real → sim";
    status.textContent = ds === sim ? "sim rollout" : "real FPV footage";
  };
} else {
  toggle.style.display = "none";
}
status.textContent = `running — ${T} frames`;

async function loop() {
  if (playing && !busy) {
    busy = true;
    try {
      await runStep(cur);
      scrub.value = cur;
      cur = (cur + 1) % T;
    } catch (e) {
      status.textContent = "error: " + (e.message || e);
      playing = false;
    }
    busy = false;
  }
  requestAnimationFrame(loop);
}
loop();
