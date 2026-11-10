// Live browser demo: the same Face Landmarker model and emotion rules as the desktop app,
// running entirely in the visitor's browser. No video or image ever leaves the device.
import { FaceLandmarker, FilesetResolver, DrawingUtils } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";
import { EmotionReader, COLORS } from "./emotion.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const MUSCLES = [
  ["Jaw open", ["jawOpen"]], ["Pucker", ["mouthPucker"]], ["Lips press", ["mouthPressLeft", "mouthPressRight"]],
  ["Smile", ["mouthSmileLeft", "mouthSmileRight"]], ["Brows up", ["browInnerUp", "browOuterUpLeft", "browOuterUpRight"]],
  ["Brows down", ["browDownLeft", "browDownRight"]], ["Eyes narrow", ["eyeSquintLeft", "eyeSquintRight"]],
  ["Nose wrinkle", ["noseSneerLeft", "noseSneerRight"]],
];
const FACES = { neutral: "😐", happy: "😊", sad: "😢", surprised: "😮", afraid: "😨", angry: "😠", disgusted: "🤢", pain: "😣" };
const PAIN_STEPS = [0.12, 0.2, 0.35, 0.5, 0.65];

const $ = id => document.getElementById(id);
const els = {
  start: $("demo-start"), stop: $("demo-stop"), view: $("demo-view"), status: $("demo-status"),
  video: $("demo-video"), canvas: $("demo-canvas"), fps: $("demo-fps"),
  face: $("emo-face"), label: $("emo-label"), conf: $("emo-conf"), why: $("emo-why"),
  timeline: $("emo-timeline"), painLevel: $("pain-level"), painSegs: [...document.querySelectorAll("#pain-meter span")],
  muscles: $("demo-muscles"), signal: $("lip-signal"), panel: $("demo-panel"),
};
let landmarker = null, stream = null, running = false, reader = null, drawer = null;
let lastVideoTime = -1, frames = 0, fpsT = performance.now(), lastUi = 0, showCamera = true;
const mouthHist = [];

// muscle rows
const bars = MUSCLES.map(([label]) => {
  const row = document.createElement("div");
  row.className = "muscle";
  row.innerHTML = `<span>${label}</span><i><b></b></i>`;
  els.muscles.appendChild(row);
  return row.querySelector("b");
});

function status(text, kind = "") {
  els.status.textContent = text;
  els.status.dataset.kind = kind;
}

async function loadModel() {
  // About 13 MB the first time (the vision engine and the face model); the browser keeps it after that.
  const t0 = performance.now();
  const msg = () => status(`Loading the face model (about 13 MB the first time)... ${Math.round((performance.now() - t0) / 1000)} s`);
  msg();
  const timer = setInterval(msg, 1000);
  try {
    const files = await FilesetResolver.forVisionTasks(WASM);
    const options = delegate => ({ baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: "VIDEO",
      numFaces: 1, outputFaceBlendshapes: true });
    try {
      return await FaceLandmarker.createFromOptions(files, options("GPU"));
    } catch (e) {
      return await FaceLandmarker.createFromOptions(files, options("CPU"));
    }
  } finally {
    clearInterval(timer);
  }
}

async function start() {
  els.start.disabled = true;
  try {
    landmarker = landmarker || await loadModel();
    status("Asking for your camera...");
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 },
      facingMode: "user" }, audio: false });
    els.video.srcObject = stream;
    await els.video.play();
    reader = new EmotionReader();
    running = true;
    els.panel.classList.add("live");
    els.stop.hidden = false;
    els.start.hidden = true;
    status("Learning your resting face for 3 seconds - sit naturally, then try a big smile, raised eyebrows, a frown.", "ok");
    requestAnimationFrame(loop);
  } catch (e) {
    els.start.disabled = false;
    const name = e && e.name;
    status(name === "NotAllowedError" ? "The camera was blocked. Allow the camera for this page (the icon in the address bar) and try again."
      : name === "NotFoundError" ? "No camera was found on this device."
      : name === "NotReadableError" ? "The camera is busy in another app. Close it there and try again."
      : `Could not start the demo: ${e && e.message ? e.message : e}`, "error");
  }
}

function stop() {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null;
  els.panel.classList.remove("live");
  els.stop.hidden = true;
  els.start.hidden = false;
  els.start.disabled = false;
  status("Camera off. Nothing was recorded or sent anywhere.");
}

let frameErrors = 0;
function loop() {
  if (!running) return;
  requestAnimationFrame(loop);                     // first, so one bad frame can never freeze the demo
  const v = els.video;
  if (v.readyState >= 2 && v.currentTime !== lastVideoTime) {
    lastVideoTime = v.currentTime;
    const now = performance.now();
    try {
      render(landmarker.detectForVideo(v, now), now / 1000);
    } catch (e) {
      if (frameErrors++ === 0) console.warn("Skipped a frame:", e);
      return;
    }
    frames++;
    if (now - fpsT > 1000) {
      els.fps.textContent = `${(frames * 1000 / (now - fpsT)).toFixed(0)} FPS`;
      frames = 0; fpsT = now;
    }
  }
}

function render(res, t) {
  const c = els.canvas, v = els.video, ctx = c.getContext("2d");
  if (!drawer || c.width !== v.videoWidth || c.height !== v.videoHeight) {
    c.width = v.videoWidth; c.height = v.videoHeight;
    drawer = new DrawingUtils(ctx);
  }
  const w = c.width, h = c.height;
  ctx.save();
  ctx.translate(w, 0); ctx.scale(-1, 1);                       // mirror, like a mirror
  ctx.fillStyle = "#0b0e12"; ctx.fillRect(0, 0, w, h);
  if (showCamera) { ctx.globalAlpha = 0.42; ctx.drawImage(v, 0, 0, w, h); ctx.globalAlpha = 1; }
  ctx.strokeStyle = "rgba(40,180,190,0.10)"; ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 32) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  for (let y = 0; y < h; y += 32) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
  const face = res.faceLandmarks && res.faceLandmarks[0];
  let box = null;
  if (face) {
    drawer.drawConnectors(face, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "rgba(60,215,225,0.42)", lineWidth: 0.6 });
    drawer.drawConnectors(face, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#7dffa0", lineWidth: 1.8 });
    drawer.drawConnectors(face, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: "#7dffa0", lineWidth: 1 });
    drawer.drawConnectors(face, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: "#7dffa0", lineWidth: 1 });
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const p of face) { x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
    box = [x0 * w - 12, y0 * h - 12, x1 * w + 12, y1 * h + 12];
    ctx.strokeStyle = "#5af0a0"; ctx.lineWidth = 2;
    const s = Math.max(12, (box[2] - box[0]) / 8);
    for (const [cx, cy, dx, dy] of [[box[0], box[1], 1, 1], [box[2], box[1], -1, 1], [box[0], box[3], 1, -1], [box[2], box[3], -1, -1]]) {
      ctx.beginPath(); ctx.moveTo(cx + dx * s, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * s); ctx.stroke();
    }
  }
  ctx.restore();

  const shapes = res.faceBlendshapes && res.faceBlendshapes[0];
  if (!face || !shapes) {
    if (t * 1000 - lastUi > 250) { lastUi = t * 1000; status("No face found - face the camera, with light in front of you.", "warn"); }
    return;
  }
  const scores = {};
  for (const cat of shapes.categories) scores[cat.categoryName] = cat.score;
  const r = reader.update(scores, t);
  const strongPain = r.painLevel === "strong";
  const key = strongPain ? "pain" : r.emotion;
  const color = strongPain ? "#eb4b4b" : COLORS[r.emotion];
  const shown = strongPain ? r.pain : r.confidence;

  // tag next to the face (in mirrored coordinates)
  if (box) {
    const tx = Math.max(6, w - box[2]);
    ctx.font = "600 15px Inter, system-ui, sans-serif";
    ctx.fillStyle = color;
    ctx.fillText(`${strongPain ? "IN PAIN" : r.label.toUpperCase()} ${Math.round(shown * 100)}%`, tx, Math.max(18, box[1] - 8));
    if (r.painLevel !== "none") {
      ctx.fillStyle = "#fab42d";
      ctx.fillText(`PAIN EXPRESSION: ${r.painLevel.toUpperCase()}`, tx, Math.min(h - 10, box[3] + 22));
    }
  }

  // mouth opening, last 3 seconds
  mouthHist.push([t, (scores.jawOpen || 0) + 0.5 * ((scores.mouthLowerDownLeft || 0) + (scores.mouthLowerDownRight || 0)) / 2]);
  while (mouthHist.length && t - mouthHist[0][0] > 3) mouthHist.shift();
  drawSignal(t);
  drawTimeline(t);
  MUSCLES.forEach(([, names], i) => {
    const v = names.reduce((s, n) => s + (scores[n] || 0), 0) / names.length;
    bars[i].style.width = `${Math.min(100, v * 100).toFixed(1)}%`;
  });

  if (t * 1000 - lastUi > 120) {                      // text: about 8 times a second is plenty
    lastUi = t * 1000;
    els.face.textContent = FACES[key];
    els.face.style.borderColor = color;
    els.label.textContent = strongPain ? "In pain" : r.label;
    els.label.style.color = color;
    els.conf.textContent = `${Math.round(shown * 100)}%`;
    els.why.textContent = r.learning ? "learning your resting face..." : (r.drivers.join(" · ") || "relaxed face");
    const lit = PAIN_STEPS.filter(s => r.pain >= s).length;
    els.painSegs.forEach((seg, i) => seg.classList.toggle("on", i < lit));
    els.painLevel.textContent = r.painLevel;
    els.painLevel.dataset.level = r.painLevel;
    if (!r.learning && els.status.dataset.kind !== "live") status("Live. Try a big smile, raised eyebrows, a frown, or wrinkle your nose.", "live");
  }
}

function drawSignal(t) {
  const c = els.signal, ctx = c.getContext("2d"), w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#7dffa0"; ctx.lineWidth = 2; ctx.beginPath();
  mouthHist.forEach(([tt, m], i) => {
    const x = (tt - t + 3) / 3 * (w - 4) + 2, y = h - 4 - Math.min(1, m / 0.6) * (h - 8);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

function drawTimeline(t) {
  const c = els.timeline, ctx = c.getContext("2d"), w = c.width, h = c.height;
  ctx.fillStyle = "#262c35"; ctx.fillRect(0, 0, w, h);
  const tl = reader.timeline;
  for (let i = 0; i < tl.length; i++) {
    const [tt, k, conf] = tl[i];
    const x0 = (tt - t + 30) / 30 * w, x1 = i + 1 < tl.length ? (tl[i + 1][0] - t + 30) / 30 * w : w;
    if (x1 < 0) continue;
    ctx.globalAlpha = 0.35 + 0.65 * Math.min(1, conf);
    ctx.fillStyle = Object.values(COLORS)[k];
    ctx.fillRect(Math.max(0, x0), 0, Math.max(1, x1 - Math.max(0, x0)), h);
  }
  ctx.globalAlpha = 1;
}

els.start.addEventListener("click", start);
els.stop.addEventListener("click", stop);
els.view.addEventListener("change", () => { showCamera = els.view.checked; });
if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
  els.start.disabled = true;
  status("This browser can't use the camera here. Try Chrome, Edge or Safari.", "error");
}
