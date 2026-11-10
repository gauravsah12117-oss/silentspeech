// Facial emotion and pain expression from the Face Landmarker's 52 muscle scores.
// A line-by-line port of src/emotion.py (the desktop app), so the browser demo reads
// faces exactly the way the app does. See that file for the reasoning behind each rule.
//
// FACS action units (Ekman & Friesen) are read from MediaPipe blendshapes; emotions from
// the EMFACS combinations; pain from the core of the Prkachin & Solomon Pain Intensity.

export const EMOTIONS = ["neutral", "happy", "sad", "surprised", "afraid", "angry", "disgusted"];
export const LABELS = { neutral: "Neutral", happy: "Happy", sad: "Sad", surprised: "Surprised",
  afraid: "Afraid", angry: "Angry", disgusted: "Disgusted" };
export const COLORS = { neutral: "#96acbe", happy: "#f5cd46", sad: "#468cdc", surprised: "#e66ec8",
  afraid: "#a064dc", angry: "#eb4b4b", disgusted: "#78b450" };

// AU -> [plain words, blendshapes averaged, score of a clear, strong movement]
export const ACTION_UNITS = {
  AU1: ["inner brows up", ["browInnerUp"], 0.55],
  AU2: ["outer brows up", ["browOuterUpLeft", "browOuterUpRight"], 0.45],
  AU4: ["brows down", ["browDownLeft", "browDownRight"], 0.45],
  AU5: ["eyes wide", ["eyeWideLeft", "eyeWideRight"], 0.35],
  AU6: ["cheeks raised", ["cheekSquintLeft", "cheekSquintRight"], 0.45],
  AU7: ["eyes narrowed", ["eyeSquintLeft", "eyeSquintRight"], 0.55],
  AU9: ["nose wrinkled", ["noseSneerLeft", "noseSneerRight"], 0.35],
  AU10: ["upper lip raised", ["mouthUpperUpLeft", "mouthUpperUpRight"], 0.40],
  AU12: ["smile", ["mouthSmileLeft", "mouthSmileRight"], 0.60],
  AU15: ["lip corners down", ["mouthFrownLeft", "mouthFrownRight"], 0.35],
  AU17: ["chin raised", ["mouthShrugLower"], 0.50],
  AU20: ["lips stretched", ["mouthStretchLeft", "mouthStretchRight"], 0.35],
  AU23: ["lips tightened", ["mouthPressLeft", "mouthPressRight"], 0.45],
  AU26: ["jaw dropped", ["jawOpen"], 0.50],
  AU43: ["eyes closed", ["eyeBlinkLeft", "eyeBlinkRight"], 0.65],
};
const AU_NAMES = Object.keys(ACTION_UNITS);
const MOUTH_AUS = new Set(["AU10", "AU12", "AU15", "AU17", "AU20", "AU23", "AU26"]);
const RULES = {
  happy: [["AU12"], ["AU6"]],
  sad: [["AU1", "AU15"], ["AU4", "AU17"]],
  surprised: [["AU1", "AU2"], ["AU5", "AU26"]],
  afraid: [["AU1", "AU4", "AU5"], ["AU2", "AU20", "AU26"]],
  angry: [["AU4", "AU7"], ["AU23", "AU5"]],
  disgusted: [["AU9"], ["AU10", "AU15", "AU17"]],
};
const TALKING_RULES = { happy: [["AU6"], ["AU12"]], surprised: [["AU1", "AU2", "AU5"], ["AU26"]] };
const TALK_WEIGHT = 0.3, NEUTRAL_LEVEL = 0.20, NEUTRAL_LEVEL_TALKING = 0.30, SHARPNESS = 16, EPS = 0.04;
const BASELINE_SECONDS = 60, BASELINE_PERCENTILE = 15, BASELINE_CAP = 0.7;
const FULL = AU_NAMES.map(a => ACTION_UNITS[a][2]);

const clip01 = x => Math.min(1, Math.max(0, x));

export function actionUnits(scores) {            // scores: {blendshapeName: 0..1}
  return AU_NAMES.map(a => {
    const names = ACTION_UNITS[a][1];
    return names.reduce((s, n) => s + (scores[n] || 0), 0) / names.length;
  });
}

export function softAnd(values, eps = EPS) {
  const logs = values.map(v => Math.log(clip01(v) + eps));
  return Math.exp(logs.reduce((s, v) => s + v, 0) / logs.length) - eps;
}

function evidence(au, talking) {
  const x = {};
  for (const a of AU_NAMES) x[a] = au[a] * (talking && MOUTH_AUS.has(a) ? TALK_WEIGHT : 1);
  return EMOTIONS.slice(1).map(emo => {
    const [core, support] = (talking && TALKING_RULES[emo]) || RULES[emo];
    const c = softAnd(core.map(a => x[a]));
    const s = support.length ? support.reduce((t, a) => t + x[a], 0) / support.length : 0;
    return Math.max(0, c) * (0.65 + 0.35 * s);
  });
}

function probabilities(ev, talking) {
  const logits = [talking ? NEUTRAL_LEVEL_TALKING : NEUTRAL_LEVEL, ...ev].map(v => SHARPNESS * v);
  const top = Math.max(...logits);
  const e = logits.map(v => Math.exp(v - top));
  const sum = e.reduce((s, v) => s + v, 0);
  return e.map(v => v / sum);
}

export function painScore(au, talking = false) {
  const smile = au.AU12;
  const levator = Math.max(au.AU9, au.AU10 * (1 - smile) * (talking ? TALK_WEIGHT : 1));
  const orbit = Math.max(au.AU7, au.AU6) * (1 - 0.5 * smile);
  const core = softAnd([au.AU4, orbit, levator], 0.02);
  return Math.min(1, Math.max(0, core) * (1 + 0.15 * au.AU43));
}

export function painLevel(score) {
  return score < 0.2 ? "none" : score < 0.35 ? "mild" : score < 0.5 ? "moderate" : "strong";
}

function percentile(values, p) {                  // numpy's default (linear) percentile
  const v = [...values].sort((a, b) => a - b);
  const pos = (v.length - 1) * p / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

export class EmotionReader {
  constructor(smoothS = 0.35, painSmoothS = 0.8) {
    this.smoothS = smoothS; this.painSmoothS = painSmoothS;
    this.samples = [];                              // [time, raw AUs] of resting frames
    this.baseline = FULL.map(f => 0.25 * f);
    this.learned = false;
    this.probs = probabilities(new Array(EMOTIONS.length - 1).fill(0), false);
    this.pain = 0; this.lastT = null;
    this.timeline = [];                             // [time, emotion index, confidence], 4 per second, 30 s
  }

  update(scores, now, talking = false) {         // now: seconds
    const raw = actionUnits(scores);
    const last = this.samples[this.samples.length - 1];
    if (!talking && (!last || now - last[0] >= 0.2)) {
      this.samples.push([now, raw]);
      while (now - this.samples[0][0] > BASELINE_SECONDS) this.samples.shift();
      if (this.samples.length >= 15) {
        this.baseline = AU_NAMES.map((_, i) =>
          Math.min(percentile(this.samples.map(s => s[1][i]), BASELINE_PERCENTILE), BASELINE_CAP * FULL[i]));
        this.learned = true;
      }
    }
    const au = {};
    AU_NAMES.forEach((a, i) => {
      au[a] = clip01((raw[i] - this.baseline[i]) / Math.max(FULL[i] - this.baseline[i], 0.1));
    });
    const target = probabilities(evidence(au, talking), talking);
    const painNow = painScore(au, talking);
    const dt = this.lastT === null ? 1 : Math.min(1, Math.max(0, now - this.lastT));
    const a = 1 - Math.exp(-dt / this.smoothS), ap = 1 - Math.exp(-dt / this.painSmoothS);
    this.probs = this.probs.map((p, i) => (1 - a) * p + a * target[i]);
    this.pain = (1 - ap) * this.pain + ap * painNow;
    this.lastT = now;
    let k = 0;
    this.probs.forEach((p, i) => { if (p > this.probs[k]) k = i; });
    const tl = this.timeline[this.timeline.length - 1];
    if (!tl || now - tl[0] >= 0.25) {
      this.timeline.push([now, k, this.probs[k]]);
      if (this.timeline.length > 120) this.timeline.shift();
    }
    const emotion = EMOTIONS[k];
    return { emotion, label: LABELS[emotion], confidence: this.probs[k], probs: this.probs, pain: this.pain,
      painLevel: painLevel(this.pain), drivers: drivers(emotion, au, talking), learning: !this.learned, au };
  }
}

function drivers(emotion, au, talking) {
  if (emotion === "neutral") return [];
  const [core, support] = (talking && TALKING_RULES[emotion]) || RULES[emotion];
  return [...core, ...support].filter(a => au[a] >= 0.15)
    .sort((a, b) => (au[b] - au[a]) || (core.includes(b) - core.includes(a)))
    .slice(0, 2).map(a => ACTION_UNITS[a][0]);
}
