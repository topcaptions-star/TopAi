import { parentPort, workerData } from 'node:worker_threads';
import { mkdtemp, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import cvModule from '@techstark/opencv-js';
import ffmpegPath from 'ffmpeg-static';
import sharp from 'sharp';

const job = parentPort ? workerData : JSON.parse(process.env.TOPAI_TRACKING_JOB || '{}');
const { videoPath, durationSeconds, sourceStart = 0, sampleFps = 6, maxFrames = 96, seed = null } = job;
let cvPromise = null;

function clamp(value, minimum, maximum) { return Math.min(maximum, Math.max(minimum, value)); }
function rounded(value, digits = 6) { return Number(Number(value).toFixed(digits)); }
function publicError(message) { const error = new Error(message); error.publicMessage = message; return error; }
function median(values) { if (!values.length) return 0; const ordered = values.slice().sort((a, b) => a - b); return ordered[Math.floor(ordered.length / 2)]; }
function run(command, args, timeoutMs = 90000) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'ignore', 'pipe'] }); let stderr = '';
    const timer = setTimeout(() => { child.kill('SIGKILL'); reject(publicError('Auto Center timed out while reading the video.')); }, timeoutMs);
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('close', code => { clearTimeout(timer); code === 0 ? resolve() : reject(publicError(`Could not decode the selected video for Auto Center. ${stderr.slice(-300)}`)); });
  });
}
async function getCv() {
  if (!cvPromise) cvPromise = (async () => {
    if (cvModule instanceof Promise) return await cvModule;
    if (cvModule.Mat) return cvModule;
    return await new Promise(resolve => { cvModule.onRuntimeInitialized = () => resolve(cvModule); });
  })();
  return cvPromise;
}
async function extractFrames(cv) {
  const work = await mkdtemp(join(tmpdir(), 'topai-optical-track-'));
  const rate = clamp(Number(sampleFps) || 6, 3, 6);
  const budget = clamp(Math.round(Number(maxFrames) || 96), 12, 96);
  await run(ffmpegPath, ['-hide_banner', '-loglevel', 'error', '-ss', String(sourceStart), '-t', String(durationSeconds), '-i', videoPath, '-vf', `fps=${rate},scale=480:-2:force_original_aspect_ratio=decrease`, '-frames:v', String(budget), '-q:v', '4', join(work, 'frame-%06d.jpg')]);
  const files = (await readdir(work)).filter(file => /\.jpg$/i.test(file)).sort();
  if (files.length < 2) { await rm(work, { recursive: true, force: true }); throw publicError('The selected Work Area needs at least two decodable video frames.'); }
  const frames = [];
  try {
    for (const file of files) {
      const raw = await sharp(join(work, file)).resize({ width: 480, withoutEnlargement: true }).grayscale().raw().toBuffer({ resolveWithObject: true });
      frames.push(cv.matFromArray(raw.info.height, raw.info.width, cv.CV_8UC1, raw.data));
    }
    return { work, frames, width: frames[0].cols, height: frames[0].rows, rate };
  } catch (error) { frames.forEach(frame => frame.delete()); await rm(work, { recursive: true, force: true }); throw error; }
}
function makeGrid(cv, width, height, center, box) {
  const halfW = clamp((Number(box?.width) || 0.28) * width * 0.42, 18, width * 0.26);
  const halfH = clamp((Number(box?.height) || 0.42) * height * 0.38, 18, height * 0.30);
  const points = [];
  for (let row = -2; row <= 2; row += 1) for (let col = -2; col <= 2; col += 1) {
    points.push([clamp(center.x + col * halfW / 2.4, 6, width - 7), clamp(center.y + row * halfH / 2.4, 6, height - 7)]);
  }
  const mat = new cv.Mat(1, points.length, cv.CV_32FC2);
  points.forEach((point, index) => { mat.data32F[index * 2] = point[0]; mat.data32F[index * 2 + 1] = point[1]; });
  return mat;
}
function opticalStep(cv, previous, next, active) {
  const output = new cv.Mat(); const status = new cv.Mat(); const errors = new cv.Mat();
  try {
    cv.calcOpticalFlowPyrLK(previous, next, active, output, status, errors, new cv.Size(21, 21), 3, new cv.TermCriteria(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 20, 0.03), 0, 0.0001);
    const dx = []; const dy = []; const values = [];
    for (let index = 0; index < active.cols; index += 1) {
      if (!status.data[index]) continue;
      const fromX = active.data32F[index * 2]; const fromY = active.data32F[index * 2 + 1];
      const toX = output.data32F[index * 2]; const toY = output.data32F[index * 2 + 1];
      const moveX = toX - fromX; const moveY = toY - fromY;
      if (!Number.isFinite(moveX) || !Number.isFinite(moveY) || Math.abs(moveX) > 45 || Math.abs(moveY) > 45) continue;
      dx.push(moveX); dy.push(moveY); values.push([toX, toY]);
    }
    if (values.length < 5) return null;
    const nextActive = new cv.Mat(1, values.length, cv.CV_32FC2);
    values.forEach((point, index) => { nextActive.data32F[index * 2] = point[0]; nextActive.data32F[index * 2 + 1] = point[1]; });
    return { active: nextActive, dx: median(dx), dy: median(dy), usable: values.length };
  } finally { output.delete(); status.delete(); errors.delete(); }
}
function follow(cv, frames, fromIndex, direction, startCenter, startPoints, width, height, box) {
  const tracked = {}; let center = { ...startCenter }; let active = startPoints.clone(); tracked[fromIndex] = { ...center, usable: active.cols };
  for (let index = fromIndex; ; index += direction) {
    const nextIndex = index + direction; if (nextIndex < 0 || nextIndex >= frames.length) break;
    const step = opticalStep(cv, frames[index], frames[nextIndex], active); active.delete();
    if (!step) { active = makeGrid(cv, width, height, center, box); tracked[nextIndex] = { ...center, usable: 0 }; continue; }
    center = { x: clamp(center.x + step.dx, 0, width), y: clamp(center.y + step.dy, 0, height) };
    active = step.active; tracked[nextIndex] = { ...center, usable: step.usable };
  }
  active.delete(); return tracked;
}
function stabilize(points, width, height) {
  const output = []; let previous = null;
  for (const point of points) {
    if (!previous) { previous = { ...point }; output.push(point); continue; }
    const maxJumpX = width * 0.12; const maxJumpY = height * 0.12;
    const x = Math.abs(point.px - previous.px) > maxJumpX ? previous.px : point.px;
    const y = Math.abs(point.py - previous.py) > maxJumpY ? previous.py : point.py;
    const next = { ...point, px: x, py: y }; previous = next; output.push(next);
  }
  return output;
}
async function track() {
  if (!seed || !Number.isFinite(Number(seed.x)) || !Number.isFinite(Number(seed.y))) throw publicError('Choose a subject in the preview before Auto Center starts.');
  const cv = await getCv(); const extracted = await extractFrames(cv);
  try {
    const { frames, width, height, rate } = extracted;
    const previewTime = clamp(Number(seed.time) || 0, 0, Math.max(0, Number(durationSeconds) || 0));
    const pivot = clamp(Math.round(previewTime * rate), 0, frames.length - 1);
    const center = { x: clamp(Number(seed.x), 0, 1) * width, y: clamp(Number(seed.y), 0, 1) * height };
    const box = seed.box || { width: Number(seed.width) || 0.28, height: Number(seed.height) || 0.42 };
    const grid = makeGrid(cv, width, height, center, box);
    const forward = follow(cv, frames, pivot, 1, center, grid, width, height, box);
    const backward = follow(cv, frames, pivot, -1, center, grid, width, height, box);
    grid.delete();
    const raw = [];
    for (let index = 0; index < frames.length; index += 1) {
      const value = forward[index] || backward[index];
      if (value) raw.push({ time: rounded(index / rate, 4), px: value.x, py: value.y, confidence: value.usable ? 1 : 0.5 });
    }
    if (raw.length < 2) throw publicError('The selected subject did not contain enough trackable visual detail. Choose a clearer subject and retry.');
    const points = stabilize(raw, width, height).map(point => ({ time: point.time, x: rounded(point.px / width), y: rounded(point.py / height), confidence: point.confidence }));
    return { sampleFps: rate, framesAnalyzed: frames.length, landmarksPerFace: 0, cuts: [], shots: [{ index: 0, start: 0, end: rounded(Math.min(Number(durationSeconds), (frames.length - 1) / rate + 1 / rate), 4), points }], tracker: 'lucas-kanade-optical-flow', pivotTime: rounded(pivot / rate, 4) };
  } finally { extracted.frames.forEach(frame => frame.delete()); await rm(extracted.work, { recursive: true, force: true }); }
}
function send(message) { if (parentPort) parentPort.postMessage(message); else if (process.send) process.send(message); }
try { send({ ok: true, data: await track() }); } catch (error) { send({ ok: false, error: error.publicMessage || error.message || 'Auto Center worker failed.' }); } finally { if (!parentPort) setTimeout(() => process.exit(0), 10); }
