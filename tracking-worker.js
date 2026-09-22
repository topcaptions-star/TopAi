import { parentPort, workerData } from 'node:worker_threads';
import { mkdtemp, readdir, rm, readFile, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';
import * as tf from '@tensorflow/tfjs-node';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import ffmpegPath from 'ffmpeg-static';
import sharp from 'sharp';

const job = parentPort ? workerData : JSON.parse(process.env.TOPAI_TRACKING_JOB || '{}');
const { action, videoPath, durationSeconds, sourceStart = 0, sampleFps = 3, maxFrames = 96, seed = null } = job;
const modelUrl = pathToFileURL(join(process.cwd(), 'models', 'coco-ssd', 'model.json')).href;
let modelPromise = null;

function clamp(value, minimum, maximum) { return Math.min(maximum, Math.max(minimum, value)); }
function rounded(value, digits = 6) { return Number(Number(value).toFixed(digits)); }
function publicError(message) { const error = new Error(message); error.publicMessage = message; return error; }
function run(command, args, timeoutMs = 180000) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    const timer = setTimeout(() => { child.kill('SIGKILL'); reject(publicError('Auto Center timed out while analyzing the video.')); }, timeoutMs);
    child.stderr.on('data', chunk => { stderr += chunk.toString(); });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('close', code => { clearTimeout(timer); code === 0 ? resolve(stderr) : reject(publicError(`Could not decode the selected video for Auto Center. ${stderr.slice(-300)}`)); });
  });
}
async function getModel() {
  if (!modelPromise) {
    modelPromise = (async () => {
      await tf.setBackend('cpu');
      await tf.ready();
      return cocoSsd.load({ base: 'lite_mobilenet_v2', modelUrl });
    })();
  }
  return modelPromise;
}
async function detectSubjects(framePath) {
  const buffer = await sharp(framePath).resize({ width: 640, withoutEnlargement: true }).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const image = tf.tensor3d(new Uint8Array(buffer.data), [buffer.info.height, buffer.info.width, buffer.info.channels], 'int32');
  try {
    const predictions = await (await getModel()).detect(image, 10, 0.2);
    return {
      width: buffer.info.width,
      height: buffer.info.height,
      subjects: predictions.filter(item => item.class === 'person').map(item => ({
        x: (item.bbox[0] + item.bbox[2] / 2) / buffer.info.width,
        y: (item.bbox[1] + item.bbox[3] / 2) / buffer.info.height,
        score: Number(item.score),
        box: { x: item.bbox[0] / buffer.info.width, y: item.bbox[1] / buffer.info.height, width: item.bbox[2] / buffer.info.width, height: item.bbox[3] / buffer.info.height }
      }))
    };
  } finally { image.dispose(); }
}
function selectSubject(subjects, selected, prior) {
  if (!subjects.length) return null;
  const target = prior || selected;
  if (!target) return subjects.sort((a, b) => (b.box.width * b.box.height) - (a.box.width * a.box.height))[0];
  return subjects.sort((a, b) => ((a.x - target.x) ** 2 + (a.y - target.y) ** 2) - ((b.x - target.x) ** 2 + (b.y - target.y) ** 2))[0];
}
function previewOffsets(duration) {
  const max = Math.max(0, duration - 0.08);
  const candidates = [0.25, duration * 0.15, duration * 0.32, duration * 0.5, duration * 0.68, duration * 0.85];
  const out = [];
  for (const candidate of candidates) {
    const offset = rounded(clamp(candidate, 0, max), 3);
    if (!out.some(value => Math.abs(value - offset) < 0.1)) out.push(offset);
  }
  return out;
}
async function preview() {
  const work = await mkdtemp(join(tmpdir(), 'topai-worker-preview-'));
  try {
    const offsets = previewOffsets(durationSeconds);
    for (let index = 0; index < offsets.length; index += 1) {
      const offset = offsets[index]; const framePath = join(work, `preview-${index}.jpg`);
      await run(ffmpegPath, ['-hide_banner', '-loglevel', 'error', '-ss', String(sourceStart + offset), '-i', videoPath, '-frames:v', '1', '-vf', 'scale=640:-2:force_original_aspect_ratio=decrease', '-q:v', '4', framePath]);
      try { await access(framePath); } catch { continue; }
      const result = await detectSubjects(framePath);
      if (!result.subjects.length) continue;
      const image = await readFile(framePath);
      return { imageBase64: image.toString('base64'), mime: 'image/jpeg', width: result.width, height: result.height, time: offset, sampleCount: offsets.length, scannedFrames: index + 1, detector: 'person-detector-preview', faces: result.subjects.map((subject, id) => ({ id, x: rounded(subject.x), y: rounded(subject.y), size: rounded(Math.max(subject.box.width, subject.box.height) * 0.45), box: { x: rounded(subject.box.x), y: rounded(subject.box.y), width: rounded(subject.box.width), height: rounded(subject.box.height) }, score: rounded(subject.score, 4) })) };
    }
    throw publicError('No person was detected in the Work Area.');
  } finally { await rm(work, { recursive: true, force: true }); }
}
function smooth(points) {
  const alpha = 0.42; let previous = null;
  return points.map(point => {
    if (!previous) { previous = { ...point }; return point; }
    previous = { ...point, x: rounded(previous.x + alpha * (point.x - previous.x)), y: rounded(previous.y + alpha * (point.y - previous.y)) };
    return previous;
  });
}
async function track() {
  const work = await mkdtemp(join(tmpdir(), 'topai-worker-track-'));
  try {
    const pattern = join(work, 'frame-%06d.jpg');
    await run(ffmpegPath, ['-hide_banner', '-loglevel', 'error', '-ss', String(sourceStart), '-t', String(durationSeconds), '-i', videoPath, '-vf', `fps=${sampleFps},scale=640:-2:force_original_aspect_ratio=decrease`, '-frames:v', String(maxFrames), '-q:v', '4', pattern]);
    const frames = (await readdir(work)).filter(file => file.endsWith('.jpg')).sort();
    if (!frames.length) throw publicError('No decodable video frames were found in the Work Area.');
    const points = []; let prior = null;
    for (let index = 0; index < frames.length; index += 1) {
      const result = await detectSubjects(join(work, frames[index]));
      const subject = selectSubject(result.subjects, seed, prior);
      if (!subject) continue;
      prior = subject;
      points.push({ time: rounded(index / sampleFps, 4), x: rounded(subject.x), y: rounded(subject.y), confidence: rounded(subject.score, 4) });
    }
    if (points.length < 2) throw publicError('No stable subject was detected in the selected Work Area.');
    const end = Math.min(durationSeconds, Math.max(points[points.length - 1].time + 1 / sampleFps, 1 / sampleFps));
    return { sampleFps, framesAnalyzed: frames.length, landmarksPerFace: 0, cuts: [], shots: [{ index: 0, start: 0, end: rounded(end, 4), points: smooth(points) }] };
  } finally { await rm(work, { recursive: true, force: true }); }
}

function send(message) { if (parentPort) parentPort.postMessage(message); else if (process.send) process.send(message); }
try {
  const data = action === 'preview' ? await preview() : await track();
  send({ ok: true, data });
} catch (error) {
  send({ ok: false, error: error.publicMessage || error.message || 'Auto Center worker failed.' });
} finally {
  if (!parentPort) setTimeout(() => process.exit(0), 10);
}
