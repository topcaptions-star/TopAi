import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { mkdtemp, readdir, rm, unlink, writeFile, readFile, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { openAsBlob } from 'node:fs';
import { BatchClient } from '@speechmatics/batch-client';
import * as tf from '@tensorflow/tfjs-node';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import * as faceDetection from '@tensorflow-models/face-detection';
import ffmpegPath from 'ffmpeg-static';
import sharp from 'sharp';

const app = express();
const port = Number(process.env.PORT || 10000);
const maxMb = Number(process.env.MAX_UPLOAD_MB || 250);
const maxTrackSeconds = Number(process.env.MAX_TRACK_SECONDS || 60);
const maxTrackFrames = clamp(Number(process.env.MAX_TRACK_FRAMES || 96), 12, 96);
const processTimeoutMs = Number(process.env.TRACK_TIMEOUT_MS || 240000);
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: maxMb * 1024 * 1024 } });
let detectorPromise = null;
let fallbackDetectorPromise = null;

app.disable('x-powered-by');
app.use(cors({ origin: process.env.ALLOWED_ORIGIN || '*' }));
app.use((req, _res, next) => { console.log(JSON.stringify({ event: 'request', method: req.method, path: req.path })); next(); });
app.get('/health', (_req, res) => res.json({ ok: true, service: 'topai-api', tracking: { enabled: true, engine: 'topai-facemesh-face-detector-autofacecenter', landmarksPerFace: 478, maxTrackSeconds, maxTrackFrames, workflow: 'multi-preview-select-shot-aware-center' } }));
app.get('/healthz', (_req, res) => res.json({ ok: true, service: 'topai-api' }));

function safeName(name = 'media') { return String(name).replace(/[^a-zA-Z0-9._-]/g, '_'); }
function publicError(error, fallback) { return error?.publicMessage || fallback; }
function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
function delaySafeNumber(value, fallback) { const n = Number(value); return Number.isFinite(n) ? n : fallback; }
function rounded(n, digits = 6) { return Number(Number(n).toFixed(digits)); }

function run(command, args, timeoutMs = processTimeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    const timer = setTimeout(() => { child.kill('SIGKILL'); reject(Object.assign(new Error('Tracking process timed out'), { publicMessage: 'Face tracking timed out. Try a shorter Work Area.' })); }, timeoutMs);
    child.stderr.on('data', chunk => { stderr += chunk.toString(); if (stderr.length > 12000) stderr = stderr.slice(-12000); });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('close', code => { clearTimeout(timer); code === 0 ? resolve(stderr) : reject(Object.assign(new Error(`ffmpeg exited ${code}: ${stderr}`), { publicMessage: 'Could not decode the selected video for face tracking.' })); });
  });
}

async function getDetector() {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      await tf.setBackend('cpu');
      await tf.ready();
      return faceLandmarksDetection.createDetector(faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, { runtime: 'tfjs', maxFaces: 5, refineLandmarks: true });
    })();
  }
  return detectorPromise;
}
async function getFallbackDetector() {
  if (!fallbackDetectorPromise) {
    fallbackDetectorPromise = (async () => {
      await tf.setBackend('cpu');
      await tf.ready();
      return faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, { runtime: 'tfjs', maxFaces: 5, modelType: 'short' });
    })();
  }
  return fallbackDetectorPromise;
}

function faceArea(face) { return Math.max(0, face.box.width) * Math.max(0, face.box.height); }
function faceCenter(face) { return { x: face.box.xMin + face.box.width / 2, y: face.box.yMin + face.box.height / 2 }; }
function landmark(face, index) { return face.keypoints?.[index] || null; }
function normalizeFace(face, width, height, index) {
  const nose = landmark(face, 1) || faceCenter(face);
  const leftEye = landmark(face, 33); const rightEye = landmark(face, 263);
  const eyeDistance = leftEye && rightEye ? Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y) : Math.max(face.box.width, face.box.height) * 0.45;
  return {
    id: index,
    x: rounded(nose.x / width),
    y: rounded(nose.y / height),
    size: rounded(eyeDistance / width),
    box: { x: rounded(face.box.xMin / width), y: rounded(face.box.yMin / height), width: rounded(face.box.width / width), height: rounded(face.box.height / height) },
    score: face.score == null ? null : rounded(Array.isArray(face.score) ? face.score[0] : face.score, 4)
  };
}
function chooseFace(faces, seed, previous, width, height) {
  if (!faces?.length) return null;
  const normalized = faces.map((face, index) => ({ face, candidate: normalizeFace(face, width, height, index) }));
  if (seed) return normalized.sort((a, b) => ((a.candidate.x - seed.x) ** 2 + (a.candidate.y - seed.y) ** 2) - ((b.candidate.x - seed.x) ** 2 + (b.candidate.y - seed.y) ** 2))[0];
  if (previous) return normalized.sort((a, b) => ((a.candidate.x - previous.x) ** 2 + (a.candidate.y - previous.y) ** 2) - ((b.candidate.x - previous.x) ** 2 + (b.candidate.y - previous.y) ** 2))[0];
  return normalized.sort((a, b) => faceArea(b.face) - faceArea(a.face))[0];
}
function smoothCenterPoints(points) {
  const alpha = 0.38;
  let prior = null;
  return points.map(point => {
    if (!prior) { prior = { ...point }; return point; }
    const next = { ...point, x: rounded(prior.x + alpha * (point.x - prior.x)), y: rounded(prior.y + alpha * (point.y - prior.y)) };
    prior = next;
    return next;
  });
}
function parseShowInfoTimes(stderr) {
  const hits = []; const regex = /pts_time:([0-9.]+)/g; let match;
  while ((match = regex.exec(stderr))) hits.push(Number(match[1]));
  return hits.filter(Number.isFinite);
}
function normalizeCuts(cuts, duration) {
  const minShotSeconds = 0.85;
  const ordered = [...cuts].filter(t => t > minShotSeconds && t < duration - minShotSeconds).sort((a, b) => a - b);
  const output = []; let last = 0;
  for (const cut of ordered) {
    if (cut - last >= minShotSeconds && duration - cut >= minShotSeconds) { output.push(rounded(cut, 3)); last = cut; }
  }
  return output;
}
async function detectCuts(videoPath, durationSeconds, sourceStart = 0) {
  if (!ffmpegPath) return [];
  const stderr = await run(ffmpegPath, ['-hide_banner', '-loglevel', 'info', '-ss', String(sourceStart), '-t', String(durationSeconds), '-i', videoPath, '-vf', "setpts=PTS-STARTPTS,select='gt(scene,0.32)',showinfo", '-an', '-f', 'null', '-']);
  return normalizeCuts(parseShowInfoTimes(stderr), durationSeconds);
}
function shotsFromCuts(cuts, duration) {
  const edges = [0, ...cuts, duration]; const shots = [];
  for (let index = 0; index < edges.length - 1; index += 1) {
    const start = edges[index]; const end = edges[index + 1];
    if (end - start >= 0.5) shots.push({ index, start: rounded(start, 4), end: rounded(end, 4) });
  }
  return shots;
}
async function extractFrames(videoPath, workingDir, sampleFps, maxFrames, durationSeconds, sourceStart = 0) {
  const pattern = join(workingDir, 'frame-%06d.jpg');
  await run(ffmpegPath, ['-hide_banner', '-loglevel', 'error', '-ss', String(sourceStart), '-t', String(durationSeconds), '-i', videoPath, '-vf', `fps=${sampleFps},scale=960:-2:force_original_aspect_ratio=decrease`, '-frames:v', String(maxFrames), '-q:v', '4', pattern]);
  return (await readdir(workingDir)).filter(file => file.endsWith('.jpg')).sort();
}
async function imageFaces(path) {
  const buffer = await sharp(path).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const image = tf.tensor3d(new Uint8Array(buffer.data), [buffer.info.height, buffer.info.width, buffer.info.channels], 'int32');
  try {
    let faces = await (await getDetector()).estimateFaces(image, { flipHorizontal: false });
    let engine = 'facemesh';
    if (!faces.length) { faces = await (await getFallbackDetector()).estimateFaces(image, { flipHorizontal: false }); engine = 'face-detector-fallback'; }
    return { faces, width: buffer.info.width, height: buffer.info.height, engine };
  } finally { image.dispose(); }
}
async function autoFaceCenter(videoPath, sampleFps, maxFrames, durationSeconds, seed, sourceStart = 0) {
  if (!ffmpegPath) throw Object.assign(new Error('ffmpeg-static unavailable'), { publicMessage: 'Face tracking engine is unavailable on the server.' });
  const workingDir = await mkdtemp(join(tmpdir(), 'topai-center-'));
  try {
    const frameFiles = await extractFrames(videoPath, workingDir, sampleFps, maxFrames, durationSeconds, sourceStart);
    if (!frameFiles.length) throw Object.assign(new Error('No frames extracted'), { publicMessage: 'The uploaded clip did not contain decodable video frames.' });
    const cuts = await detectCuts(videoPath, durationSeconds, sourceStart).catch(error => { console.warn(JSON.stringify({ event: 'cut_detect_warning', error: error.message })); return []; });
    const shots = shotsFromCuts(cuts, durationSeconds);
    const rawPoints = []; let prior = null;
    for (let index = 0; index < frameFiles.length; index += 1) {
      const { faces, width, height } = await imageFaces(join(workingDir, frameFiles[index]));
      const pick = chooseFace(faces, prior ? null : seed, prior, width, height);
      if (!pick) continue;
      prior = pick.candidate;
      rawPoints.push({ time: rounded(index / sampleFps, 4), x: pick.candidate.x, y: pick.candidate.y, confidence: pick.candidate.score });
    }
    if (rawPoints.length < 2) throw Object.assign(new Error('No face detected'), { publicMessage: 'No stable face was detected in the selected clip. Choose a clearer face or shorten the Work Area.' });
    const points = smoothCenterPoints(rawPoints);
    const enrichedShots = shots.map(shot => ({ ...shot, points: points.filter(point => point.time >= shot.start - 1e-4 && point.time <= shot.end + 1e-4) })).filter(shot => shot.points.length >= 2);
    if (!enrichedShots.length) throw Object.assign(new Error('No trackable shots'), { publicMessage: 'Face tracking data did not cover a usable shot.' });
    return { sampleFps, framesAnalyzed: frameFiles.length, landmarksPerFace: 478, cuts, shots: enrichedShots };
  } finally { await rm(workingDir, { recursive: true, force: true }); }
}
function previewOffsets(durationSeconds) {
  const duration = Math.max(0.1, Number(durationSeconds) || 0.1);
  const values = [0.4, duration * 0.15, duration * 0.32, duration * 0.5, duration * 0.68, duration * 0.85];
  const output = [];
  for (const value of values) {
    const offset = rounded(clamp(value, 0, Math.max(0, duration - 0.08)), 3);
    if (!output.some(existing => Math.abs(existing - offset) < 0.15)) output.push(offset);
  }
  return output;
}
async function previewFaces(videoPath, durationSeconds, sourceStart = 0) {
  if (!ffmpegPath) throw Object.assign(new Error('ffmpeg-static unavailable'), { publicMessage: 'Face tracking engine is unavailable on the server.' });
  const workingDir = await mkdtemp(join(tmpdir(), 'topai-preview-'));
  try {
    let best = null;
    const offsets = previewOffsets(durationSeconds);
    for (let index = 0; index < offsets.length; index += 1) {
      const offset = offsets[index]; const framePath = join(workingDir, `preview-${index}.jpg`);
      await run(ffmpegPath, ['-hide_banner', '-loglevel', 'error', '-ss', String(sourceStart + offset), '-i', videoPath, '-frames:v', '1', '-vf', 'scale=960:-2:force_original_aspect_ratio=decrease', '-q:v', '4', framePath]);
      try { await access(framePath); } catch (_missingFrame) { continue; }
      const faceData = await imageFaces(framePath);
      if (!faceData.faces.length) continue;
      const largestArea = Math.max(...faceData.faces.map(faceArea));
      if (!best || largestArea > best.largestArea) best = { framePath, faceData, offset, largestArea };
    }
    if (!best) throw Object.assign(new Error('No face found in preview samples'), { publicMessage: 'No face was found across six points in the Work Area. Move the Work Area to include a clear front-facing face, then retry.' });
    const image = await readFile(best.framePath);
    return { imageBase64: image.toString('base64'), mime: 'image/jpeg', width: best.faceData.width, height: best.faceData.height, time: best.offset, sampleCount: offsets.length, detector: best.faceData.engine, faces: best.faceData.faces.map((face, index) => normalizeFace(face, best.faceData.width, best.faceData.height, index)) };
  } finally { await rm(workingDir, { recursive: true, force: true }); }
}

function normalizeWords(response) {
  const results = response?.results || [];
  return results.filter(r => r.type === 'word' || r.type === 'punctuation').map(r => {
    const alt = r.alternatives?.[0] || {};
    return { text: alt.content || r.content || '', start: Number(r.start_time ?? 0), end: Number(r.end_time ?? r.start_time ?? 0), confidence: alt.confidence == null ? null : Number(alt.confidence) };
  });
}
function makeSegments(words, maxChars = 32) {
  const out = []; let current = null;
  for (const word of words) {
    if (!word.text) continue;
    const candidate = current ? `${current.text} ${word.text}` : word.text;
    if (current && candidate.length > maxChars) { out.push(current); current = null; }
    if (!current) current = { text: word.text, start: word.start, end: word.end, words: [word] };
    else { current.text += ` ${word.text}`; current.end = word.end; current.words.push(word); }
    if (/[.!?؟。！？]$/.test(word.text)) { out.push(current); current = null; }
  }
  if (current) out.push(current);
  return out;
}

app.post('/transcribe', upload.single('media'), async (req, res) => {
  const requestId = randomUUID();
  console.log(JSON.stringify({ event: 'transcribe_received', requestId, hasFile: Boolean(req.file), bytes: req.file?.size || 0 }));
  if (!process.env.SPEECHMATICS_API_KEY) return res.status(500).json({ error: 'SPEECHMATICS_API_KEY is not configured on the server' });
  if (!req.file) return res.status(400).json({ error: 'Send an audio/video file in multipart field: media' });
  const language = String(req.body.language || 'auto');
  const model = ['standard', 'enhanced', 'melia-1'].includes(req.body.model) ? req.body.model : 'melia-1';
  const maxChars = Math.min(80, Math.max(10, Number(req.body.maxChars || 32)));
  const mediaPath = join(tmpdir(), `topai-${randomUUID()}-${safeName(req.file.originalname)}`);
  try {
    await writeFile(mediaPath, req.file.buffer);
    const apiKey = String(process.env.SPEECHMATICS_API_KEY || '').trim();
    const apiUrl = process.env.SPEECHMATICS_API_URL || 'https://eu1.asr.api.speechmatics.com';
    const authCheck = await fetch(`${apiUrl}/v2/jobs?sm-app=topai-captions`, { headers: { Authorization: `Bearer ${apiKey}` } });
    if (!authCheck.ok) throw Object.assign(new Error(`Speechmatics auth HTTP ${authCheck.status}`), { publicMessage: 'Speechmatics authentication failed on the server.' });
    const client = new BatchClient({ apiKey, appId: 'topai-captions', apiUrl });
    const blob = await openAsBlob(mediaPath);
    const file = new File([blob], req.file.originalname || 'media');
    const transcriptionConfig = model === 'melia-1' ? { language: 'multi', model } : { language: language === 'auto' ? 'en' : language, model };
    const response = await client.transcribe(file, { transcription_config: transcriptionConfig }, 'json-v2');
    const words = normalizeWords(response);
    console.log(JSON.stringify({ event: 'transcribe_done', requestId, words: words.length }));
    return res.json({ language: response?.metadata?.transcription_config?.language || language, model, words, segments: makeSegments(words, maxChars), source: 'speechmatics-json-v2' });
  } catch (error) {
    console.error(JSON.stringify({ event: 'transcribe_error', requestId, error: error.message }));
    return res.status(502).json({ error: 'Speechmatics transcription failed', detail: publicError(error, 'Speechmatics transcription failed') });
  } finally { await unlink(mediaPath).catch(() => {}); }
});

app.post('/track-preview', upload.single('media'), async (req, res) => {
  const requestId = randomUUID();
  if (!req.file) return res.status(400).json({ error: 'Send a video file in multipart field: media' });
  const durationSeconds = clamp(delaySafeNumber(req.body.durationSeconds, maxTrackSeconds), 1, maxTrackSeconds);
  const sourceStart = Math.max(0, delaySafeNumber(req.body.sourceStart, 0));
  const mediaPath = join(tmpdir(), `topai-preview-source-${requestId}-${safeName(req.file.originalname)}`);
  try {
    console.log(JSON.stringify({ event: 'preview_received', requestId, bytes: req.file.size, durationSeconds, sourceStart }));
    await writeFile(mediaPath, req.file.buffer);
    const preview = await previewFaces(mediaPath, durationSeconds, sourceStart);
    console.log(JSON.stringify({ event: 'preview_done', requestId, faces: preview.faces.length }));
    return res.json({ ok: true, preview, engine: 'topai-mediapipe-facemesh-478' });
  } catch (error) {
    const status = error?.publicMessage?.startsWith('No stable') ? 422 : 502;
    console.error(JSON.stringify({ event: 'preview_error', requestId, error: error.message }));
    return res.status(status).json({ error: publicError(error, 'Could not prepare face selection.') });
  } finally { await unlink(mediaPath).catch(() => {}); }
});

app.post('/auto-face-center', upload.single('media'), async (req, res) => {
  const requestId = randomUUID(); const started = Date.now();
  if (!req.file) return res.status(400).json({ error: 'Send a video file in multipart field: media' });
  const requestedFps = clamp(delaySafeNumber(req.body.sampleFps, 6), 2, 12);
  const maxFrames = clamp(Math.round(delaySafeNumber(req.body.maxFrames, maxTrackFrames)), 12, maxTrackFrames);
  const durationSeconds = clamp(delaySafeNumber(req.body.durationSeconds, maxTrackSeconds), 1, maxTrackSeconds);
  const sampleFps = Math.max(1, Math.min(requestedFps, maxFrames / durationSeconds));
  const sourceStart = Math.max(0, delaySafeNumber(req.body.sourceStart, 0));
  let seed = null;
  if (req.body.seedX !== undefined && req.body.seedY !== undefined) seed = { x: clamp(delaySafeNumber(req.body.seedX, 0.5), 0, 1), y: clamp(delaySafeNumber(req.body.seedY, 0.5), 0, 1) };
  const mediaPath = join(tmpdir(), `topai-center-source-${requestId}-${safeName(req.file.originalname)}`);
  try {
    console.log(JSON.stringify({ event: 'center_received', requestId, bytes: req.file.size, requestedFps, sampleFps, maxFrames, durationSeconds, sourceStart, hasSeed: Boolean(seed) }));
    await writeFile(mediaPath, req.file.buffer);
    const tracked = await autoFaceCenter(mediaPath, sampleFps, maxFrames, durationSeconds, seed, sourceStart);
    console.log(JSON.stringify({ event: 'center_done', requestId, shots: tracked.shots.length, cuts: tracked.cuts.length, points: tracked.shots.reduce((count, shot) => count + shot.points.length, 0), durationMs: Date.now() - started }));
    return res.json({ ok: true, tracking: { ...tracked, engine: 'topai-mediapipe-autofacecenter', coordinateSpace: 'normalized-video-frame', smoothing: 'ema-0.38', composition: 'shot-aware-reframe' } });
  } catch (error) {
    const status = error?.publicMessage?.startsWith('No') ? 422 : 502;
    console.error(JSON.stringify({ event: 'center_error', requestId, error: error.message }));
    return res.status(status).json({ error: publicError(error, 'Auto Face Center failed') });
  } finally { await unlink(mediaPath).catch(() => {}); }
});

app.use((err, _req, res, _next) => {
  if (err?.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ error: `File exceeds ${maxMb}MB limit` });
  console.error(JSON.stringify({ event: 'http_error', error: err?.message || String(err) }));
  return res.status(500).json({ error: 'Server error' });
});

app.listen(port, '0.0.0.0', () => console.log(`TopAi API listening on ${port}`));
