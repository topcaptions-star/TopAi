import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { mkdtemp, readdir, rm, unlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { openAsBlob } from 'node:fs';
import { BatchClient } from '@speechmatics/batch-client';
import * as tf from '@tensorflow/tfjs-node';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
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

app.disable('x-powered-by');
app.use(cors({ origin: process.env.ALLOWED_ORIGIN || '*' }));
app.use((req, _res, next) => { console.log(JSON.stringify({ event: 'request', method: req.method, path: req.path })); next(); });
app.get('/health', (_req, res) => res.json({ ok: true, service: 'topai-api', tracking: { enabled: true, engine: 'topai-mediapipe-facemesh-478', landmarksPerFace: 478, maxTrackSeconds, maxTrackFrames } }));
app.get('/healthz', (_req, res) => res.json({ ok: true, service: 'topai-api' }));

function safeName(name = 'media') { return String(name).replace(/[^a-zA-Z0-9._-]/g, '_'); }
function publicError(error, fallback) { return error?.publicMessage || fallback; }
function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
function delaySafeNumber(value, fallback) { const n = Number(value); return Number.isFinite(n) ? n : fallback; }

function run(command, args, timeoutMs = processTimeoutMs) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    const timer = setTimeout(() => { child.kill('SIGKILL'); reject(Object.assign(new Error('Tracking process timed out'), { publicMessage: 'Face tracking timed out. Try a shorter Work Area.' })); }, timeoutMs);
    child.stderr.on('data', chunk => { stderr += chunk.toString(); if (stderr.length > 4000) stderr = stderr.slice(-4000); });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('close', code => { clearTimeout(timer); code === 0 ? resolve() : reject(Object.assign(new Error(`ffmpeg exited ${code}: ${stderr}`), { publicMessage: 'Could not decode the selected video for face tracking.' })); });
  });
}

async function getDetector() {
  if (!detectorPromise) {
    detectorPromise = (async () => {
      await tf.setBackend('cpu');
      await tf.ready();
      return faceLandmarksDetection.createDetector(faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, { runtime: 'tfjs', maxFaces: 1, refineLandmarks: true });
    })();
  }
  return detectorPromise;
}

function faceArea(face) { return Math.max(0, face.box.width) * Math.max(0, face.box.height); }
function faceCenter(face) { return { x: face.box.xMin + face.box.width / 2, y: face.box.yMin + face.box.height / 2 }; }
function landmark(face, index) { return face.keypoints?.[index] || null; }
function normalizedPoint(point, width, height) { return point ? { x: Number((point.x / width).toFixed(6)), y: Number((point.y / height).toFixed(6)), z: Number((point.z || 0).toFixed(6)) } : null; }
function selectFace(faces, prior, width, height) {
  if (!faces?.length) return null;
  if (!prior) return [...faces].sort((a, b) => faceArea(b) - faceArea(a))[0];
  return [...faces].sort((a, b) => {
    const ca = faceCenter(a); const cb = faceCenter(b);
    const da = ((ca.x / width) - prior.x) ** 2 + ((ca.y / height) - prior.y) ** 2;
    const db = ((cb.x / width) - prior.x) ** 2 + ((cb.y / height) - prior.y) ** 2;
    return da - db;
  })[0];
}
function smoothPoints(points) {
  const alpha = 0.62;
  let prior = null;
  return points.map(point => {
    if (!prior) { prior = { ...point }; return point; }
    const smoothFeature = (current, previous) => !current ? null : !previous ? current : { x: previous.x + alpha * (current.x - previous.x), y: previous.y + alpha * (current.y - previous.y), z: previous.z + alpha * (current.z - previous.z) };
    const smoothed = { ...point, x: prior.x + alpha * (point.x - prior.x), y: prior.y + alpha * (point.y - prior.y), scale: prior.scale + alpha * (point.scale - prior.scale), rotation: prior.rotation + alpha * (point.rotation - prior.rotation), features: { leftEye: smoothFeature(point.features.leftEye, prior.features.leftEye), rightEye: smoothFeature(point.features.rightEye, prior.features.rightEye), nose: smoothFeature(point.features.nose, prior.features.nose), mouth: smoothFeature(point.features.mouth, prior.features.mouth) } };
    prior = smoothed;
    return smoothed;
  });
}

async function trackFace(videoPath, sampleFps, maxFrames, maxSeconds) {
  if (!ffmpegPath) throw Object.assign(new Error('ffmpeg-static unavailable'), { publicMessage: 'Face tracking engine is unavailable on the server.' });
  const workingDir = await mkdtemp(join(tmpdir(), 'topai-track-'));
  try {
    const pattern = join(workingDir, 'frame-%06d.jpg');
    await run(ffmpegPath, ['-hide_banner', '-loglevel', 'error', '-t', String(maxSeconds), '-i', videoPath, '-vf', `fps=${sampleFps},scale=512:-2:force_original_aspect_ratio=decrease`, '-frames:v', String(maxFrames), '-q:v', '4', pattern]);
    const frameFiles = (await readdir(workingDir)).filter(file => file.endsWith('.jpg')).sort();
    if (!frameFiles.length) throw Object.assign(new Error('No frames extracted'), { publicMessage: 'The uploaded clip did not contain decodable video frames.' });
    const detector = await getDetector();
    const points = []; let prior = null;
    for (let index = 0; index < frameFiles.length; index += 1) {
      const buffer = await sharp(join(workingDir, frameFiles[index])).removeAlpha().raw().toBuffer({ resolveWithObject: true });
      const image = tf.tensor3d(new Uint8Array(buffer.data), [buffer.info.height, buffer.info.width, buffer.info.channels], 'int32');
      let faces;
      try { faces = await detector.estimateFaces(image, { flipHorizontal: false }); } finally { image.dispose(); }
      const face = selectFace(faces, prior, buffer.info.width, buffer.info.height);
      if (!face) continue;
      const leftEye = landmark(face, 33); const rightEye = landmark(face, 263); const nose = landmark(face, 1); const mouth = landmark(face, 13);
      if (!leftEye || !rightEye || !nose || !mouth) continue;
      const center = { x: nose.x, y: nose.y };
      let rotation = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180 / Math.PI;
      while (rotation > 90) rotation -= 180;
      while (rotation < -90) rotation += 180;
      const rawConfidence = Array.isArray(face.score) ? face.score[0] : face.score;
      const point = { time: Number((index / sampleFps).toFixed(4)), x: Number((center.x / buffer.info.width).toFixed(6)), y: Number((center.y / buffer.info.height).toFixed(6)), scale: Number((Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y) / buffer.info.width).toFixed(6)), rotation: Number(rotation.toFixed(4)), confidence: rawConfidence == null ? null : Number(rawConfidence.toFixed(4)), features: { leftEye: normalizedPoint(leftEye, buffer.info.width, buffer.info.height), rightEye: normalizedPoint(rightEye, buffer.info.width, buffer.info.height), nose: normalizedPoint(nose, buffer.info.width, buffer.info.height), mouth: normalizedPoint(mouth, buffer.info.width, buffer.info.height) } };
      prior = point; points.push(point);
    }
    if (!points.length) throw Object.assign(new Error('No face detected'), { publicMessage: 'No face was detected in the selected clip. Use a clearer front-facing shot or a shorter Work Area.' });
    return { sampleFps, framesAnalyzed: frameFiles.length, landmarksPerFace: 478, points: smoothPoints(points) };
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

app.post('/track-face', upload.single('media'), async (req, res) => {
  const requestId = randomUUID();
  const started = Date.now();
  if (!req.file) return res.status(400).json({ error: 'Send a video file in multipart field: media' });
  const requestedFps = clamp(delaySafeNumber(req.body.sampleFps, 4), 2, 12);
  const maxFrames = clamp(Math.round(delaySafeNumber(req.body.maxFrames, maxTrackFrames)), 12, maxTrackFrames);
  const durationSeconds = clamp(delaySafeNumber(req.body.durationSeconds, maxTrackSeconds), 1, maxTrackSeconds);
  const sampleFps = Math.max(1, Math.min(requestedFps, maxFrames / durationSeconds));
  const mediaPath = join(tmpdir(), `topai-track-source-${requestId}-${safeName(req.file.originalname)}`);
  try {
    console.log(JSON.stringify({ event: 'track_received', requestId, bytes: req.file.size, requestedFps, sampleFps, maxFrames, durationSeconds }));
    await writeFile(mediaPath, req.file.buffer);
    const tracked = await trackFace(mediaPath, sampleFps, maxFrames, durationSeconds);
    console.log(JSON.stringify({ event: 'track_done', requestId, points: tracked.points.length, durationMs: Date.now() - started }));
    return res.json({ ok: true, tracking: { ...tracked, engine: 'topai-mediapipe-facemesh-478', coordinateSpace: 'normalized-video-frame', smoothing: 'ema-0.62' } });
  } catch (error) {
    const status = error?.publicMessage?.startsWith('No face') ? 422 : 502;
    console.error(JSON.stringify({ event: 'track_error', requestId, error: error.message }));
    return res.status(status).json({ error: publicError(error, 'Face tracking failed') });
  } finally { await unlink(mediaPath).catch(() => {}); }
});

app.use((err, _req, res, _next) => {
  if (err?.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ error: `File exceeds ${maxMb}MB limit` });
  console.error(JSON.stringify({ event: 'http_error', error: err?.message || String(err) }));
  return res.status(500).json({ error: 'Server error' });
});

app.listen(port, '0.0.0.0', () => console.log(`TopAi API listening on ${port}`));
