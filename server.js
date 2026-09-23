import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { mkdtemp, readdir, rm, unlink, writeFile, readFile, access } from 'node:fs/promises';
import { tmpdir, setPriority } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { spawn, fork } from 'node:child_process';
import { openAsBlob } from 'node:fs';
import { BatchClient } from '@speechmatics/batch-client';
import ffmpegPath from 'ffmpeg-static';

const app = express();
const port = Number(process.env.PORT || 10000);
const maxMb = Number(process.env.MAX_UPLOAD_MB || 250);
const maxTrackSeconds = Number(process.env.MAX_TRACK_SECONDS || 60);
const maxTrackFrames = clamp(Number(process.env.MAX_TRACK_FRAMES || 96), 12, 96);
const processTimeoutMs = Number(process.env.TRACK_TIMEOUT_MS || 240000);
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: maxMb * 1024 * 1024 } });
const centerJobs = new Map();

app.disable('x-powered-by');
app.use(cors({ origin: process.env.ALLOWED_ORIGIN || '*' }));
app.use((req, _res, next) => { console.log(JSON.stringify({ event: 'request', method: req.method, path: req.path })); next(); });
app.get('/health', (_req, res) => res.json({ ok: true, service: 'topai-api', tracking: { enabled: true, engine: 'topai-lucas-kanade-optical-flow', maxTrackSeconds, maxTrackFrames, workflow: 'manual-subject-select-optical-flow' } }));
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
function runTrackingWorker(workerData) {
  return new Promise((resolve, reject) => {
    const child = fork(new URL('./tracking-worker.js', import.meta.url), [], { env: { ...process.env, TOPAI_TRACKING_JOB: JSON.stringify(workerData) }, stdio: ['ignore', 'ignore', 'pipe', 'ipc'] });
    try { setPriority(child.pid, 19); } catch (_priorityError) {}
    let settled = false;
    const finish = (callback, value) => { if (settled) return; settled = true; clearTimeout(timer); callback(value); };
    const timer = setTimeout(() => { child.kill('SIGKILL'); finish(reject, Object.assign(new Error('Worker timed out'), { publicMessage: 'Auto Center timed out while analyzing the video.' })); }, processTimeoutMs);
    child.once('message', message => {
      clearTimeout(timer);
      message?.ok ? finish(resolve, message.data) : finish(reject, Object.assign(new Error(message?.error || 'Auto Center worker failed'), { publicMessage: message?.error || 'Auto Center worker failed.' }));
    });
    child.once('error', error => finish(reject, error));
    child.once('exit', code => { if (code !== 0) finish(reject, Object.assign(new Error(`Worker exited ${code}`), { publicMessage: 'Auto Center processing stopped unexpectedly.' })); });
  });
}
function expireCenterJob(jobId) {
  const job = centerJobs.get(jobId);
  if (!job) return;
  centerJobs.delete(jobId);
  if (job.mediaPath) unlink(job.mediaPath).catch(() => {});
}
function createCenterJob({ mediaPath, sampleFps, maxFrames, durationSeconds, sourceStart, seed }) {
  const jobId = randomUUID();
  const job = { id: jobId, kind: 'center', state: 'processing', createdAt: Date.now(), mediaPath, result: null, error: null };
  centerJobs.set(jobId, job);
  runTrackingWorker({ action: 'track', videoPath: mediaPath, durationSeconds, sourceStart, sampleFps, maxFrames, seed })
    .then(tracked => {
      job.state = 'complete';
      job.result = { ...tracked, engine: 'topai-lucas-kanade-optical-flow', coordinateSpace: 'normalized-video-frame', smoothing: 'optical-flow-median-stabilizer', composition: 'subject-center' };
      console.log(JSON.stringify({ event: 'center_job_done', jobId, points: tracked.shots.reduce((count, shot) => count + shot.points.length, 0), durationMs: Date.now() - job.createdAt }));
    })
    .catch(error => {
      job.state = 'failed';
      job.error = publicError(error, 'Auto Center failed');
      console.error(JSON.stringify({ event: 'center_job_error', jobId, error: error.message }));
    })
    .finally(() => { if (job.mediaPath) unlink(job.mediaPath).catch(() => {}); job.mediaPath = null; setTimeout(() => expireCenterJob(jobId), 10 * 60 * 1000); });
  return job;
}
function createPreviewJob({ mediaPath, durationSeconds, sourceStart }) {
  const jobId = randomUUID();
  const job = { id: jobId, kind: 'preview', state: 'processing', createdAt: Date.now(), mediaPath, result: null, error: null };
  centerJobs.set(jobId, job);
  previewFaces(mediaPath, durationSeconds, sourceStart)
    .then(preview => {
      job.state = 'complete';
      job.result = preview;
      console.log(JSON.stringify({ event: 'preview_job_done', jobId, faces: preview.faces.length, durationMs: Date.now() - job.createdAt }));
    })
    .catch(error => {
      job.state = 'failed';
      job.error = publicError(error, 'Could not prepare subject selection.');
      console.error(JSON.stringify({ event: 'preview_job_error', jobId, error: error.message }));
    })
    .finally(() => { setTimeout(() => expireCenterJob(jobId), 10 * 60 * 1000); });
  return job;
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
    const offset = rounded(clamp(Math.min(0.4, durationSeconds * 0.25), 0, Math.max(0, durationSeconds - 0.08)), 3);
    const framePath = join(workingDir, 'subject-preview.jpg');
    await run(ffmpegPath, ['-hide_banner', '-loglevel', 'error', '-ss', String(sourceStart + offset), '-i', videoPath, '-frames:v', '1', '-vf', 'scale=720:-2:force_original_aspect_ratio=decrease', '-q:v', '4', framePath]);
    try { await access(framePath); } catch (_missingFrame) { throw Object.assign(new Error('Preview frame was not created'), { publicMessage: 'TopAi could not read a preview frame from the selected Work Area.' }); }
    const image = await readFile(framePath);
    return { imageBase64: image.toString('base64'), mime: 'image/jpeg', time: offset, manual: true, sampleCount: 1, scannedFrames: 1, detector: 'manual-subject-selection', faces: [{ id: 0, x: 0.5, y: 0.5, box: { x: 0.38, y: 0.30, width: 0.24, height: 0.40 }, score: 1 }] };
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
    const job = createPreviewJob({ mediaPath, durationSeconds, sourceStart });
    return res.status(202).json({ ok: true, state: job.state, jobId: job.id, pollUrl: `/auto-center-jobs/${job.id}` });
  } catch (error) {
    const status = error?.publicMessage?.startsWith('No stable') ? 422 : 502;
    console.error(JSON.stringify({ event: 'preview_error', requestId, error: error.message }));
    return res.status(status).json({ error: publicError(error, 'Could not prepare face selection.') });
  }
});

app.post('/auto-face-center', upload.single('media'), async (req, res) => {
  const requestId = randomUUID();
  const previewJobId = String(req.body.previewJobId || '');
  const previewJob = previewJobId ? centerJobs.get(previewJobId) : null;
  if (!req.file && (!previewJob || previewJob.kind !== 'preview' || previewJob.state !== 'complete' || !previewJob.mediaPath)) return res.status(400).json({ error: 'Send a video file in multipart field: media, or complete subject selection first.' });
  const requestedFps = clamp(delaySafeNumber(req.body.sampleFps, 6), 3, 6);
  const maxFrames = clamp(Math.round(delaySafeNumber(req.body.maxFrames, maxTrackFrames)), 12, maxTrackFrames);
  const durationSeconds = clamp(delaySafeNumber(req.body.durationSeconds, maxTrackSeconds), 1, maxTrackSeconds);
  const sampleFps = Math.max(1, Math.min(requestedFps, maxFrames / durationSeconds));
  const sourceStart = Math.max(0, delaySafeNumber(req.body.sourceStart, 0));
  let seed = null;
  if (req.body.seedX !== undefined && req.body.seedY !== undefined) seed = {
    x: clamp(delaySafeNumber(req.body.seedX, 0.5), 0, 1),
    y: clamp(delaySafeNumber(req.body.seedY, 0.5), 0, 1),
    time: clamp(delaySafeNumber(req.body.seedTime, 0), 0, durationSeconds),
    box: { width: clamp(delaySafeNumber(req.body.seedWidth, 0.28), 0.05, 0.9), height: clamp(delaySafeNumber(req.body.seedHeight, 0.42), 0.05, 0.95) }
  };
  const mediaPath = req.file ? join(tmpdir(), `topai-center-source-${requestId}-${safeName(req.file.originalname)}`) : previewJob.mediaPath;
  try {
    console.log(JSON.stringify({ event: 'center_received', requestId, bytes: req.file?.size || 0, reusedPreviewUpload: Boolean(previewJob), requestedFps, sampleFps, maxFrames, durationSeconds, sourceStart, hasSeed: Boolean(seed) }));
    if (req.file) await writeFile(mediaPath, req.file.buffer);
    if (previewJob) { previewJob.mediaPath = null; centerJobs.delete(previewJob.id); }
    const job = createCenterJob({ mediaPath, sampleFps, maxFrames, durationSeconds, sourceStart, seed });
    return res.status(202).json({ ok: true, state: job.state, jobId: job.id, pollUrl: `/auto-center-jobs/${job.id}` });
  } catch (error) {
    const status = error?.publicMessage?.startsWith('No') ? 422 : 502;
    console.error(JSON.stringify({ event: 'center_error', requestId, error: error.message }));
    return res.status(status).json({ error: publicError(error, 'Auto Face Center failed') });
  }
});
app.get('/auto-center-jobs/:jobId', (req, res) => {
  const job = centerJobs.get(String(req.params.jobId));
  if (!job) return res.status(404).json({ error: 'Auto Center job expired or the server restarted. Please retry.' });
  if (job.state === 'complete') return res.json(job.kind === 'preview' ? { ok: true, state: 'complete', preview: job.result } : { ok: true, state: 'complete', tracking: job.result });
  if (job.state === 'failed') return res.status(422).json({ ok: false, state: 'failed', error: job.error || 'Auto Center failed.' });
  return res.status(202).json({ ok: true, state: 'processing', elapsedSeconds: Math.round((Date.now() - job.createdAt) / 1000) });
});

app.use((err, _req, res, _next) => {
  if (err?.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ error: `File exceeds ${maxMb}MB limit` });
  console.error(JSON.stringify({ event: 'http_error', error: err?.message || String(err) }));
  return res.status(500).json({ error: 'Server error' });
});

app.listen(port, '0.0.0.0', () => console.log(`TopAi API listening on ${port}`));
