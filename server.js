import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { writeFile, unlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { openAsBlob } from 'node:fs';
import { BatchClient } from '@speechmatics/batch-client';

const app = express();
const port = Number(process.env.PORT || 10000);
const maxMb = Number(process.env.MAX_UPLOAD_MB || 250);
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: maxMb * 1024 * 1024 } });
app.use(cors({ origin: process.env.ALLOWED_ORIGIN || '*' }));
app.use((req, _res, next) => { console.log(`[request] ${req.method} ${req.path}`); next(); });
app.get('/health', (_req, res) => res.json({ ok: true, service: 'alvar-caption-api' }));
app.get('/healthz', (_req, res) => res.json({ ok: true, service: 'alvar-caption-api' }));

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
  console.log(`[transcribe] received file=${Boolean(req.file)} bytes=${req.file?.size || 0} language=${req.body?.language || 'auto'}`);
  if (!process.env.SPEECHMATICS_API_KEY) return res.status(500).json({ error: 'SPEECHMATICS_API_KEY is not configured on the server' });
  if (!req.file) return res.status(400).json({ error: 'Send an audio/video file in multipart field: media' });
  const language = String(req.body.language || 'auto');
  const model = ['standard', 'enhanced', 'melia-1'].includes(req.body.model) ? req.body.model : 'melia-1';
  const maxChars = Math.min(80, Math.max(10, Number(req.body.maxChars || 32)));
  const path = join(tmpdir(), `alvar-${randomUUID()}-${req.file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_')}`);
  try {
    await writeFile(path, req.file.buffer);
    const client = new BatchClient({ apiKey: process.env.SPEECHMATICS_API_KEY, appId: 'alvar-motion-suite' });
    const blob = await openAsBlob(path);
    const file = new File([blob], req.file.originalname || 'media');
    const config = { transcription_config: { language, model } };
    const response = await client.transcribe(file, config, 'json-v2');
    const words = normalizeWords(response);
    console.log(`[transcribe] completed words=${words.length}`);
    return res.json({ language: response?.metadata?.transcription_config?.language || language, model, words, segments: makeSegments(words, maxChars), source: 'speechmatics-json-v2' });
  } catch (error) {
    console.error('[transcribe] Speechmatics error:', error?.stack || error);
    return res.status(502).json({ error: 'Speechmatics transcription failed', detail: error?.message || String(error) });
  } finally { await unlink(path).catch(() => {}); }
});
app.use((err, _req, res, _next) => { if (err?.code === 'LIMIT_FILE_SIZE') return res.status(413).json({ error: `File exceeds ${maxMb}MB limit` }); res.status(500).json({ error: err.message || 'Server error' }); });
app.listen(port, '0.0.0.0', () => console.log(`Alvar Caption API listening on ${port}`));
