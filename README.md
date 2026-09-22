# TopAi Render API

Node.js backend for the TopAi After Effects CEP panel.

## Render deployment

1. Connect this repository as a Render Web Service.
2. Set **Build Command** to `npm install`.
3. Set **Start Command** to `npm start`.
4. Add server-only values in **Render Dashboard → Environment**:

```text
SPEECHMATICS_API_KEY=your_secret_key
ALLOWED_ORIGIN=*
MAX_UPLOAD_MB=250
MAX_TRACK_SECONDS=60
MAX_TRACK_FRAMES=360
TRACK_TIMEOUT_MS=240000
```

Never put the Speechmatics key in GitHub, the ZXP, or panel JavaScript.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health and tracking capability |
| `GET /healthz` | Basic health check |
| `POST /transcribe` | Speechmatics transcript from multipart `media` WAV/video |
| `POST /track-face` | MediaPipe FaceMesh 478-landmark tracking from multipart MP4 `media` |

`/track-face` extracts a capped number of low-resolution frames, runs MediaPipe FaceMesh with 478 landmarks, follows the most consistent face, smooths head and feature points, and returns keyframes for the After Effects Head, Left Eye, Right Eye, Nose, and Mouth Nulls. It contains no EXE, JSXBIN, or third-party plug-in UI.

All uploaded media, decoded frames, and temporary files are removed after each request.

## Local test

```bash
npm install
cp .env.example .env
npm start
curl http://localhost:10000/health
```

`SPEECHMATICS_API_KEY` is required only for `/transcribe`; `/track-face` does not use it.
