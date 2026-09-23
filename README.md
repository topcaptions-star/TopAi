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
MAX_TRACK_FRAMES=96
TRACK_TIMEOUT_MS=240000
```

Never put the Speechmatics key in GitHub, the ZXP, or panel JavaScript.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health and tracking capability |
| `GET /healthz` | Basic health check |
| `POST /transcribe` | Speechmatics transcript from multipart `media` WAV/video |
| `POST /track-preview` | Lightweight preview image from multipart source video `media` |
| `POST /auto-face-center` | Continuous optical tracking from the retained preview upload or multipart source video `media` |

`/track-preview` prepares one lightweight frame. The user clicks the intended subject, then `/auto-face-center` follows visual features around that selected point with Lucas–Kanade optical flow at up to six samples per second. The preview upload can be referenced through `previewJobId`, avoiding a second media upload. The CEP host builds the pre-comp and writes Position keyframes plus Motion Tile locally in After Effects. The service contains no EXE, JSXBIN, Mocha component, or third-party plug-in UI.

All uploaded media, decoded frames, and temporary files are removed after each request.

## Local test

```bash
npm install
cp .env.example .env
npm start
curl http://localhost:10000/health
```

`SPEECHMATICS_API_KEY` is required only for `/transcribe`; Auto Face Center endpoints do not use it.
