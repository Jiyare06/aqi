# frontend

Plain HTML/CSS/JS PWA, no build step. Talks to the SAM API (`infra/template.yaml`, output `ApiUrl`):
`POST /jobs` → `PUT` the clip to `upload_url` → poll `GET /jobs/{id}` every 2 s (90 s timeout).
The result screen renders `result` (shape: `backend/contract.json`) plus `audio_urls` / `gradcam_url`.

| File | What |
|---|---|
| `config.js` | `API_BASE` (the deployed API URL) |
| `api.js` | create job, XHR upload with progress, polling |
| `app.js` | input / progress / result screens, geolocation + city fallback, retry, `?mock=1` |
| `result.js` | decision, AQI + source, forecast strip, causes, EN/HI advice + audio |
| `sw.js`, `manifest.webmanifest` | installable PWA; network-first for own files (cache = offline fallback), never intercepts cross-origin (API, S3) |

## Run locally

```bash
cd frontend && python3 -m http.server 8000
```

Open http://localhost:8000 (geolocation works on `localhost` and HTTPS only; on a phone over plain LAN HTTP use the city dropdown).
`http://localhost:8000/?mock=1` renders `mock/result.json` (gitignored): either a full `GET /jobs/{id}` body or a bare contract result.

## Deploy (Amplify Hosting, manual)

1. Set `API_BASE` in `config.js` to the `ApiUrl` stack output, and **bump `CACHE_VERSION` in `sw.js`** (every deploy).
2. Zip the **contents** of `frontend/` (`index.html` must sit at the zip root), without `mock/`. Pass `icons` and `bg` as folders: listing their files individually makes `zipfile` drop the folder paths.
   ```bash
   cd frontend && python3 -m zipfile -c ../frontend.zip index.html styles.css config.js format.js api.js result.js app.js sw.js manifest.webmanifest icons bg
   ```
3. AWS console → Amplify (region ap-south-1) → **Create new app** → **Deploy without Git** → app name `hazebusters`, branch `main` → drag `frontend.zip` → **Save and deploy**.
4. Open the `https://main.<id>.amplifyapp.com` URL. Redeploy: rebuild the zip → app → branch `main` → **Deploy updates** → drop the new zip.

The API and S3 bucket allow CORS from `*`, so no backend change is needed for the Amplify domain.
