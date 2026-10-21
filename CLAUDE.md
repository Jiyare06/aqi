# PROJECT CONTEXT — Environmental Hacks (Air track)

> Handoff for Claude Code or any new AI chat session. Read this fully before helping.
> Lives in the repo as `CLAUDE.md`. Last updated: Fri 9 Oct 2026, night (AWS skeleton deployed in ap-south-1, §7.1; WAQI live-tested, §11).
> **Strategy is expected to change** as we build; when something here conflicts with what the
> user says now, the user wins — then update this file.

---

## 0. Working rules for AI sessions (read first)

- Datasets live at `$DATA_DIR` (= `/home/abhaysingh/datasets`), **outside the repo**. Read only CSVs; never browse image folders.
- Never commit data, model weights, `.env`, keys or secrets.
- Never add `Co-Authored-By` trailers or AI attribution to commits/PRs.
- Don't reuse anything from the old "AQI Vision" project (see §2).
- Prefer small, working, committed steps over big rewrites. Commit after each working step.
- User prefers concise, practical, informal answers.
- **Background monitors produced fake results** (made-up runs with plausible metrics). Only report metrics read from files on disk (`ml/runs/...`); run long jobs in the foreground or detached and verify on disk.

## 1. Who / what

- **User:** Abhay, B.Tech ECE, BIT Mesra (batch 2027). Python-first. Team of **3**.
- **Event:** WeMakeDevs × AWS **Environmental Hacks** (Bharat Builds Tour). Online, India. **Track: Air.** Own problem allowed.
- **Schedule (from the Schedule page, hours "being finalised"):**
  - Thu 8 Oct — Kickoff: the clock starts online, teams form, repos are created. **Exact start time not announced yet — watch the Schedule page + WeMakeDevs Discord. No commits before it.**
  - Fri 9 Oct — Build day. Sat 10 Oct — optional in-person day at DTU Delhi (8 AM–8 PM); online continues.
  - Sun 11 Oct — last day to submit (exact time TBA — assume early in the day until announced).
- Prizes per track: ₹2,00,000 + $2,000 AWS credits. Top 5 Builder Center blogs: AirPods.
- **Real goal:** Amazon **fast-track interview** (max 10 per hackathon; 2027/2028 grads; decided separately from prizes; needs Builder Center student verification).
- Judging: idea & impact (small problem solved well), built on AWS, design & usability, execution ("one feature that runs beats five that almost do"), 3-min demo video.

## 2. Hard rules (violating these = disqualification)

1. **Build only after the clock starts.** No project code before kickoff. Planning, learning, downloading/inspecting data = allowed. Exploration scripts written before kickoff must be **rewritten in the repo after kickoff**, not copied in.
2. **No old projects.** The academic minor project ("AQI Vision", https://github.com/ASR222/AQI_project) **must not be reused**: no code, no weights (`aqi_model.keras`), no frontend, not the name. Ideas/knowledge are fine.
3. Fresh repo after kickoff; commit early and often (history must match event dates).
4. **Must use AWS** (an AWS open-source tool such as Strands Agents / SAM CLI, or deploy on AWS) **and the demo video must show it.**
5. Submission = public repo + YouTube video **≤ 3 min** (public/unlisted, test signed-out) + short writeup (problem, build, where AWS fits). Judges see only the submission.
6. **List AI coding tools used** in the writeup (Claude Code etc.) even though commits carry no AI trailer. Credit everything not written by us.
7. Each member registers individually and verifies student status on Builder Center.

## 3. The product

**Product:** For a place with no monitor nearby: how bad the air likely is today, how far that number comes from, what's visibly polluting nearby, and what to do, in Hindi.

- **Names:** repo folder is `hazebusters`. Confirm final **team name** and **product name** (not "AQI Vision"; candidate product name: SmogSight).
- **Hero user:** a **school** deciding whether today's PE/assembly go outdoors. Second profile: outdoor/gig worker. Plus general.

**Flow:**
1. Pick a profile.
2. Record a 10 s clip with location.
3. Air: nearest WAQI station (US AQI → µg/m³ → §5) plus the CAMS model value for the area via Open-Meteo (~45 km, µg/m³ → §5). Pick a source with the 25 km / 3 h rule. Confidence comes from agreement and distance.
4. Decision for the profile from the AQI number plus today's 3-hourly forecast (outdoor yes/limit/no, best window).
5. Clip → quality gate → EfficientNet-B0 smoke/fire/clear classifier + CAM heatmap, and YOLO11n smoke/fire boxes + vehicle count. Causes only; they never change the AQI.
6. Seasonal line.
7. Bedrock advice in EN/HI (Converse, Nova 2 Lite; template fallback), plus Polly Hindi + English audio.
8. Optional report-burning draft.

**No image-based AQI:** see §4 and ml/RESULTS.md.

**Pitch:** "Most places have no monitor nearby. We tell you how far your nearest reading really is, what's polluting the air around you, and what to do today, in Hindi."

**Credits:** CAMS + Open-Meteo (non-commercial).

API response shape: `backend/contract.json` (`air.measured` can be null, e.g. Ranchi).

## 4. ML design

### Current image models (Fri) — causes only, never AQI
- **Image → AQI is an experiment only, not in the product.** It failed on unseen days and cities (seed-0 CNN 32% vs 47.5% baseline; frozen-embedding / handcrafted probes at baseline; India/Nepal leave-one-city-out inverted). Numbers: `ml/RESULTS.md`.
- **Smoke/fire classifier (trained Fri):** EfficientNet-B0, 2 sigmoid outputs (smoke, fire; "clear" = neither), whole frame squashed to 224×224, trained on **D-Fire** (CC0) with the dHash + AoF-sequence grouped split `ml/splits/dfire.json`. Code: `hazeml/train_smokefire.py`, `smokefire_data.py`, `export_onnx.py`. D-Fire test AP smoke 0.979 / fire 0.982 (thresholds from val: smoke 0.359, fire 0.667). **External Pyro-SDIS: smoke recall 13.4%, FPR 20.5%, AP 0.845 below the 0.868 base rate → doesn't transfer to distant wildfire plumes.** ONNX (features + logits) verified vs torch (max diff 2.1e-4). Heatmap = **CAM** in numpy (`hazeml/cam.py`) from the ONNX features + `head.npz`, not Grad-CAM (no gradients needed on Lambda). Details: `ml/RESULTS.md` §3.
- **YOLO11n** (Ultralytics, **AGPL** — repo stays open source), fine-tuned for smoke/fire; vehicles from pretrained COCO classes. Output: smoke/fire boxes + vehicle count.
- The quality gate (night, no sky, blur, indoor) runs before both models; their outputs feed `causes` in `backend/contract.json` and never change the AQI.
- **LLM only phrases** numbers our pipeline computed. Never estimates AQI or invents numbers/health claims.
- **Export to ONNX** for Lambda.
- **Compute:** train locally on **RTX 4050 (6 GB, ~5 GB free)** in WSL; Kaggle/Colab as backup or for a teammate's parallel experiment. Optional speed-up: pre-resize TRAQID images to 256 px once into `$DATA_DIR/traqid_256/`.
- **No 3-stream CNN** (old project's design): too slow for video on Lambda, doubtful gain, too close to the old project. Optional Sat ablation only if ahead.

### Image → AQI (experiment only; archived — see `ml/RESULTS.md`)
- **Training data: TRAQID only, daytime (`Day_or_Night == Day`), front view only.** The app's quality gate rejects night frames, so training matches real use and drops the noisiest half. After the filter: **13,789 frames from only 17 calendar dates** (2022-10 → 2024-07). Loader: `ml/src/hazeml/traqid.py` (`load_traqid()`, reads only the CSV).
- **Time-of-day confound:** the worst air (21–23 Dec 2023) was filmed on **morning drives**; most other frames are late afternoon (15:00–18:00). Very Poor/Severe share: **43% of frames before 11:00 vs 9% from 15:00** (July mornings are mostly clean, so it's not absolute). The CNN can learn "morning light = bad air" → use strong colour/brightness/contrast augmentation, report accuracy by morning vs evening, and lean on the OOD test + own clips.
- **Model:** pretrained **EfficientNet-B0 or MobileNetV3**, fine-tuned, 224 px, mixed precision, batch ~32 (reduce on OOM).
- **Classes: 3 CPCB bands** — Good/Satisfactory (0–100), Moderate/Poor (101–300), Very Poor/Severe (301+). Use **class weights** (Very Poor/Severe is only ~12% of TRAQID).
- **Multimodal variant (Fri):** CNN embedding ⊕ [mean saturation, contrast, dark-channel, brightness, temperature, humidity, season, YOLO vehicle count] → MLP head. TRAQID only. **No `hour`** (it encodes the time-of-day confound above). **`season` is risky**: with 17 dates it nearly identifies the collection trip (all Very Poor/Severe-heavy dates are Winter/Dec) → include only if an ablation without it is also reported. **Compare vs image-only on the same split; keep the winner.** Image-only stays the runtime fallback.
- **Never feed station AQI/PM as a model input** (it's the label; the model would copy it). Station reading is shown separately.

### Image → AQI evaluation (be honest; archived)
- **TRAQID split by calendar date** (from `created_at`): hold out ~20% of dates as test (plus a val split). Random splits leak (frames 5 s apart).
  - **Frozen in `ml/splits/traqid_dates.json`** (committed; dates only). Chosen by exhaustive search for ~70/10/20 rows with each split's band mix close to overall: train 9,657 rows / 10 dates, val 1,361 / 2 (2022-10-26, 2024-03-06), test 2,771 / 5 (2023-01-30, 02-01, 02-08, 2023-12-21, 2024-03-02). Don't change it once results are reported; `python -m hazeml.traqid --make-split` regenerates.
  - **Very Poor/Severe test recall rests mostly on one morning (21 Dec 2023, ~430 of ~460 band-2 test rows)** → expect it to swing; say so.
  - **21 Dec is in test while 22–23 Dec (same smog episode) are in train** — no shared frames, but shared weather/lighting. Keeping the episode together left too little band 2 in train or test; disclose it in the writeup.
- **India/Nepal = out-of-domain test set, not training.** Recompute CPCB labels (§5), evaluate the TRAQID-trained model on it. Pitch line: *"Trained in Hyderabad, tested on 7 other Indian cities."* If ever used for training, split by **moment** (location+date+hour), not image.
- Report: **macro-F1**, accuracy vs **majority baseline** (≈52% on all TRAQID 3-band; **≈47% after the day-only filter** — use 47% for our numbers; per split 47.2/47.4/47.5%), **within-one-band accuracy**, and **recall for Very Poor/Severe** separately.
- Sanity check: agreement between our recomputed TRAQID labels and its own `aqi_cat`.
- **Result (Thu, EfficientNet-B0, seed 0, `ml/runs/20261008-174356_efficientnet_b0_s0/`): test accuracy 32.0% vs 47.5% majority baseline** (macro-F1 0.312, band-2 recall 57%, confidently wrong: mean confidence 0.80); val 22%. Same checkpoint on its **train dates: 89%** (82–100% per date) → no pipeline bug; **the model learns the day, not haze**. Frames within a date are highly correlated (a single band covers a median ~78% of a date's frames, 47–99%; ≥94% on 4 of 17 dates), so there are effectively ~17 independent samples. Seeds 1–2 not run (stopped). **Probes (Thu night) confirmed it:** no image signal transfers to unseen dates or cities → image→AQI dropped from the product. Details in `ml/RESULTS.md` (raw files in `ml/runs/probes/`, gitignored).
- Old rough expectation (now unrealistic for TRAQID-only): ~65–80% on held-out TRAQID dates; lower out-of-domain and on our own clips. Collect 20–30 own clips/photos as a real-world test set.

## 5. Label mapping — CPCB-style AQI (first thing to code after kickoff)

Ignore datasets' own class labels; recompute from raw PM2.5 / PM10 with one shared function.

| AQI | Category | PM2.5 (µg/m³) | PM10 (µg/m³) |
|---|---|---|---|
| 0–50 | Good | 0–30 | 0–50 |
| 51–100 | Satisfactory | 31–60 | 51–100 |
| 101–200 | Moderate | 61–90 | 101–250 |
| 201–300 | Poor | 91–120 | 251–350 |
| 301–400 | Very Poor | 121–250 | 351–430 |
| 401–500 | Severe | 250+ | 430+ |

Sub-index = linear interpolation within the row; **AQI = max(sub-index PM2.5, sub-index PM10)** (use whichever pollutants exist); then category → 3 bands. Call it "CPCB-style AQI" in the writeup (official AQI uses 24-h averages and ≥3 pollutants).

## 6. Data

### 6.1 On disk (WSL, outside the repo)
```
~/datasets/
  traqid/       front/<Image>.png, rear/<Image>.png (flat folders), TRAQID.csv
  india-nepal/  "Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/"
                  All_img/ (use this only) + IND_and_Nep_AQI_Dataset.csv  (loader: ml/src/hazeml/india_nepal.py)
  pm25vision/   dropped — not used (safe to delete, 1.1 GB)
  dfire_hf/     D-Fire (CC0) parquet from HF mirror badsaarow/d-fire, 21,527 imgs (ml/download_datasets.py); split: ml/splits/dfire.json
  pyro_sdis/    Pyro-SDIS (Apache-2.0) parquet from pyronear/pyro-sdis, 12,378 imgs, smoke only, camera + date columns
  stations/     one folder per state, one CSV per station + stations.csv (8.4 GB)
  cpcb_2015_2020/  city_day, city_hour, station_day, station_hour, stations CSVs (Kaggle rohanrao/air-quality-data-in-india)
```
`cpcb_2015_2020` is **only a unit test for the §5 label function**: on rows where PM2.5/PM10 drive the AQI, our CPCB-style AQI should match its `AQI` / `AQI_Bucket` columns. Not used for training or seasonal context. Reference for the full CPCB method: Kaggle notebook "Calculating AQI (Air Quality Index) Tutorial" by rohanrao — read, don't copy; credit if followed closely. Check the dataset's licence on Kaggle before citing.
`.env` in the repo (gitignored): `DATA_DIR=/home/abhaysingh/datasets`, `WAQI_TOKEN=...`

### 6.2 Image datasets
| Dataset | Role | Facts | Licence / credit |
|---|---|---|---|
| **TRAQID** (Kathalkar et al., ICVGIP 2024, doi 10.1145/3702250.3702260; data on india-data.org) | **Training (only)** | 26,678 rows. CSV columns: `created_at, Image, PM2.5, PM10, Temperature, Humidity, aqi, Season, Day_or_Night, aqi_cat`. Image file = `front/<Image>.png` / `rear/<Image>.png`; collection day comes from `created_at`. 6-class counts: Satisfactory 13,033 · Moderate 7,453 · Poor 2,283 · Very Poor 2,233 · Severe 932 · Good 744. 3-band: 51.6% / 36.5% / 11.9%. Hyderabad dashcam, frames every 5 s, ~half night. | India Data agreement: research use, **no commercial use without consent**, attribution, no misrepresentation; faces/plates blurred |
| **India/Nepal** (Kaggle, doi 10.34740/KAGGLE/DS/3152196) | **Out-of-domain test** | Use only `Combined_Dataset/All_img` (12,240 imgs); other folders are copies. Only **115 distinct location+date+hour moments** → heavy near-duplicates. Authors' split (`Dataset_for_AQI_Classification/*.csv`) **leaks — don't use**. Its AQI column uses US bands and can disagree with raw PM2.5 → recompute per §5. | **CC BY-NC-SA 3.0 IGO**. Cite dataset + Eff-AQI (Utomo et al., GoodIT '23, doi 10.1145/3582515.3609531) + Future Internet 2023 paper |
| PM25Vision | **Dropped** | Its `pm25` field is a US AQI value, not µg/m³ | — |

### 6.3 Seasonal context (stations dataset)
- Source: Kaggle "India AQI 2010–2024" (omsandeeppatil/indian-aqi-stations), 546 stations, **CC BY-SA 4.0**.
- Built (as pre-kickoff exploration): city × month **median daily PM2.5**, city-day = mean of that city's stations. Medians peak Nov–Dec (Delhi ~216 µg/m³).
- Coverage: **Delhi** 43 stations (15–16 yrs/month), **Hyderabad** 12 (9–10 yrs), **Patna** 4 (2019–2024, 5 yrs). **No Ranchi station**; Jharkhand only JH01 Jamshedpur (31% PM2.5 coverage) and JH02 Dhanbad (from 2023).
- **After kickoff:** rewrite as `ml/build_seasonal_lookup.py` → `backend/data/seasonal_lookup.json`; use §5 mapping (PM2.5 and PM10 if available) so labels match the model; optionally restrict to 2018+ (Delhi's network grew over time). JSON carries `source`, `licence: CC BY-SA 4.0`, `metric`, and per city `lat/lon` + per month `pm25, p75, days, category`. Credit in README.
- App use: nearest lookup city **within ~50 km** of the user, else hide the line (Ranchi gets none). Wording: *"A typical November day here: Very Poor (PM2.5 ≈ 216)."* — never call it the AQI.

### 6.4 Pitch data
- Line (true for this dataset): *"In this 546-station dataset, Jharkhand has 2 stations and none in Ranchi."* UrbanEmissions NCAP bulletins (2015–2023) also had no Ranchi.
- **WAQI live check (9 Oct): no Ranchi station** (geo feed → `can not connect`, city search → `Unknown station`) → pitch: "no official monitor in Ranchi". Re-check right before the demo; if one appears → "a city of 1.5M relies on a single recently added monitor".

### 6.5 Live readings
WAQI API (free token); backup OpenAQ v3. Sanity-check values (live CPCB data is unvalidated).

Rejected: data.gov.in real-time AQI (no API, 502s, snapshot only); adityarc19/aqi-india (tabular, no licence, ends 2020); Kaggle "India AQI 2024" (city-level, unclear source); IJEP 2024 "Image to Insight" paper (ignored).

**Repo data rules:** never commit datasets, images, relabelled CSVs or weights. README gives download steps + citations. Commit relabel/lookup **scripts**; the tiny CC BY-SA seasonal JSON may be committed with its licence + credit. Code licence: open source (AGPL-compatible because of Ultralytics). Both training/test image datasets are non-commercial → a future startup needs permission or our own data.

## 7. AWS architecture

```
PWA (Amplify Hosting)
  └─► API Gateway ─► Lambda: presigned S3 URL
PWA uploads video ─► S3 ─(event)─► Lambda container image (ONNX: quality gate + smoke/fire/clear CNN + YOLO11n, frame sampling)
                                     ├─► DynamoDB (results, history)
                                     ├─► WAQI (nearest station) + Open-Meteo Air Quality (CAMS global, ~45 km) for measured + modelled PM
                                     ├─► Bedrock, called directly via Converse (Amazon Nova 2 Lite; advice EN + HI; template fallback; no Strands)
                                     └─► Polly (Hindi audio of the advice) ─► S3 (audio key in the result)
PWA fetches result ◄─ API Gateway ◄─ Lambda
```
- Deploy with **SAM CLI**; image pushed to **ECR** (no Docker Hub account needed; base image `public.ecr.aws/lambda/python`). Skip SageMaker.
- One region for everything: **ap-south-1** (Mumbai).
- Free Tier + up to $200 credits; zero-spend / small budget alert set.
- Show the flow on screen in the demo video.
- Air logic lives in `backend/src/air/` (WAQI, Open-Meteo, combine); response shape in `backend/contract.json`. Open-Meteo is free for non-commercial use and needs CAMS + Open-Meteo attribution.

### 7.1 Deployed infra (`infra/`)
- Stack **`hazebusters`**, region **ap-south-1**. Template `infra/template.yaml`; build/deploy from `infra/` with `sam build && sam deploy` (one Docker build at a time, foreground; redact the token from output).
- API base URL: see team doc (`POST /jobs`, `GET /jobs/{id}`; `e2e_test.py` reads it from the stack outputs). Bucket `hazebusters-uploads-<account>`: clips under `uploads/`, audio under `audio/`, both deleted after 7 days; the processor never writes to `uploads/` (that prefix triggers it).
- Processor = container Lambda (`infra/processor/`): frames ~1/s → `causes.py` → `air_for` → `advice.py` → `audio.py`. Result passes through `forecast_today`, `forecast_tomorrow` and `decision` (incl. `decision.for`) from `air_for`.
- **Smoke/fire in the processor (`causes.py`):** the ML run's `smokefire.onnx` + `head.npz` + `export_report.json` (from `ml/runs/smokefire/20261009-220554_efficientnet_b0_s0/`) are copied to `infra/processor/models/` (gitignored, allowed in `.dockerignore`, baked into the image). onnxruntime 1.31.0 (CPU), session loaded once per cold start. Preprocessing = ML eval path (RGB, whole frame squashed to 224×224 bilinear+antialias, /255, ImageNet mean/std); checked vs `eval_transform` on 16 D-Fire test images: probs within 0.0053, identical decisions. Per clip: prob = max over frames; present = ≥ 2 frames over threshold (≥ 1 if < 3 frames); thresholds **smoke 0.600** (raised 11 Oct from the val max-F1 0.359) / fire 0.667. Model ~0.2–0.4 s for 10 frames. Model failure → stub causes + caveat "Smoke check unavailable for this clip" (job still done).
- **CAM overlay:** if smoke/fire is present, a CAM overlay (vendored numpy copy of `hazeml/cam.py`) of the best frame, ≤ 640 px wide, goes to `overlays/{job_id}.jpg` (7-day lifecycle); `causes.gradcam_key` holds the key and GetJob adds a presigned **`gradcam_url`** (1 h), like `audio_urls`.
- **Quality gate (`infra/processor/quality.py`, 10 Oct):** before the cause models, each sampled frame (downscaled to 320 px wide) is rejected as "dark" (mean luma < 40), "blur" (80th percentile over a 4×4 grid of tile Laplacian variances < 10; tiles because plain sky has little texture) or "no_sky" (sky-like share of the top 40% < 0.08; sky-like = V ≥ 150 & S ≤ 80, or blue hue). Fewer than 3 passing frames → no smoke/fire/YOLO, `present` false, `vehicle_count` null, caveat "Couldn't check what's in this clip (indoor / too dark / blurry). Point the camera at the sky or street outdoors.", causes not sent to Bedrock; air/decision/advice unchanged. Calibrated on `~/datasets/own_clips` (24 clips, grid search; only 21/216 settings feasible): all 23 outdoor clips pass, `reject_indoor_01` keeps 2/10 frames → rejected. **Margins are thin** (indoor 2 vs 3 needed; `fire_smoke_01` passes 9/15 on sky); recalibrate as own clips grow. Side effect: TRAQID dashcam clip keeps 5/10 frames and its 0.71 smoke false positive disappears.
- **Known limitation (quality gate, 11 Oct; for the writeup):** the rule-based gate rejects plainly indoor, dark or blurred clips (`reject_indoor_01`: 2/10 frames pass), but **bright indoor scenes (white walls/ceiling, windows) can pass as overcast sky** (deployed job `7bfd8dcf…`: 9/9 frames passed, sky 0.36–1.00). Tightening the sky rule wasn't possible without the failing clips, and overcast outdoor clips have no blue to require (`nosmoke_cloud_02`: sky 1.0). A learned indoor/outdoor classifier is future work; no code change was made.
- **Vehicle count (YOLO11n, 10 Oct):** `ml/runs/yolo/yolo11n.onnx` (COCO-pretrained, **AGPL-3.0**, from PR #4's `hazeml.yolo_export`) copied to `infra/processor/models/`; numpy pre/post-processing vendored as `infra/processor/vehicles.py` (identical to `ml/src/hazeml/vehicles.py`, a test checks it). Letterbox 640, conf 0.35, NMS 0.5, car/motorcycle/bus/truck; `causes.vehicle_count` = median of the per-frame counts. Lazy-loaded like the smoke model; fails independently (null + "Vehicle count unavailable for this clip"). `boxes` stays `[]` (meant for smoke/fire boxes, not done). Cost: ~1.5 s per 10-frame clip on Lambda (~150 ms/frame), processor now ~4.7–5.8 s warm (was ~3.1–3.4 s); image 1.26 GB. e2e vehicle clip = 10 TRAQID front frames from the YOLO parity report (scratchpad only): vehicle_count 3. Note: the smoke model flags that Hyderabad dashcam clip as smoke (0.71).
- **Causes in advice (11 Oct, feature freeze):** Bedrock never gets causes and any smoke/fire wording from it is rejected (template fallback). Whenever smoke and/or fire is present, a hedged line is appended in code (Bedrock or template): "Our camera check suggests possible smoke in your clip (experimental). If you can see a smoke source nearby, move away from it." / "कैमरा जाँच के अनुसार आपकी क्लिप में धुआँ हो सकता है (प्रयोगात्मक)। अगर पास में धुएँ का स्रोत दिखे, तो उससे दूर रहें।" (fire / smoke+fire variants in `advice.CAUSE_LINES`). No "possible/likely" tiers. The model gets a sentence budget (`max_sentences`, 3 when a cause line follows). **Own clips at 0.600:** 6/7 smoke found, 5/13 no-smoke flagged (duplicate `nosmoke_haze_cloud_02` excluded; 6/13 at 0.359; scores are mostly near 0 or 1; `ml/runs/own_clips/rules/results.json`), both fire clips; the hedge does the real work.
- **Image / cold start:** image 1.24 GB (was 1.08 GB before onnxruntime + Pillow + model). Cold start (`Init Duration`) 1.8–6.9 s in our runs, but one fresh environment **hit Lambda's 10 s init limit** (`INIT_REPORT … Status: timeout`); Lambda re-ran init inside the invoke and the job still finished (13.3 s). If this recurs: lazy-load the session in the handler, slim the image, or provisioned concurrency (costs).
- **Advice:** Bedrock Converse, **`global.amazon.nova-2-lite-v1:0`** (`BEDROCK_MODEL_ID`; `llm.py`), template fallback (`advice.source = "bedrock" | "template"`). The model only gets computed facts (AQI, category + CPCB health statement, outdoor/for/profile, best window pre-written as spoken EN/HI, source, confidence). Validation before use, else template + logged reason: JSON only (```json fence tolerated; no text around it, no extra keys), en + hi non-empty, ≤ 700 chars, ≤ 4 sentences each, Hindi has Devanagari, AQI in both, every number already in the facts, no markdown/line breaks, no "होगी/होगा" in a Hindi health sentence. Timeouts 8 s, 1 retry.
- **Anthropic models blocked:** the Marketplace subscription isn't permitted for `hackathon-dev` (`INVALID_PAYMENT_INSTRUMENT`). The Haiku 4.5 India profile ID is kept as a comment in the template.
- **Audio:** Polly neural `Kajal`, hi-IN and en-IN in parallel, SSML (90% rate, breaks; "।"/"|" stripped) → `audio/{job_id}_{hi,en}.mp3`; result stores keys only, GetJob adds `audio_urls` (presigned GET, 1 h). A failed language → null key + "Hindi/English audio unavailable" caveat. hi-IN Kajal shows in `aws polly describe-voices` only with `--include-additional-language-codes`.
- **WAQI lookup (`waqi.lookup`, fixed 10 Oct after a phone test):** WAQI's geo feed answers `nope / "can not connect"` both where there is no station (Ranchi) and where stations stopped reporting (**Hyderabad: all 9 WAQI stations last reported 23 Jun 2026**; WAQI's map hides them too). Order: geo feed → stations still on WAQI's map within 25 km (`/feed/@uid`) → our official-station index `backend/src/air/official_stations.json` (552 CPCB stations from the §6.3 dataset, CC BY-SA 4.0; rebuild with `backend/build_station_index.py`) + WAQI search on that city for the name and last update. "No official monitoring station near this location" only when the index has none within 25 km. One merged caveat, **always first in `caveats`** (the page leads with it): "<why no usable station: name, distance, age>, so this uses the CAMS model estimate with PM2.5/PM10 scaled x1.25 (...)". With no official station within 25 km it names the nearest from the index: "No official station within 25 km; the nearest is Jamshedpur (~110 km), …". The index JSON carries source, licence (CC BY-SA 4.0), ShareAlike note and changes; README credits it. Live 10 Oct: Ranchi → none; Hyderabad → New Malakpet, 2.7 km, last reported 109 days ago; Delhi → Mandir Marg, 2.4 km, measured.
- **Protection:** HTTP API default-route throttling, burst 10 / 5 req/s. No reserved concurrency on the processor: the account's Lambda concurrency limit is 10 and Lambda keeps 10 unreserved (`put-function-concurrency` 5 was rejected); request a quota increase if needed.
- `infra/samconfig.toml` is **gitignored and holds the WAQI token** (`parameter_overrides`, written from `.env`; `WaqiToken` is a NoEcho parameter). Never commit or print it.
- E2E test (needs 10 s clips, e.g. ffmpeg `testsrc`; don't commit clips): `python3 infra/e2e_test.py test.mp4 test.webm smoke.mp4 --expect-smoke smoke.mp4 [--place Ranchi|Hyderabad|Delhi] [--expect-vehicles CLIP] [--save-audio DIR]` → POST job (Ranchi, school) → PUT clip → poll → checks contract keys, ~10 frames, air numbers, `forecast_tomorrow`, `decision.for`, EN/HI advice, both mp3s, smoke present/absent as expected and a loading JPEG `gradcam_url` for smoke clips. Smoke test clip: 10 D-Fire **test-split** `smoke` images, 1 s each (scratchpad only, never in the repo). Latest (10 Oct): Bedrock ~0.9–1.0 s, smoke model 0.2–0.4 s, Polly HI+EN ~0.9 s, processor ~3.0–3.6 s warm, upload → result 4.5–6.6 s (2 s polling).
- Unit tests: `cd infra && uv run --no-project --with pytest --with boto3 --with numpy --with pillow --with onnxruntime==1.31.0 pytest -q tests` (Bedrock/Polly/S3/ONNX session faked).

## 8. Dev environment (Abhay's laptop)

- **WSL2 Ubuntu** on Windows; repo at `~/hazebusters` (inside WSL filesystem, not `/mnt/c`). `.wslconfig`: memory 10 GB, swap 8 GB.
- GPU: RTX 4050 6 GB via Windows NVIDIA driver; `nvidia-smi` works in WSL. PyTorch CUDA build installed per pytorch.org; Python via **uv**.
- **Docker Desktop** with WSL integration (the old `podman-docker` shim was removed — `docker` must be real Docker for SAM).
- AWS CLI v2 configured with an **IAM user** (`hackathon-dev`, not root); SAM CLI installed; Node via nvm (LTS).
- Claude Code: Desktop app **WSL session** on `~/hazebusters` (integrated terminal + file pane are unavailable in WSL sessions → use **Open in editor** = VS Code Remote-WSL for terminal/files; diff review works in Desktop). Commit attribution disabled in `~/.claude/settings.json` (`attribution.commit/pr = ""`).
- `.gitignore`: `.env .venv/ data/ node_modules/ __pycache__/ .aws-sam/ runs/ *.onnx *.pt *.pth *.keras *.zip samconfig.toml`

## 9. Team split

- **ML:** §5 label function, TRAQID loader + date split, CNN, haze features, YOLO, ONNX export, evaluation (incl. India/Nepal OOD test), seasonal lookup script.
- **Backend/AWS:** S3/Lambda/DynamoDB/API Gateway via SAM, WAQI + weather, nearest-city seasonal helper, Bedrock advice (Converse, Nova 2 Lite).
- **Frontend + story:** PWA (profile picker, camera/upload, result, advice), demo video, writeup, Builder Center blog.

## 10. Day plan

- **Thu (done):** repo + first commit; §5 label function; TRAQID loader + frozen date split; baseline image→AQI CNN; probes → image→AQI dropped (`ml/RESULTS.md`).
- **Fri:** air backend (`backend/src/air/`: WAQI + Open-Meteo/CAMS + combine + decision) with tests and live checks; `backend/contract.json`; optional CAMS-vs-stations validation; pick a smoke/fire dataset and start the smoke/fire/clear classifier + YOLO11n; AWS skeleton (S3, Lambda, API).
- **Sat:** quality gate + causes end-to-end; Grad-CAM; seasonal line; Bedrock/Strands advice (EN/HI) + Polly; PWA; deploy; ONNX; test with non-team users; cut anything not working by Sat night.
- **Sun:** film demo outdoors; 3-min video (AWS visible); writeup (AI tools listed, citations incl. CAMS/Open-Meteo); blog; submit early.

## 11. Status (Fri 9 Oct → Sat 10 Oct)

Done (Thu): repo and branches (`ml/aqi-labels` → PR #1; `ml/traqid-loader` → PR #2, which also contains #1's commit — merge #1 first or close it), §5 function + tests (matches CPCB city_day on 74% of categories / 85% of bands; misses are almost all one category off, consistent with daily vs rolling averaging), TRAQID loader + split, CNN + probes (negative result), India/Nepal loader.
Done (Fri): pivot docs; `ml/RESULTS.md`; §5 function moved to `shared/` (`cpcb_aqi`, used by ml + backend); `backend/contract.json`; `backend/src/air/` (WAQI, Open-Meteo/CAMS, combine); CAMS vs stations validation (CAMS ≈ 0.8× stations, misses ~2/3 of Very Poor/Severe days in Delhi/Patna).
Air rules now: measured if station ≤ 25 km and < 3 h old, else modelled; **whenever `used == modelled` → CAMS PM ×1.25** (from the validation; PM2.5-based, PM10 assumed; helps Delhi/Patna, **hurts Hyderabad** — `ml/RESULTS.md` §2b); decision = worst slot over the profile's remaining hours today (school 07–17, worker 06–19, general 06–21), or tomorrow's hours after they end (`decision.for`); `no_window_today` / `no_window_tomorrow` if no "yes" slot. Details: `backend/src/air/decision_rules.md`.
Done (Fri night): `forecast_tomorrow`; D-Fire + Pyro-SDIS downloaded and extracted (`$DATA_DIR/dfire_hf`, `$DATA_DIR/pyro_sdis`) — **D-Fire came from the Hugging Face mirror `badsaarow/d-fire`** (the official OneDrive links block scripted downloads; counts match the README); D-Fire split v2 (no consecutive-frame leaks ≤ 15; **the 10% group cap was hit 4 times**); smoke/fire model trained + ONNX + CAM: **D-Fire test AP smoke 0.979 / fire 0.982; Pyro-SDIS smoke recall 13.4%, so it doesn't transfer yet** (`ml/RESULTS.md` §3).
Done (Fri night): **WAQI token set and live-tested** (Delhi returns a station; invalid key / HTTP errors / timeouts → "WAQI unavailable"). Backend tests: 66 passing. **AWS skeleton deployed** (§7.1), e2e passing.
Done (Fri, late): advice from Bedrock Nova 2 Lite with validation + template fallback; Polly HI + EN audio; e2e passing with `advice.source = "bedrock"` (§7.1). Infra tests: 66 passing.
Done (Sat early, infra): smoke/fire ONNX model + CAM overlay in the processor; likely smoke/fire always shown in advice; merged `origin/main` (forecast_tomorrow) into `infra/sam-skeleton` (§7.1).
Tests (10 Oct, after the merge): backend 68, shared 50, infra 92 (ml 33 on main, not rerun here).
Env notes: PyTorch is installed in the `ml/` uv project (cu128), not system-wide; WSL currently sees 7 GB RAM, not the 10 GB in `.wslconfig`.
Done (Sat, infra): quality gate; Open-Meteo retried once (a single 10 s timeout had failed a Ranchi job); model told its sentence budget; YOLO11n vehicle count in the processor; WAQI lookup fix + caveat changes; PR #3 merged, PR #6 open.
Remaining: **seasonal lookup script + JSON** (§6.3), smoke/fire boxes (YOLO fine-tune), PWA deploy, **own test clips (the smoke model's real test)**, final names, **merging PRs #1 then #2**.

## 12. Open items

- Future work: learned indoor/outdoor classifier for the quality gate (bright indoor scenes pass the rule-based gate, §7.1).
- Exact Sunday submission deadline (time) — Schedule page / Discord.
- Final team + product names.
- Anthropic models on Bedrock blocked (Marketplace subscription not permitted for `hackathon-dev`); using Nova 2 Lite.
- Owners for the PWA and the SAM/AWS deploy.
- Later, for the semester project: native app, 3-stream ablation, own labelled Indian dataset.
