# HazeBusters — how bad is the air where there's no monitor?

**Problem.** Ranchi has no official air-quality monitor. From Ranchi city, the nearest official station is Jamshedpur, **~110 km** away, and it isn't reporting to WAQI either: on 11 Oct 2026 WAQI's geo feed said "can not connect" and no reporting station showed within 25 km. Big cities go dark too: all **9 WAQI stations in Hyderabad last reported on 23 Jun 2026**. A school deciding whether today's assembly or PE can be outdoors has nothing to go on.

**What we built** (a phone web app):
1. **Air.** Your location → the nearest WAQI station (US AQI converted back to µg/m³) plus the CAMS forecast via Open-Meteo (~45 km grid) → a CPCB-style AQI. The page always shows **the source, its distance and our confidence**. If the station is more than 25 km away or more than 3 h old, it uses CAMS ×1.25.
2. **Decision.** For a school or an outdoor worker: outdoor *yes / limit / no*, from the worst 3-hour slot left in their hours, plus the **best window** today (or tomorrow).
3. **Advice** in **English and Hindi**, with **audio**, ready to share. The LLM only phrases numbers we computed, with a template fallback.
4. **Clip (experimental).** Quality gate → smoke/fire classifier with a CAM heatmap → YOLO11n vehicle count. Causes never change the AQI; the advice line is hedged ("possible smoke").

**AWS.** S3, Lambda (a container image for the processor, plus API handlers), DynamoDB, API Gateway (HTTP API), **Amazon Bedrock** (Nova 2 Lite, Converse), **Amazon Polly** (Hindi and English), Amplify Hosting, SAM / CloudFormation, ECR. All in ap-south-1.

**Results (honest).**

| Test | Result |
|---|---|
| Image → AQI band (EfficientNet-B0 on TRAQID, unseen days) | **32.0%** vs a **47.5%** majority baseline (89% on training days): it learns the day, not the haze → **dropped** |
| Random frame split vs leave-one-day-out | 63.3% vs 33.5%: a random split inflates accuracy by **~30 points** |
| CAMS vs ground stations, daily PM2.5 2023–24 | ~0.8× the stations (median 0.78 / 0.79 / 0.81 for Delhi / Hyderabad / Patna); Very Poor/Severe days caught: 67/201 Delhi, 46/133 Patna |
| CAMS ×1.25 (in-sample) | Same band: Delhi 56.2→63.2%, Patna 62.5→67.6%, **Hyderabad 94.8→86.3% (worse)**. Very Poor/Severe caught: 125/201 and 80/133; false alarms 1→10 and 8→20 |
| Smoke/fire classifier, D-Fire test (grouped split) | AP **smoke 0.979**, fire 0.982 |
| Same model on Pyro-SDIS (distant wildfire cameras) | Smoke recall 13.4%; AP 0.845, **below the 0.868 base rate** → doesn't transfer |
| Own phone clips at smoke threshold 0.600 (23 clips, duplicate excluded) | Smoke **6/7**, fire 2/2, indoor clip rejected; **5/13 no-smoke clips flagged** (city haze, bright sky) |

**Limitations.**
- **AQI method:** "CPCB-style" AQI from hourly PM, not the official 24-hour, 3-or-more-pollutant AQI.
- **WAQI conversion:** assumes the 2012 EPA PM2.5 breakpoints.
- **CAMS correction:** CAMS is a 45 km model, and ×1.25 was fitted on three cities and PM2.5 only.
- **Smoke model:** flags city haze and is tested on just 23 own clips.
- **Quality gate:** calibrated on 24 clips with **thin margins** (the indoor clip kept 2 frames against the 3 needed; `fire_smoke_01` passes 9/15).
- **Health wording:** only CPCB's Severe line is confirmed against an official source.
- **Licences:** TRAQID, the India/Nepal images and the Open-Meteo free tier are non-commercial.

**Roadmap.**
- **Data flywheel:** opt-in clips paired with the nearest reading → an Indian haze/smoke set → retrain with haze as a negative, and recalibrate the gate and correction per region.
- **School daily alerts** with today's window.
- **More languages:** Bengali, Tamil, Telugu, Marathi.

**AI tools used.** Claude Code (code, experiments, docs) and Claude chat (planning, review).

**Credits.**
- TRAQID (Kathalkar et al., ICVGIP 2024).
- India/Nepal air-pollution images (Kaggle, CC BY-NC-SA 3.0 IGO) and Eff-AQI (Utomo et al., GoodIT '23).
- D-Fire (Venâncio et al. 2022, CC0).
- Pyro-SDIS (Pyronear, Apache-2.0).
- India AQI 2010–2024 stations (Kaggle, omsandeeppatil, CC BY-SA 4.0).
- CAMS (Copernicus) via Open-Meteo.
- World Air Quality Index project (aqicn.org).
- Ultralytics YOLO11n (AGPL-3.0).
- PyTorch / torchvision.
- Code: AGPL-3.0.
