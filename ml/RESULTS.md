# ML results

All numbers are copied from files in `ml/runs/` (gitignored; regenerate with the commands below).

## 1. Image → AQI band: negative result (Thu 8 Oct 2026)

**Question:** can a phone/dashcam image tell the CPCB-style AQI band (Good/Satisfactory · Moderate/Poor · Very Poor/Severe) on a day or place the model hasn't seen?
**Answer on our data: no.** This is why the product doesn't estimate AQI from images (CLAUDE.md §3/§4).

Data: TRAQID day/front (13,789 frames, **17 dates**, Hyderabad dashcam); India/Nepal `All_img` (12,240 frames, 115 moments, 8 locations incl. Biratnagar, Nepal). Labels recomputed from PM2.5/PM10 with `hazeml.aqi`.

### 1a. Fine-tuned EfficientNet-B0, frozen date split
Run: `ml/runs/20261008-174356_efficientnet_b0_s0/` (`uv run python -m hazeml.train --seed 0`).

| split | accuracy | majority baseline | macro-F1 | band-2 recall |
|---|---|---|---|---|
| test (5 dates, 2,771 frames) | **32.0%** | 47.5% | 0.312 | 57.3% |
| val (2 dates, 1,361 frames) | 22.2% | 47.4% | 0.175 | 2.7% |
| its own train dates (eval preprocessing) | 89.0% | 47.2% | 0.881 | 95.9% |

Mean confidence on test 0.80 (confidently wrong). Per test date: 50.3% / 42.9% / **9.4%** / 43.4% / 15.1%. 89% on train dates rules out a pipeline bug: the model memorises each day's look, not haze. Seeds 1–2 were stopped once this was clear.

### 1b. Probes (no CNN training; `uv run python -m hazeml.probes features && ... run`)
Logistic regression, balanced class weights, standardised; C fixed (0.1 embeddings, 1.0 handcrafted), not tuned. Baseline = majority class of each training fold. "Group" = majority vote per held-out date (TRAQID) or moment (India/Nepal). Source: `ml/runs/probes/summary.md`.

| probe | task | level | n | macro-F1 | accuracy | baseline | top-class recall |
|---|---|---|---|---|---|---|---|
| frozen B0 embeddings, leave-one-date-out | 3-band | frame | 13789 | 0.318 | 33.5% | 33.7% | 33.6% |
| | 3-band | date | 17 | 0.373 | 41.2% | 47.1% | 33.3% |
| | VP/Severe vs rest | frame | 13789 | 0.512 | 63.2% | 83.4% | 40.9% |
| | VP/Severe vs rest | date | 17 | 0.595 | 76.5% | 82.4% | 33.3% |
| handcrafted (dark channel, contrast, saturation, brightness), LODO | 3-band | frame | 13789 | 0.336 | 36.1% | 33.7% | 46.4% |
| | 3-band | date | 17 | 0.342 | 41.2% | 47.1% | 66.7% |
| | VP/Severe vs rest | frame | 13789 | 0.469 | 51.6% | 83.4% | 66.1% |
| | VP/Severe vs rest | date | 17 | 0.564 | 58.8% | 82.4% | 100.0% |
| handcrafted + humidity, LODO | 3-band | frame | 13789 | 0.464 | 52.8% | 33.7% | 28.5% |
| | 3-band | date | 17 | 0.398 | 47.1% | 47.1% | 0.0% |
| | VP/Severe vs rest | frame | 13789 | 0.497 | 56.2% | 83.4% | 60.6% |
| | VP/Severe vs rest | date | 17 | 0.530 | 58.8% | 82.4% | 66.7% |
| frozen B0 embeddings, **random 5-fold over frames** (leaky) | 3-band | frame | 13789 | 0.601 | 63.3% | 47.3% | 59.2% |
| | 3-band | date | 17 | 1.000 | 100.0% | 52.9% | 100.0% |
| | VP/Severe vs rest | frame | 13789 | 0.663 | 76.2% | 83.4% | 66.1% |
| | VP/Severe vs rest | date | 17 | 1.000 | 100.0% | 82.4% | 100.0% |
| India/Nepal frozen B0 embeddings, leave-one-city-out | 3-band | frame | 12240 | 0.204 | 20.0% | 2.8% | 8.6% |
| | 3-band | moment | 115 | 0.144 | 13.0% | 3.5% | 2.4% |
| | VP/Severe vs rest | frame | 12240 | 0.293 | 34.8% | 60.7% | 8.6% |
| | VP/Severe vs rest | moment | 115 | 0.266 | 33.0% | 63.5% | 4.8% |

**Takeaways**
- 65% of TRAQID's band variance (70% of AQI variance) is **between dates** → ~17 independent samples.
- Leaving out whole dates, image features sit at the baseline. Humidity (a weather reading) adds frame-level signal, but date-level accuracy stays at the baseline and no Very Poor/Severe day is found.
- The **random frame split inflates accuracy by ~30 points** (63.3% vs 33.5%) — a random split would have produced a convincing false result.
- India/Nepal leave-one-city-out is **inverted** (clean cities called polluted and vice versa; Delhi 4.5%): each city is nearly one band, so the model learns the city. The per-fold majority baseline is degenerate there (held-out city's band is never the training majority); against 3-class chance (~33%) 20% is still below.

## 2. CAMS (Open-Meteo) vs ground stations, daily PM2.5, 2023–2024 (Fri 9 Oct 2026)

Script: `ml/validate_cams.py` (`uv run python validate_cams.py`). Output: `ml/runs/cams_validation/summary.json` + `{city}_daily.csv`.
Ground truth: the stations dataset (Kaggle "India AQI 2010–2024", CC BY-SA 4.0), 15-min PM2.5 → station-day mean (≥ 18 hours with data) → city-day mean over stations. CAMS: Open-Meteo air-quality API, `domains=cams_global` (0.4°), hourly PM2.5 at the mean station location, daily mean (≥ 18 hours). Bands: shared CPCB-style function on PM2.5 only.

| city | stations | days | same 3-band | same 6-category | bias mean / median (CAMS − station, µg/m³) | median ratio CAMS/station | MAE | Pearson r | median PM2.5 station / CAMS |
|---|---|---|---|---|---|---|---|---|---|
| Delhi | 43 | 731 | 56.2% | 42.3% | −27.9 / −17.2 | 0.78 | 36.3 | 0.79 | 70 / 55 |
| Hyderabad | 12 | 731 | 94.8% | 75.0% | −6.6 / −6.9 | 0.79 | 9.5 | 0.71 | 33 / 27 |
| Patna | 4 | 731 | 62.5% | 47.2% | −19.5 / −10.6 | 0.81 | 25.2 | 0.79 | 68 / 49 |

Band confusion (rows = station band, columns = CAMS band; 0 = Good/Satisfactory, 1 = Moderate/Poor, 2 = Very Poor/Severe):

| city | station 0 → CAMS 0/1/2 | station 1 → CAMS 0/1/2 | station 2 → CAMS 0/1/2 |
|---|---|---|---|
| Delhi | 260 / 44 / 0 | 141 / 84 / 1 | 15 / 119 / **67** |
| Hyderabad | 689 / 21 / 0 | 15 / 4 / 0 | 2 / 0 / 0 |
| Patna | 304 / 20 / 0 | 159 / 107 / 8 | 5 / 82 / **46** |

**Takeaways**
- **CAMS runs low**: about 0.8× the stations (median), and it **misses most bad days** — of the station days that were Very Poor/Severe, CAMS put only 67/201 (33%) in Delhi and 46/133 (35%) in Patna in that band. It almost never over-calls.
- It follows the day-to-day ups and downs reasonably (r 0.71–0.79), so the shape of the forecast is useful; the level is biased low.
- **Hyderabad's 94.8% is mostly base rate**: 97.1% of its days are band 0, so "always band 0" would score higher. Don't quote it as skill.
- Product consequence: when the answer is **modelled** (e.g. Ranchi), the raw CAMS decision can be too lenient on bad days → the backend now scales CAMS ×1.25 whenever `air.used == "modelled"` (see 2b for what that costs).
- Caveats: city-day means smooth over local hotspots; stations sit in cities while a 45 km CAMS cell mixes in cleaner surroundings, which explains part of the low bias. Hyderabad and Patna have fewer stations per day (median 9 and 4; minimum 2 and 1).

### 2b. With the ×1.25 correction (re-run Fri 9 Oct 2026, same days and stations)

`validate_cams.py` now scores CAMS raw and ×1.25 (`CORRECTION`, = `CAMS_CORRECTION` in `backend/src/air/combine.py`). Source: `ml/runs/cams_validation/summary.json`. "VP/Severe caught" = station Very Poor/Severe days that CAMS also put in that band; "false alarms" = CAMS Very Poor/Severe on days the station wasn't.

| city | CAMS | same 3-band | same 6-category | VP/Severe caught | VP/Severe false alarms | bias mean / median (µg/m³) | MAE | median ratio |
|---|---|---|---|---|---|---|---|---|
| Delhi | raw | 56.2% | 42.3% | 67 / 201 | 1 | −27.9 / −17.2 | 36.3 | 0.78 |
| Delhi | ×1.25 | **63.2%** | 45.4% | **125 / 201** | 10 | −11.4 / −2.4 | 31.0 | 0.97 |
| Hyderabad | raw | **94.8%** | **75.0%** | 0 / 2 | 0 | −6.6 / −6.9 | **9.5** | 0.79 |
| Hyderabad | ×1.25 | 86.3% | 73.1% | 0 / 2 | 0 | +0.6 / −0.3 | 9.7 | 0.99 |
| Patna | raw | 62.5% | 47.2% | 46 / 133 | 8 | −19.5 / −10.6 | 25.2 | 0.81 |
| Patna | ×1.25 | **67.6%** | 50.8% | **80 / 133** | 20 | −5.1 / +0.3 | 22.0 | 1.01 |

Band confusion with ×1.25 (rows = station, cols = CAMS): Delhi `[[199,104,1],[79,138,9],[7,69,125]]`, Hyderabad `[[620,90,0],[8,11,0],[2,0,0]]`, Patna `[[258,66,0],[98,156,20],[1,52,80]]`.

**Verdict**
- **The correction hurts Hyderabad.** Band agreement falls from 94.8% to 86.3% and 6-category agreement from 75.0% to 73.1%; MAE rises slightly (9.5 → 9.7). Cause: 90 of 710 Good/Satisfactory days are now called Moderate/Poor (raw: 21). In a clean city the ×1.25 pushes borderline days over the AQI-100 line, which turns "yes" into "limit".
- **It helps Delhi and Patna**: band agreement +7.0 / +5.1 points, Very Poor/Severe caught 67 → 125 (Delhi) and 46 → 80 (Patna), MAE down. The price is more over-calls: Very Poor/Severe false alarms 1 → 10 (Delhi) and 8 → 20 (Patna), and in Delhi Good/Satisfactory days called Moderate/Poor rise 44 → 104.
- Median bias goes to about zero in all three cities, but **this is in-sample**: the factor was chosen from these same three cities' ratios, so these numbers flatter it. It has not been checked on any other city.
- Net for the product: the correction trades leniency on bad days for over-caution on clean days. For a "should the kids go out" decision that trade is defensible, but say so in the caveat/writeup rather than presenting ×1.25 as accurate. A city- or level-dependent correction would need more cities than we have.

## 3. Smoke/fire classifier (Fri 9 Oct 2026)

Run: `ml/runs/smokefire/20261009-220554_efficientnet_b0_s0/` (`uv run python -m hazeml.train_smokefire`); numbers below are from its `train.log`, `report.json` / `report.md` and `export_report.json`.

**Data and split.** D-Fire (CC0 1.0, 21,527 images; downloaded from the Hugging Face mirror `badsaarow/d-fire` because the official OneDrive links refuse scripted downloads; every per-class image and box count matches the D-Fire README). Split v2 in `ml/splits/dfire.json`: dHash groups (Hamming ≤ 6) plus consecutive `AoF` video frames linked at Hamming ≤ 15, with a 10% group cap (hit 4 times; 4 of 6,051 links skipped). Largest group 1,882 images (8.7%).

| split | clear | smoke | fire | smoke + fire | total |
|---|---|---|---|---|---|
| train | 7,318 | 4,271 | 943 | 3,317 | 15,849 |
| val | 1,260 | 798 | 108 | 669 | 2,835 |
| test | 1,260 | 798 | 113 | 672 | 2,843 |

Leak check: 612 of 8,383 consecutive `AoF` frame pairs sit in different splits, **none** within Hamming distance 15 (min bin 16–23: 114; median 29) — v1 had 782, 88 of them at distance 7–15.

**Setup.** EfficientNet-B0 (ImageNet), 2 sigmoid outputs (smoke, fire), BCEWithLogits with pos_weight from train (smoke 1.09, fire 2.72), whole frame squashed to 224×224, flip + RandomResizedCrop 0.6–1 + moderate colour jitter, bf16, batch 32, AdamW (head 1e-3, backbone 1e-4), early stopping on val mean AP (patience 4), seed 0. Stopped after epoch 10; best epoch 6 (val AP smoke 0.987, fire 0.979). Per-label thresholds = max F1 on val: smoke 0.359, fire 0.667.

**D-Fire test (2,843 images, grouped split):**

| label | positives / n | AP | precision | recall | F1 |
|---|---|---|---|---|---|
| smoke | 1,470 / 2,843 | 0.979 | 91.6% | 93.4% | 0.925 |
| fire | 785 / 2,843 | 0.982 | 94.2% | 95.0% | 0.946 |

Exact 4-way match 89.5%. Confusion (rows = true, cols = predicted; clear / smoke / fire / smoke+fire):
clear `1194 / 58 / 4 / 4` · smoke `87 / 673 / 0 / 38` · fire `5 / 6 / 44 / 58` · smoke+fire `0 / 28 / 10 / 634`.
Weak spot: **fire-only images** (113) are mostly called smoke+fire (58) — the fire flag is right, the extra smoke flag isn't. 87 smoke-only images are missed as clear.

**Pyro-SDIS, external test (12,378 images from 209 fixed wildfire cameras, smoke label only, val-picked smoke threshold):**
- Overall: AP 0.845, precision 81.1%, **recall 13.4%** (1,436 / 10,744), **false-positive rate 20.5%** on its 1,634 negatives (335).
- **The AP is below chance**: 86.8% of Pyro-SDIS images contain smoke, so a random score already gets AP ≈ 0.868.
- By camera (macro): mean recall 27.0% over 118 cameras with smoke; mean FPR 20.6% over 174 cameras with negatives; mean AP 0.804 over 83 cameras with both.
- Reading: the model **does not transfer** to distant, small wildfire plumes on wide-angle mountain cameras (they shrink to a few pixels at 224×224), and Pyro's negatives are hard look-alikes (clouds, haze). D-Fire's test numbers are in-domain and optimistic for our use; neither dataset shows Indian urban burning (garbage, crop residue), so our own clips are the real test.

**ONNX** (`smokefire.onnx`, 15.3 MB, opset 17; outputs `features` 1280×7×7 and `logits`): on 64 D-Fire test images, max |ONNX − torch| logits **2.1e-4**, features 1.2e-3. `ml/src/hazeml/cam.py` (numpy only) reproduces the ONNX logits from features + `head.npz` (GAP identity) to 5.7e-6. Six CAM overlays in `cam/` (not committed).
