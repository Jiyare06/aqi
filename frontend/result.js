// Result screen, built from backend/contract.json (`result`) plus GET /jobs' presigned `audio_urls` / `gradcam_url`.
(function () {
  const { category, esc, parts, clock, ago } = window.fmt;

  const PROFILES = { school: "School", worker: "Outdoor worker", general: "General" };
  const PROFILE_HOURS = { school: [7, 17], worker: [6, 19], general: [6, 21] };  // backend combine.PROFILE_WINDOWS
  const VERDICT = { yes: "Outdoors OK", limit: "Limit outdoor time", no: "Avoid outdoors" };
  const REASONS = {
    aqi_le_100: "AQI stays at or below 100 (Good/Satisfactory) through these hours.",
    aqi_101_200: "AQI reaches 101–200 (Moderate) in these hours: keep outdoor time short and light, especially for children and people with asthma.",
    aqi_gt_200: "AQI goes above 200 (Poor or worse) in these hours.",
    no_window_today: "No 3-hour slot with AQI at or below 100 is left in these hours today.",
    no_window_tomorrow: "No 3-hour slot with AQI at or below 100 in these hours tomorrow.",
  };

  const hourText = (h) => clock(`2000-01-01T${String(h).padStart(2, "0")}:00`);
  // CPCB category colour class from the AQI number (DESIGN.md §5); colour always comes with the name/number.
  const catClass = (aqi) => (aqi == null ? "" : `c-${category(aqi).toLowerCase().replace(" ", "-")}`);

  // Hindi counterparts for the decision and share text.
  const HI = {
    profiles: { school: "स्कूल", worker: "बाहर काम करने वाले", general: "सामान्य" },
    verdict: { yes: "बाहर जाना ठीक है", limit: "बाहर कम समय बिताएँ", no: "बाहर जाने से बचें" },
    category: { Good: "अच्छी", Satisfactory: "संतोषजनक", Moderate: "मध्यम", Poor: "ख़राब", "Very Poor": "बहुत ख़राब", Severe: "गंभीर" },
    confidence: { high: "ज़्यादा भरोसा", medium: "मध्यम भरोसा", low: "कम भरोसा" },
  };

  // "5 PM" -> "शाम 5 बजे" (from the ISO wall-clock time).
  function hiClock(iso) {
    const p = parts(iso);
    if (!p) return "";
    const part = p.h < 4 ? "रात" : p.h < 12 ? "सुबह" : p.h < 16 ? "दोपहर" : p.h < 20 ? "शाम" : "रात";
    const h12 = p.h % 12 === 0 ? 12 : p.h % 12;
    return `${part} ${h12}${p.min ? ":" + String(p.min).padStart(2, "0") : ""} बजे`;
  }
  const hourIso = (h) => `2000-01-01T${String(h).padStart(2, "0")}:00`;

  // Verdict + window line from decision.{outdoor, for, best_window}; used by the card (en) and share text (en/hi).
  function decisionWords(r, lang = "en") {
    const d = r.decision;
    const hi = lang === "hi";
    const t = hi ? hiClock : clock;
    const [h0, h1] = PROFILE_HOURS[r.profile] || [];
    const tomorrow = d.for === "tomorrow";
    let windowLine = hi ? `${tomorrow ? "कल" : "आज"} कोई अच्छा समय नहीं` : `No good window ${tomorrow ? "tomorrow" : "today"}`;
    if (d.outdoor === "yes" && h1 != null) {
      // "yes" = every slot left in the hours is fine, so the first-slot best_window would undersell it.
      windowLine = tomorrow
        ? (hi ? `पूरे दिन ठीक, ${t(hourIso(h0))}–${t(hourIso(h1))}` : `Fine all day, ${t(hourIso(h0))}–${t(hourIso(h1))}`)
        : (hi ? `अभी से ${t(hourIso(h1))} तक ठीक` : `Fine from now until ${t(hourIso(h1))}`);
    } else if (d.best_window) {
      const start = parts(d.best_window.start), end = parts(d.best_window.end);
      // The backend clips today's first slot to the current time (e.g. 11:17); show that as "now".
      const now = !tomorrow && start && start.min !== 0;
      const from = now ? (hi ? "अभी" : "now") : t(d.best_window.start);
      if (end && end.h === h1 && end.min === 0) {
        windowLine = hi ? (now ? "अभी से बाहर जाना ठीक" : `${from} के बाद बाहर जाना ठीक`)
          : (now ? "Outdoors fine from now" : `Outdoors after ${from}`);
      } else {
        windowLine = hi ? `सबसे अच्छा समय: ${from}–${t(d.best_window.end)}` : `Best window: ${from}–${t(d.best_window.end)}`;
      }
    }
    const verdict = (hi ? HI.verdict : VERDICT)[d.outdoor] || d.outdoor;
    return { verdict, windowLine };
  }

  function decisionCard(r) {
    const d = r.decision;
    if (!d) return "";
    const day = d.for === "tomorrow" ? "Tomorrow" : "Today";
    const [h0, h1] = PROFILE_HOURS[r.profile] || [];
    const hours = h0 != null ? ` · ${PROFILES[r.profile]} hours ${hourText(h0)}–${hourText(h1)}` : "";
    const { verdict, windowLine } = decisionWords(r);
    return `
      <div class="card decision ${esc(d.outdoor)}">
        <p class="when eyebrow">${day}${esc(hours)}</p>
        <p class="verdict"><strong>${esc(verdict)}</strong></p>
        <p class="window">${esc(windowLine)}</p>
        <p class="reason">${esc(REASONS[d.reason_code] || d.reason_code || "")}</p>
      </div>`;
  }

  function sourceLine(air) {
    if (air.used === "measured" && air.measured) {
      const m = air.measured;
      return `Measured: ${esc(m.station)} · ${esc(m.dist_km)} km away · updated ${esc(ago(m.updated_at))}`;
    }
    if (air.modelled) return `Modelled: CAMS model, ~${esc(air.modelled.grid_km ?? 45)} km grid`;
    return "";
  }

  function otherLine(air) {
    const o = air.used === "measured" ? air.modelled : air.measured;
    if (!o || o.aqi_cpcb == null) return "";
    const what = air.used === "measured"
      ? "CAMS model for this area"
      : `Nearest station (${esc(o.station)}, ${esc(o.dist_km)} km, updated ${esc(ago(o.updated_at))})`;
    return `<p class="muted">${what}: AQI ${esc(o.aqi_cpcb)} (${esc(category(o.aqi_cpcb))})</p>`;
  }

  function aqiCard(r) {
    const air = r.air;
    if (!air || air.aqi_cpcb == null) return "";
    const chosen = air.used === "measured" ? air.measured : air.modelled;
    const pm = chosen
      ? [chosen.pm25 != null && `PM2.5 ${esc(chosen.pm25)}`, chosen.pm10 != null && `PM10 ${esc(chosen.pm10)}`].filter(Boolean).join(" · ")
      : "";
    const caveats = (r.caveats || []).filter(Boolean);
    return `
      <div class="card">
        <h2>Air now</h2>
        <div class="aqi-head">
          <span class="aqi-num">${esc(air.aqi_cpcb)}</span>
          <span class="pill ${catClass(air.aqi_cpcb)}">${esc(category(air.aqi_cpcb))}</span>
          <span class="muted">CPCB-style AQI</span>
        </div>
        <div class="kv">
          <p>${sourceLine(air)}</p>
          ${pm ? `<p class="muted">${pm} µg/m³</p>` : ""}
          <p><span class="badge conf-${esc(air.confidence)}">Confidence: ${esc(air.confidence)}</span>
            ${air.confidence_reason ? `${esc(air.confidence_reason)}` : ""}</p>
          ${otherLine(air)}
          ${r.seasonal_line ? `<p>${esc(r.seasonal_line)}</p>` : ""}
        </div>
        ${caveats.length ? `<p class="caveat">${esc(caveats[0])}</p>` : ""}
        ${caveats.length > 1 ? `<details class="caveats"><summary>Details (${caveats.length - 1})</summary>
          <ul>${caveats.slice(1).map((c) => `<li>${esc(c)}</li>`).join("")}</ul></details>` : ""}
      </div>`;
  }

  function strip(title, slots, scale) {
    if (!slots || !slots.length) return "";
    const bars = slots.map((s) => {
      const pct = Math.max(4, Math.min(100, (s.aqi_cpcb / scale) * 100));
      return `<div class="slot" title="${esc(clock(s.time))}: AQI ${esc(s.aqi_cpcb)} (${esc(category(s.aqi_cpcb))})">
          <span class="val">${esc(s.aqi_cpcb)}</span>
          <div class="bar ${catClass(s.aqi_cpcb)}" style="height:${pct}%"></div>
        </div>`;
    }).join("");
    const labels = slots.map((s) => `<span>${esc(clock(s.time).replace(":00", ""))}</span>`).join("");
    return `<div class="strip-day"><h3>${title}</h3>
        <div class="strip" role="img" aria-label="${title} AQI every 3 hours">${bars}</div>
        <div class="strip-labels">${labels}</div></div>`;
  }

  function forecastCard(r) {
    const today = r.forecast_today || [];
    const tomorrow = r.forecast_tomorrow || [];
    if (!today.length && !tomorrow.length) return "";
    const scale = Math.max(200, ...today.concat(tomorrow).map((s) => s.aqi_cpcb || 0));
    const legend = [25, 75, 150, 250, 350, 450].map((a) => `<span><i class="${catClass(a)}"></i>${esc(category(a))}</span>`).join("");
    return `
      <div class="card">
        <h2>Forecast (3-hour slots)</h2>
        ${strip("Today", today, scale)}
        ${strip("Tomorrow", tomorrow, scale)}
        <div class="legend">${legend}</div>
      </div>`;
  }

  function causeRow(name, c) {
    if (!c) return "";
    // Classifier output, not a confirmed detection (backend: present only at a score >= 0.60): never "Likely"/"Detected".
    const [label, cls] = c.present ? ["Possible", "possible"] : ["Not seen", "muted"];
    const score = c.prob != null ? ` (model score ${Math.round(c.prob * 100)}%)` : "";
    return `<div class="cause"><span>${name}</span>
      <span class="${cls}">${label}${score}</span></div>`;
  }

  // Quality-gate reject codes (processor) in plain words; unknown codes are shown with spaces.
  const REJECT_WORDS = { dark: "too dark", blur: "too blurry", no_sky: "no sky or street in view" };
  const GATE_MIN_FRAMES = 3;  // the processor skips causes when fewer frames pass the gate
  const GATE_CAVEAT = "Couldn't check what's in this clip";
  const plainReason = (code) => REJECT_WORDS[String(code).toLowerCase()] || String(code).replace(/_/g, " ");

  // The processor skipped causes: too few frames passed the gate, or its caveat says so.
  function gateBlocked(r) {
    const c = r.causes;
    if (c.frames_used < GATE_MIN_FRAMES && c.frames_rejected > 0) return true;
    return (r.caveats || []).some((t) => String(t).startsWith(GATE_CAVEAT));
  }

  function causesCard(r, camUrl) {
    const c = r.causes;
    if (!c) return "";
    const reasons = (c.reject_reasons || []).filter(Boolean);
    if (gateBlocked(r)) {
      const words = [...new Set(reasons.map(plainReason))];
      return `
      <div class="card">
        <h2>What's in your video <span class="exp">(experimental)</span></h2>
        <p>Couldn't check this clip${words.length ? `: ${esc(words.join(", "))}` : ""}. Point the camera at the sky or street outdoors.</p>
      </div>`;
    }
    const frames = c.frames_used != null
      ? `<p class="muted">Frames analysed: ${esc(c.frames_used)}${c.frames_rejected ? `, rejected ${esc(c.frames_rejected)}` : ""}${reasons.length ? ` (${esc(reasons.join(", "))})` : ""}</p>`
      : "";
    const vehicles = c.vehicle_count != null
      ? `<div class="cause"><span>Vehicles in view</span><span>${esc(c.vehicle_count)}</span></div>` : "";
    const cam = camUrl
      ? `<img class="cam" src="${esc(camUrl)}" alt="Heatmap of where the model saw smoke or fire" loading="lazy">
         <p class="muted">Highlighted: where the model saw it (CAM).</p>` : "";
    return `
      <div class="card">
        <h2>What's in your video <span class="exp">(experimental)</span></h2>
        ${causeRow("Smoke", c.smoke)}
        ${causeRow("Fire", c.fire)}
        ${vehicles}
        ${cam}
        <p class="note">On our 23 test clips this found smoke in 6 of 7, but also flagged 5 of 13 hazy skies with no smoke. Treat it as a hint.</p>
        ${frames}
        <p class="muted">This only shows possible sources. It never changes the AQI.</p>
      </div>`;
  }

  function adviceCard(r, audio) {
    const a = r.advice;
    if (!a || (!a.en && !a.hi)) return "";
    const langs = [["en", "English", a.en], ["hi", "हिन्दी", a.hi]].filter(([, , text]) => text);
    const tabs = langs.map(([k, label], i) =>
      `<button type="button" role="tab" id="tab-${k}" aria-controls="panel-${k}" aria-selected="${i === 0}" lang="${k}">${label}</button>`).join("");
    const panels = langs.map(([k, , text], i) => {
      const url = audio && audio[k];
      return `<div role="tabpanel" id="panel-${k}" aria-labelledby="tab-${k}" ${i === 0 ? "" : "hidden"}>
          <p class="advice-text" lang="${k}">${esc(text)}</p>
          ${url ? `<audio controls preload="none" src="${esc(url)}"></audio>` : ""}
        </div>`;
    }).join("");
    const source = a.source === "bedrock" ? "Written by Amazon Bedrock from the numbers above." : "Template advice from the numbers above.";
    return `
      <div class="card advice">
        <h2>What to do</h2>
        <div class="tabs" role="tablist">${tabs}</div>
        ${panels}
        <p class="muted">${source}</p>
      </div>`;
  }

  // `picked`: {city} from the dropdown or {gps: true} (app.js); absent for mock results.
  function header(r, picked) {
    const loc = r.location || {};
    const coords = loc.lat != null ? `${(+loc.lat).toFixed(3)}, ${(+loc.lon).toFixed(3)}` : "";
    const place = loc.place || (picked && picked.city)
      || (picked && picked.gps ? `Your location${coords ? ` (${coords})` : ""}` : coords);
    return `<p class="eyebrow meta">${esc(place)}${place ? " · " : ""}${esc(PROFILES[r.profile] || "")}</p>`;
  }

  // --- Share (never mentions the video causes) -------------------------------

  function shareText(r, picked, lang) {
    const hi = lang === "hi";
    const air = r.air || {};
    const place = (r.location && r.location.place) || (picked && picked.city) || (hi ? "आपकी जगह" : "your location");
    const cat = category(air.aqi_cpcb);
    const lines = [];
    if (air.aqi_cpcb != null) {
      lines.push(hi ? `${place} के पास आज की हवा: ${HI.category[cat]} (AQI ${air.aqi_cpcb})।`
        : `Air near ${place} today: ${cat} (AQI ${air.aqi_cpcb}).`);
    }
    if (r.decision) {
      const { verdict, windowLine } = decisionWords(r, lang);
      const who = (hi ? HI.profiles : PROFILES)[r.profile] || "";
      const when = r.decision.for === "tomorrow" ? (hi ? " (कल)" : " (tomorrow)") : "";
      lines.push(hi ? `${who}${when}: ${verdict}। ${windowLine}।` : `${who}${when}: ${verdict}. ${windowLine}.`);
    }
    const conf = air.confidence ? (hi ? HI.confidence[air.confidence] : `${air.confidence} confidence`) : "";
    let src = "";
    if (air.used === "measured" && air.measured) {
      src = hi ? `स्टेशन ${air.measured.station}, ${air.measured.dist_km} किमी` : `station ${air.measured.station}, ${air.measured.dist_km} km`;
    } else if (air.used === "modelled") {
      src = hi ? "CAMS मॉडल" : "CAMS model";
    }
    if (src) lines.push(hi ? `स्रोत: ${src}${conf ? `, ${conf}` : ""}।` : `Source: ${src}${conf ? `, ${conf}` : ""}.`);
    const url = window.location.origin + window.location.pathname;
    return `${lines.join(" ")} — HazeBusters ${url}`;
  }

  const shareBlock = () => `
      <div class="share">
        <button type="button" id="btn-share" class="btn secondary">Share</button>
        <a id="share-wa" class="btn secondary" href="#" target="_blank" rel="noopener">WhatsApp</a>
        <span id="share-status" class="muted" aria-live="polite"></span>
      </div>`;

  async function copy(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    const ta = Object.assign(document.createElement("textarea"), { value: text });
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    if (!ok) throw new Error("copy failed");
  }

  function bindShare(root, r, picked) {
    const btn = root.querySelector("#btn-share");
    if (!btn) return;
    const lang = () => (root.querySelector('#tab-hi[aria-selected="true"]') ? "hi" : "en");
    const status = root.querySelector("#share-status");
    const wa = root.querySelector("#share-wa");
    const updateWa = () => { wa.href = `https://wa.me/?text=${encodeURIComponent(shareText(r, picked, lang()))}`; };
    updateWa();
    root.querySelectorAll('[role="tab"]').forEach((t) => t.addEventListener("click", updateWa));
    btn.addEventListener("click", async () => {
      const text = shareText(r, picked, lang());
      status.textContent = "";
      if (navigator.share) {
        try {
          await navigator.share({ text });
          return;
        } catch (e) {
          if (e.name === "AbortError") return;  // user closed the share sheet
        }
      }
      try {
        await copy(text);
        status.textContent = "Copied";
      } catch (_) {
        status.textContent = "Couldn't copy. Use WhatsApp instead.";
      }
    });
  }

  function bindTabs(root) {
    root.querySelectorAll('[role="tab"]').forEach((tab) => {
      tab.addEventListener("click", () => {
        root.querySelectorAll('[role="tab"]').forEach((t) => {
          const on = t === tab;
          t.setAttribute("aria-selected", String(on));
          root.querySelector(`#${t.getAttribute("aria-controls")}`).hidden = !on;
          if (!on) root.querySelectorAll(`#${t.getAttribute("aria-controls")} audio`).forEach((el) => el.pause());
        });
      });
    });
  }

  // `job` = GET /jobs/{id} body {result, audio_urls, gradcam_url}, or a bare contract result (mock).
  function render(root, job, picked) {
    const r = job.result || job;
    root.innerHTML = header(r, picked) + decisionCard(r) + (r.decision || r.air ? shareBlock() : "") + aqiCard(r)
      + forecastCard(r) + causesCard(r, job.gradcam_url) + adviceCard(r, job.audio_urls);
    bindTabs(root);
    bindShare(root, r, picked);
  }

  window.renderResult = render;
})();
