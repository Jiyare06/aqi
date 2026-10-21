// Screens and flow: input -> progress (create, upload, analyse) -> result.
(function () {
  const $ = (sel) => document.querySelector(sel);
  const screens = { input: $("#screen-input"), progress: $("#screen-progress"), result: $("#screen-result") };
  let coords = null;     // {lat, lon} from geolocation
  let lastJob = null;    // inputs of the last submit, for Retry

  function show(name) {
    Object.entries(screens).forEach(([k, el]) => { el.hidden = k !== name; });
    window.scrollTo(0, 0);
  }

  // --- Input ---------------------------------------------------------------

  $("#btn-locate").addEventListener("click", () => {
    const status = $("#loc-status");
    if (!navigator.geolocation) {
      status.textContent = "Location isn't available here. Pick a city below.";
      return;
    }
    status.textContent = "Finding you…";
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        coords = { lat: +pos.coords.latitude.toFixed(5), lon: +pos.coords.longitude.toFixed(5) };
        $("#city").value = "";
        status.textContent = `Using your location (${coords.lat.toFixed(3)}, ${coords.lon.toFixed(3)}, ±${Math.round(pos.coords.accuracy)} m).`;
      },
      (err) => {
        coords = null;
        status.textContent = (err.code === err.PERMISSION_DENIED ? "Location permission denied." : "Couldn't get your location.")
          + " Pick a city below.";
      },
      { enableHighAccuracy: false, timeout: 15000, maximumAge: 300000 },
    );
  });

  $("#city").addEventListener("change", (e) => {
    if (e.target.value) {
      coords = null;
      $("#loc-status").textContent = "";
    }
  });

  // --- Clip ----------------------------------------------------------------

  const MAX_MB = 100;
  const WARN_S = 30;
  const MAX_S = 60;
  let clip = null;  // {file, duration: Promise<seconds | null>}

  // Duration from a hidden <video>'s metadata; null if unknown (unreadable, or Infinity as some WebM recordings report).
  function clipDuration(file) {
    return new Promise((resolve) => {
      const v = $("#video-probe");
      const url = URL.createObjectURL(file);
      const done = (d) => {
        clearTimeout(timer);
        v.onloadedmetadata = v.onerror = null;
        v.removeAttribute("src");
        v.load();
        URL.revokeObjectURL(url);
        resolve(Number.isFinite(d) && d > 0 ? d : null);
      };
      const timer = setTimeout(() => done(null), 10000);
      v.onloadedmetadata = () => done(v.duration);
      v.onerror = () => done(null);
      v.src = url;
    });
  }

  // -> error message (block) or null; sets the > 30 s warning.
  function checkClip(file, duration) {
    const warn = $("#video-warn");
    warn.hidden = true;
    if (file.size > MAX_MB * 1e6) return `Clip is ${(file.size / 1e6).toFixed(0)} MB; the limit is ${MAX_MB} MB. Record about 10 s.`;
    if (duration != null && duration > MAX_S) return `Clip is ${Math.round(duration)} s; the limit is ${MAX_S} s. Trim it to ~10 s.`;
    if (duration != null && duration > WARN_S) {
      warn.textContent = `Clip is ${Math.round(duration)} s. Trim to ~10 s for a faster result.`;
      warn.hidden = false;
    }
    return null;
  }

  document.querySelectorAll(".js-video").forEach((input) => input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return;
    document.querySelectorAll(".js-video").forEach((other) => { if (other !== input) other.value = ""; });
    formError("");
    const label = `${file.name} (${(file.size / 1e6).toFixed(1)} MB`;
    $("#video-label").textContent = `${label})`;
    clip = { file, duration: clipDuration(file) };
    const mine = clip;
    const duration = await mine.duration;
    if (clip !== mine) return;  // another clip was picked meanwhile
    $("#video-label").textContent = `${label}${duration != null ? `, ${Math.round(duration)} s` : ""})`;
    formError(checkClip(file, duration));
  }));

  function pickedLocation() {
    if (coords) return { ...coords, gps: true };
    const v = $("#city").value;
    if (!v) return null;
    const [lat, lon] = v.split(",").map(Number);
    return { lat, lon, city: $("#city").selectedOptions[0].textContent };
  }

  function formError(msg) {
    const el = $("#form-error");
    el.textContent = msg || "";
    el.hidden = !msg;
  }

  $("#job-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    formError("");
    const loc = pickedLocation();
    const file = clip && clip.file;
    if (!loc) return formError("Allow location or pick a city.");
    if (!file) return formError("Record or choose a video.");
    const type = window.api.contentType(file);
    if (!type) return formError("Unsupported video type. Use MP4, WebM or MOV.");
    const blocked = checkClip(file, await clip.duration);
    if (blocked) return formError(blocked);
    const profile = document.querySelector('input[name="profile"]:checked').value;
    lastJob = { lat: loc.lat, lon: loc.lon, profile, file, type, picked: { city: loc.city, gps: !!loc.gps } };
    run(lastJob);
  });

  // --- Progress ------------------------------------------------------------

  function step(name, state) {
    const li = document.querySelector(`.steps [data-step="${name}"]`);
    li.classList.remove("active", "done", "failed");
    if (state) li.classList.add(state);
  }

  function resetSteps() {
    ["create", "upload", "analyse", "done"].forEach((s) => step(s, null));
    $("#upload-bar").value = 0;
    $("#upload-pct").textContent = "";
    $("#progress-error").hidden = true;
  }

  function friendly(err) {
    if (err instanceof TypeError) return "Network error. Check your connection and try again.";
    return err.message || "Something went wrong.";
  }

  async function run(job) {
    show("progress");
    resetSteps();
    let current = "create";
    try {
      step("create", "active");
      const { job_id, upload_url } = await window.api.createJob({
        lat: job.lat, lon: job.lon, profile: job.profile, content_type: job.type,
      });
      step("create", "done");

      current = "upload";
      step("upload", "active");
      await window.api.upload(upload_url, job.file, job.type, (f) => {
        $("#upload-bar").value = Math.round(f * 100);
        $("#upload-pct").textContent = `${Math.round(f * 100)}%`;
      });
      step("upload", "done");

      current = "analyse";
      step("analyse", "active");
      const body = await window.api.poll(job_id, () => {});
      step("analyse", "done");
      step("done", "done");
      showResult(body, job.picked);
    } catch (err) {
      step(current, "failed");
      $("#progress-error-msg").textContent = friendly(err);
      $("#progress-error").hidden = false;
    }
  }

  $("#btn-retry").addEventListener("click", () => { if (lastJob) run(lastJob); });

  document.querySelectorAll(".js-restart").forEach((b) => b.addEventListener("click", () => {
    if (new URLSearchParams(window.location.search).has("mock")) {
      window.location.href = window.location.pathname;
      return;
    }
    show("input");
  }));

  // --- Result --------------------------------------------------------------

  function showResult(body, picked) {
    window.renderResult($("#result"), body, picked);
    show("result");
  }

  async function loadMock() {
    show("progress");
    resetSteps();
    step("analyse", "active");
    try {
      const res = await fetch("mock/result.json", { cache: "no-store" });
      if (!res.ok) throw new Error(`mock/result.json: ${res.status}`);
      showResult(await res.json());
    } catch (err) {
      step("analyse", "failed");
      $("#progress-error-msg").textContent = `Couldn't load the mock result (${friendly(err)}).`;
      $("#progress-error").hidden = false;
      $("#btn-retry").onclick = loadMock;
    }
  }

  if (new URLSearchParams(window.location.search).get("mock") === "1") loadMock();
})();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
}
