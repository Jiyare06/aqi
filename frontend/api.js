// API calls: POST /jobs -> PUT clip to the presigned URL -> poll GET /jobs/{id}.
(function () {
  const TYPES = { mp4: "video/mp4", webm: "video/webm", mov: "video/quicktime" };  // what POST /jobs accepts
  const POLL_MS = 2000;
  const POLL_TIMEOUT_MS = 90000;
  const MAX_POLL_ERRORS = 3;  // consecutive network errors tolerated while polling

  // file.type if accepted, else by extension; null = unsupported.
  function contentType(file) {
    const accepted = Object.values(TYPES);
    if (accepted.includes(file.type)) return file.type;
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    return TYPES[ext] || null;
  }

  function apiBase() {
    const base = (window.API_BASE || "").replace(/\/+$/, "");
    if (!base) throw new Error("API_BASE is not set in config.js.");
    return base;
  }

  async function json(res) {
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON error page */ }
    if (!res.ok) throw new Error((body && body.error) || `Server error (${res.status}).`);
    return body;
  }

  async function createJob({ lat, lon, profile, content_type }) {
    const res = await fetch(`${apiBase()}/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, profile, content_type }),
    });
    return json(res);  // {job_id, upload_url}
  }

  // XHR for upload progress; Content-Type must match the one the URL was signed with.
  function upload(url, file, type, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("PUT", url);
      xhr.setRequestHeader("Content-Type", type);
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
      xhr.onload = () => (xhr.status >= 200 && xhr.status < 300
        ? resolve()
        : reject(new Error(`Upload failed (${xhr.status}).`)));
      xhr.onerror = () => reject(new Error("Upload failed: network error."));
      xhr.ontimeout = () => reject(new Error("Upload timed out."));
      xhr.send(file);
    });
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // Resolves with the GET /jobs/{id} body once status == done; rejects on failed, timeout or repeated network errors.
  async function poll(jobId, onStatus) {
    const deadline = Date.now() + POLL_TIMEOUT_MS;
    let errors = 0;
    while (Date.now() < deadline) {
      let body;
      try {
        body = await json(await fetch(`${apiBase()}/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" }));
        errors = 0;
      } catch (e) {
        if (++errors >= MAX_POLL_ERRORS) throw e;
        await sleep(POLL_MS);
        continue;
      }
      onStatus(body.status);
      if (body.status === "done") return body;
      if (body.status === "failed") throw new Error(body.error ? `Analysis failed: ${body.error}` : "Analysis failed.");
      await sleep(POLL_MS);
    }
    throw new Error("Analysis took longer than 90 s. Try again.");
  }

  window.api = { contentType, createJob, upload, poll };
})();
