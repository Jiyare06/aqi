// Small display helpers shared by the result screen.
(function () {
  const BANDS = ["Good/Satisfactory", "Moderate/Poor", "Very Poor/Severe"];  // shared/cpcb_aqi BANDS

  // CPCB categories by upper AQI bound (CLAUDE.md §5).
  const CATEGORIES = [[50, "Good"], [100, "Satisfactory"], [200, "Moderate"], [300, "Poor"], [400, "Very Poor"], [Infinity, "Severe"]];

  function category(aqi) {
    if (aqi == null) return null;
    return CATEGORIES.find(([upper]) => aqi <= upper)[1];
  }

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // Wall-clock parts straight from the ISO string, so times stay in the location's zone (the backend sends local offsets).
  function parts(iso) {
    const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/.exec(iso || "");
    return m ? { date: `${m[1]}-${m[2]}-${m[3]}`, h: +m[4], min: +m[5] } : null;
  }

  function clock(iso) {
    const p = parts(iso);
    if (!p) return "";
    const h12 = p.h % 12 === 0 ? 12 : p.h % 12;
    return `${h12}${p.min ? ":" + String(p.min).padStart(2, "0") : ""} ${p.h < 12 ? "AM" : "PM"}`;
  }

  function ago(iso) {
    const t = Date.parse(iso);
    if (Number.isNaN(t)) return "";
    const mins = Math.round((Date.now() - t) / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins} min ago`;
    const h = Math.round(mins / 60);
    if (h < 48) return `${h} h ago`;
    return `${Math.round(h / 24)} days ago`;
  }

  window.fmt = { BANDS, category, esc, parts, clock, ago };
})();
