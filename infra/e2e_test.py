"""End-to-end test against the deployed stack (stdlib only).

    python infra/e2e_test.py path/to/test.mp4 path/to/test.webm [--api URL] [--stack hazebusters]

Per clip: POST /jobs (--place, default Ranchi; school) -> PUT the clip to the presigned URL -> poll GET /jobs/{id} every 2 s
until done/failed (60 s) -> print the JSON and check it against backend/contract.json -> download the Hindi
and English audio URLs and check each is a non-empty mp3 (--save-audio DIR keeps the files) -> check smoke: clips
named with --expect-smoke must come back smoke.present with a loading JPEG overlay (gradcam_url) and Bedrock advice ending with the
hedged smoke line; all others must
come back without smoke -> vehicle_count must be an int (the YOLO model ran); clips named with --expect-vehicles
must have at least 1; clips named with --expect-skip must be rejected by the quality gate (causes skipped,
caveat, still status done). Exit 1 on any failure.
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PLACES = {"Ranchi": (23.3441, 85.3096), "Hyderabad": (17.385, 78.4867), "Delhi": (28.6139, 77.209)}
TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
CONTRACT = Path(__file__).resolve().parents[1] / "backend" / "contract.json"
REGION = "ap-south-1"


def api_url(stack: str) -> str:
    out = subprocess.run(["aws", "cloudformation", "describe-stacks", "--stack-name", stack, "--region", REGION,
                          "--query", "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue", "--output", "text"],
                         check=True, capture_output=True, text=True)
    return out.stdout.strip()


def request(method: str, url: str, body: bytes | None = None, headers: dict | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def key_diff(expected, actual, path="") -> list[str]:
    """Keys of the contract missing from / extra in the result (nested dicts; null result values allowed)."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return []
    errs = [f"missing {path}{k}" for k in expected if k not in actual]
    errs += [f"extra {path}{k}" for k in actual if k not in expected]
    for k in expected.keys() & actual.keys():
        errs += key_diff(expected[k], actual[k], f"{path}{k}.")
    return errs


def check(result: dict) -> list[str]:
    errs = key_diff(json.loads(CONTRACT.read_text()), result)
    c = result["causes"]
    if not 8 <= c["frames_used"] + c["frames_rejected"] <= 12:
        errs.append(f"frames_used + frames_rejected = {c['frames_used'] + c['frames_rejected']} (expected ~10 sampled)")
    if not result["forecast_tomorrow"]:
        errs.append("forecast_tomorrow empty")
    if result["decision"]["for"] not in ("today", "tomorrow"):
        errs.append(f"decision.for = {result['decision']['for']!r}")
    air = result["air"]
    if not isinstance(air["aqi_cpcb"], (int, float)):
        errs.append(f"air.aqi_cpcb = {air['aqi_cpcb']!r}")
    src = air[air["used"]]
    if not src or not all(isinstance(src.get(k), (int, float)) for k in ("pm25", "aqi_cpcb", "band")):
        errs.append(f"air.{air['used']} lacks numbers: {src!r}")
    for lang in ("en", "hi"):
        if not (result["advice"].get(lang) or "").strip():
            errs.append(f"advice.{lang} empty")
    return errs


def is_mp3(data: bytes) -> bool:
    return data[:3] == b"ID3" or (len(data) > 1 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)


def check_audio(got: dict, clip: Path, save_dir: Path | None) -> list[str]:
    errs = []
    for lang in ("hi", "en"):
        key, url = got["result"]["audio"][f"{lang}_key"], (got.get("audio_urls") or {}).get(lang)
        if not key or not url:
            errs.append(f"no {lang} audio (key={key!r}, url={'set' if url else None})")
            continue
        t0 = time.monotonic()
        status, data = request("GET", url)
        print(f"[{clip.name}] {lang} audio GET -> {status}, {len(data)} bytes in {time.monotonic() - t0:.2f} s")
        if status != 200 or not data or not is_mp3(data):
            errs.append(f"{lang} audio not a non-empty mp3 (status {status}, {len(data)} bytes, head {data[:4]!r})")
        elif save_dir:
            out = save_dir / f"{clip.stem}_{got['job_id']}_{lang}.mp3"
            out.write_bytes(data)
            print(f"[{clip.name}] saved {out}")
    return errs


def check_vehicles(got: dict, clip: Path, expect: bool) -> list[str]:
    n = got["result"]["causes"]["vehicle_count"]
    print(f"[{clip.name}] vehicle_count {n}")
    if not isinstance(n, int):
        return [f"vehicle_count = {n!r} (vehicle model did not run)"]
    if expect and n < 1:
        return ["expected vehicles, got 0"]
    return []


def check_smoke(got: dict, clip: Path, expect: bool, save_dir: Path | None) -> list[str]:
    c = got["result"]["causes"]
    print(f"[{clip.name}] smoke {c['smoke']}, fire {c['fire']}, gradcam_key {c['gradcam_key']}")
    if any("Smoke check unavailable" in x for x in got["result"]["caveats"]):
        return ["smoke model did not run (caveat 'Smoke check unavailable')"]
    if c["smoke"]["present"] != expect:
        return [f"smoke.present = {c['smoke']['present']}, expected {expect}"]
    if not expect:
        return []
    a = got["result"]["advice"]
    errs = [] if a["source"] == "bedrock" else [f"advice.source = {a['source']!r}, expected 'bedrock'"]
    if not a["en"].endswith(SMOKE_LINE[0]) or not a["hi"].endswith(SMOKE_LINE[1]):
        errs.append("hedged smoke line missing from the end of the advice")
    url = got.get("gradcam_url")
    if not c["gradcam_key"] or not url:
        return errs + [f"no overlay (gradcam_key={c['gradcam_key']!r}, url={'set' if url else None})"]
    status, data = request("GET", url)
    print(f"[{clip.name}] overlay GET -> {status}, {len(data)} bytes")
    if status != 200 or data[:3] != b"\xff\xd8\xff":
        return errs + [f"overlay not a JPEG (status {status}, head {data[:4]!r})"]
    if save_dir:
        (save_dir / f"{clip.stem}_{got['job_id']}_overlay.jpg").write_bytes(data)
    return errs


SKIP_CAVEAT = "Couldn't check what's in this clip"
# Must match infra/processor/advice.py CAUSE_LINES[("smoke",)] (appended in code, not written by Bedrock).
SMOKE_LINE = ("Our camera check suggests possible smoke in your clip (experimental). If you can see a smoke source "
              "nearby, move away from it.",
              "कैमरा जाँच के अनुसार आपकी क्लिप में धुआँ हो सकता है (प्रयोगात्मक)। अगर पास में धुएँ का स्रोत दिखे, "
              "तो उससे दूर रहें।")


def check_skipped(got: dict, clip: Path) -> list[str]:
    """Quality gate rejected the clip: models skipped, caveat present, air/advice still there."""
    r = got["result"]
    c = r["causes"]
    print(f"[{clip.name}] quality: frames_used {c['frames_used']}, rejected {c['frames_rejected']}, "
          f"reasons {c['reject_reasons']}")
    errs = []
    if c["frames_used"] >= 3:
        errs.append(f"frames_used = {c['frames_used']}, expected < 3 (gate should reject)")
    if c["smoke"]["present"] or c["fire"]["present"] or c["vehicle_count"] is not None or c["gradcam_key"]:
        errs.append(f"causes not skipped: {c}")
    if not any(x.startswith(SKIP_CAVEAT) for x in r["caveats"]):
        errs.append("quality caveat missing")
    return errs


def run(api: str, clip: Path, save_dir: Path | None = None, expect_smoke: bool = False, place: str = "Ranchi",
        expect_vehicles: bool = False, expect_skip: bool = False) -> bool:
    ctype = TYPES[clip.suffix]
    t0 = time.monotonic()
    status, body = request("POST", f"{api}/jobs", json.dumps(
        {"lat": PLACES[place][0], "lon": PLACES[place][1], "profile": "school", "content_type": ctype}).encode(),
        {"Content-Type": "application/json"})
    if status != 201:
        print(f"[{clip.name}] POST /jobs -> {status}: {body.decode()}")
        return False
    job = json.loads(body)
    status, body = request("PUT", job["upload_url"], clip.read_bytes(), {"Content-Type": ctype})
    if status != 200:
        print(f"[{clip.name}] PUT upload -> {status}: {body.decode()[:500]}")
        return False
    t_up = time.monotonic()
    deadline = t_up + 60
    while time.monotonic() < deadline:
        time.sleep(2)
        status, body = request("GET", f"{api}/jobs/{job['job_id']}")
        got = json.loads(body)
        if status == 200 and got["status"] in ("done", "failed"):
            break
    else:
        print(f"[{clip.name}] timed out after 60 s; last: {body.decode()}")
        return False
    print(json.dumps(got, indent=2, ensure_ascii=False))
    print(f"[{clip.name}] job {job['job_id']}: {got['status']} — upload->result {time.monotonic() - t_up:.1f} s "
          f"(polled every 2 s), total {time.monotonic() - t0:.1f} s")
    if got["status"] != "done":
        print(f"[{clip.name}] FAILED: {got['error']}")
        return False
    errs = check(got["result"]) + check_audio(got, clip, save_dir)
    if expect_skip:
        errs += check_skipped(got, clip)
    else:
        errs += check_smoke(got, clip, expect_smoke, save_dir) + check_vehicles(got, clip, expect_vehicles)
        if any(x.startswith(SKIP_CAVEAT) for x in got["result"]["caveats"]):
            errs.append("quality gate skipped a clip that should pass")
    print(f"[{clip.name}] checks: " + ("OK" if not errs else "; ".join(errs)))
    return not errs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("clips", nargs="+", type=Path)
    p.add_argument("--api", help="HTTP API base URL (default: the stack's ApiUrl output)")
    p.add_argument("--stack", default="hazebusters")
    p.add_argument("--save-audio", type=Path, help="directory to save the downloaded mp3s and overlays")
    p.add_argument("--place", choices=list(PLACES), default="Ranchi", help="location sent with the job")
    p.add_argument("--expect-skip", action="append", default=[], metavar="CLIP_NAME",
                   help="file name of a clip the quality gate must reject (causes skipped + caveat; repeatable)")
    p.add_argument("--expect-vehicles", action="append", default=[], metavar="CLIP_NAME",
                   help="file name of a clip that must come back with vehicle_count >= 1 (repeatable)")
    p.add_argument("--expect-smoke", action="append", default=[], metavar="CLIP_NAME",
                   help="file name of a clip that must come back with smoke (repeatable)")
    args = p.parse_args()
    api = (args.api or api_url(args.stack)).rstrip("/")
    results = [run(api, clip, args.save_audio, clip.name in args.expect_smoke, args.place, clip.name in args.expect_vehicles,
                   clip.name in args.expect_skip)
               for clip in args.clips]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
