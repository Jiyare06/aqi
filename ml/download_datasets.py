"""Download the smoke/fire datasets (parquet) from Hugging Face and verify size + SHA-256.

    python -I download_datasets.py <DATA_DIR> <LOG_FILE>

- D-Fire (CC0 1.0; Venâncio et al. 2022) via the Hugging Face mirror `badsaarow/d-fire`
  (official OneDrive links refuse scripted downloads). 21,527 images = the published D-Fire total.
- Pyro-SDIS (Apache-2.0; Pyronear) from `pyronear/pyro-sdis`.
Each goes into its own directory under DATA_DIR. Stdlib only; resumable; idempotent.
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

DATASETS = {"badsaarow/d-fire": "dfire_hf", "pyronear/pyro-sdis": "pyro_sdis"}
UA = {"User-Agent": "hazebusters/0.1"}


def log(fh, msg: str) -> None:
    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    fh.flush()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def files(repo: str) -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main?recursive=true"
    tree = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
    out = [{"path": f["path"], "size": f["lfs"]["size"], "sha256": f["lfs"]["oid"]}
           for f in tree if f["type"] == "file" and f["path"].startswith("data/") and f["path"].endswith(".parquet")]
    return out + [{"path": "README.md", "size": None, "sha256": None}]


def download(repo: str, item: dict, dest: Path, fh) -> bool:
    target = dest / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if item["sha256"] and target.exists() and target.stat().st_size == item["size"] and sha256(target) == item["sha256"]:
        log(fh, f"SKIP verified {repo}/{item['path']}")
        return True
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{item['path']}"
    part = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 4):
        try:
            start = part.stat().st_size if part.exists() else 0
            headers = dict(UA, **({"Range": f"bytes={start}-"} if start else {}))
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r, part.open("ab" if start else "wb") as out:
                if start and r.status != 206:  # server ignored the range: restart
                    out.seek(0)
                    out.truncate()
                while chunk := r.read(1 << 22):
                    out.write(chunk)
            break
        except Exception as e:
            log(fh, f"RETRY {attempt} {repo}/{item['path']}: {type(e).__name__}: {e}")
            time.sleep(5 * attempt)
    else:
        log(fh, f"FAIL download {repo}/{item['path']}")
        return False
    if item["sha256"]:
        size, digest = part.stat().st_size, sha256(part)
        if size != item["size"] or digest != item["sha256"]:
            log(fh, f"FAIL verify {repo}/{item['path']}: size {size} vs {item['size']}, sha256 {digest[:12]} vs {item['sha256'][:12]}")
            return False
    part.rename(target)
    log(fh, f"OK {repo}/{item['path']} {target.stat().st_size} bytes" + (" sha256 verified" if item["sha256"] else ""))
    return True


def main() -> None:
    data_dir, log_file = Path(sys.argv[1]), Path(sys.argv[2])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with log_file.open("a") as fh:
        log(fh, "START")
        for repo, sub in DATASETS.items():
            dest = data_dir / sub
            dest.mkdir(parents=True, exist_ok=True)
            items = files(repo)
            log(fh, f"{repo}: {len(items)} files -> {dest}")
            (dest / "MANIFEST.json").write_text(json.dumps({"repo": repo, "files": items}, indent=2) + "\n")
            for item in items:
                if download(repo, item, dest, fh):
                    ok += 1
                else:
                    fail += 1
        log(fh, f"ALL_DONE ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
