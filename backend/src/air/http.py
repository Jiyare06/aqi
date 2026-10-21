"""Minimal JSON GET (stdlib only, so the Lambda image stays small). Tests monkeypatch get_json."""

import json
import urllib.parse
import urllib.request

USER_AGENT = "hazebusters/0.1 (Environmental Hacks 2026)"


def get_json(url: str, params: dict | None = None, timeout: float = 10.0) -> dict:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, safe=',:;')}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)
