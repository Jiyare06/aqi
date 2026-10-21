"""Live check: `uv run python -m air Ranchi Hyderabad [--profile school]` prints the combined air JSON per place.

WAQI_TOKEN comes from the environment or the repo-root .env (never printed).
"""

import argparse
import json
import os
from pathlib import Path

from air.combine import PROFILE_WINDOWS, air_for

PLACES = {"Ranchi": (23.3441, 85.3096), "Hyderabad": (17.3850, 78.4867), "Delhi": (28.6139, 77.2090), "Patna": (25.5941, 85.1376)}


def _token() -> str | None:
    if os.environ.get("WAQI_TOKEN"):
        return os.environ["WAQI_TOKEN"]
    env = Path(__file__).resolve().parents[3] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("WAQI_TOKEN="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("places", nargs="*", default=["Ranchi", "Hyderabad"], choices=list(PLACES))
    parser.add_argument("--profile", default="general", choices=list(PROFILE_WINDOWS))
    args = parser.parse_args()
    token = _token()
    for place in args.places:
        lat, lon = PLACES[place]
        out = {"profile": args.profile, "location": {"lat": lat, "lon": lon, "place": place}}
        print(json.dumps(out | air_for(lat, lon, token, profile=args.profile), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
