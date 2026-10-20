#!/usr/bin/env python3
"""
Fetch OpenStreetMap data for a map's bounding box.

Downloads every feature the generator cares about (roads, buildings,
water, landuse, leisure, power, rail, barriers) for the configured
BBOX and writes a single Overpass JSON document to OSM_CACHE — the
same ``out body`` format tools/osm_parse.py already reads.

Usage
─────
    python3 tools/fetch_osm.py                 # write data/<map>_osm.json
    python3 tools/fetch_osm.py --force         # re-fetch if cache exists
    python3 tools/fetch_osm.py --out custom.json

The Overpass endpoint can be overridden with OVERPASS_URL; the public
mirror listed first is known-good when the main endpoint is throttled.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.constants import BBOX, MAP_NAME, OSM_CACHE

log = logging.getLogger("fetch_osm")

MIRRORS = [
    os.environ.get("OVERPASS_URL", ""),
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Feature classes the generator consumes.  Kept deliberately broad so a
# fetched cache never needs re-downloading when new layers are added.
_FILTERS = [
    "way[highway]",
    "way[building]",
    "way[waterway]",
    "way[natural]",
    "way[landuse]",
    "way[leisure]",
    "way[power]",
    "way[amenity]",
    "way[railway]",
    "way[barrier]",
    "way[bridge]",
    "way[tunnel]",
    "way[man_made]",
    "node[amenity]",
    "node[place]",
]


def build_query(bbox=BBOX, timeout: int = 180) -> str:
    s, w, n, e = bbox
    box = f"{s},{w},{n},{e}"
    body = "".join(f"{f}({box});" for f in _FILTERS)
    # `out body` emits the ways; `>; out skel qt;` then recurses down to
    # emit every node they reference (geometry), matching the cache format
    # tools/osm_parse.py expects.
    return f"[out:json][timeout:{timeout}];({body});out body;>;out skel qt;"


def fetch(url: str, query: str, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url,
        data=urllib.parse.urlencode({"data": query}).encode(),
        headers={"User-Agent": f"{MAP_NAME}-ng-drive/1.0 (ng-drive map generator)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    data = json.loads(raw)
    if "elements" not in data:
        raise RuntimeError(f"unexpected Overpass response: {raw[:200]!r}")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if the cache already exists")
    ap.add_argument("--out", type=Path, default=OSM_CACHE)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.out.exists() and not args.force:
        log.info("OSM cache already present: %s (use --force to re-fetch)", args.out)
        return 0

    query = build_query()
    log.info("Fetching OSM for %s BBOX %s …", MAP_NAME, BBOX)

    last_error: Exception | None = None
    for url in MIRRORS:
        if not url:
            continue
        for attempt in range(1, 4):
            try:
                data = fetch(url, query)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    ValueError, RuntimeError) as exc:
                last_error = exc
                log.warning("mirror %s attempt %d failed: %s", url, attempt, exc)
                time.sleep(10 * attempt)
                continue
            log.info("Fetched %d elements from %s", len(data["elements"]), url)
            break
        else:
            continue
        break
    else:
        log.error("all Overpass mirrors failed: %s", last_error)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data))
    log.info("Wrote %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
