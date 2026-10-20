"""
Street-level imagery fetcher (Mapillary) for the ng-drive network.

Downloads CC-BY-SA licensed street-level photos from Mapillary for the
map's bounding box and writes them into the existing photo-billboard
pipeline (data/photos/photo_manifest.json), so they appear in-game as
TSStatic billboards just like the historical photo spots.

Usage:
    export MAPILLARY_ACCESS_TOKEN=MLY|...
    python3 tools/fetch_streetview.py [--max 40] [--min-spacing 80] [--dry-run]

Or via the generator:
    python3 tools/generate.py --streetview

Notes:
  - Mapillary imagery is CC-BY-SA 4.0: attribution is written to
    data/photos/mapillary/ATTRIBUTION.txt and embedded in each manifest
    entry. Do not strip it.
  - Without a token the module is a no-op (generate.py skips it).
  - Images are cached on disk; re-runs only fetch new ones.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Iterable

import requests

from tools.constants import DATA_DIR, BBOX

log = logging.getLogger(__name__)

MAPILLARY_API = "https://graph.mapillary.com/images"
MAPILLARY_FIELDS = "id,thumb_1024_url,computed_geometry,captured_at,compass_angle"
ATTRIBUTION = "Street-level imagery (c) Mapillary contributors, CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)"


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _photo_dir() -> Path:
    d = DATA_DIR / "photos" / "mapillary"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path() -> Path:
    return DATA_DIR / "photos" / "photo_manifest.json"


def load_manifest() -> dict:
    p = _manifest_path()
    if not p.exists():
        return {"photos": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"photos": []}


def save_manifest(manifest: dict) -> None:
    p = _manifest_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_attribution() -> None:
    d = _photo_dir()
    (d / "ATTRIBUTION.txt").write_text(
        ATTRIBUTION + "\n" + "https://www.mapillary.com/app/terms\n",
        encoding="utf-8",
    )


def _tile_bboxes(bbox, tile_deg=0.02):
    """Split a (S,W,N,E) bbox into sub-bboxes no larger than tile_deg."""
    s, w, n, e = bbox
    tiles = []
    lat = s
    while lat < n:
        lat2 = min(lat + tile_deg, n)
        lon = w
        while lon < e:
            lon2 = min(lon + tile_deg, e)
            tiles.append((lat, lon, lat2, lon2))
            lon = lon2
        lat = lat2
    return tiles


def query_mapillary(bbox, token, limit=2000):
    """Return raw list of image dicts from the Mapillary graph API.

    Tiles the bbox into small sub-queries to avoid Mapillary's
    'reduce the amount of data' 500 errors on dense areas.
    """
    headers = {"Authorization": f"OAuth {token}"}
    out = []
    seen = set()
    for tb in _tile_bboxes(bbox):
        s, w, n, e = tb
        bbox_str = f"{w},{s},{e},{n}"
        params = {"bbox": bbox_str, "fields": MAPILLARY_FIELDS, "limit": 500}
        url = MAPILLARY_API
        while url and len(out) < limit:
            try:
                r = requests.get(url, headers=headers, params=params, timeout=60)
                r.raise_for_status()
            except requests.HTTPError:
                # skip this tile on failure
                break
            j = r.json()
            for d in j.get("data", []):
                if d.get("id") not in seen:
                    seen.add(d.get("id"))
                    out.append(d)
            paging = j.get("paging", {})
            url = paging.get("next")
            params = None
            if not url:
                break
            time.sleep(0.2)
        if len(out) >= limit:
            break
    return out[:limit]


def _sample(images, min_spacing_m, max_images):
    """Dedup by proximity and cap count; prefer diverse coverage."""
    sampled = []
    for img in images:
        geom = img.get("computed_geometry") or {}
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        if any(_haversine_m(lat, lon, p[1], p[2]) < min_spacing_m for p in sampled):
            continue
        sampled.append((img, lat, lon))
        if len(sampled) >= max_images:
            break
    return sampled


def _to_photo(img, lat, lon):
    mid = img["id"]
    url = img.get("thumb_1024_url")
    captured = img.get("captured_at")
    if isinstance(captured, (int, float)):
        import datetime
        try:
            date = datetime.datetime.utcfromtimestamp(captured).strftime("%Y-%m-%d")
        except (ValueError, OSError, OverflowError):
            date = ""
    elif isinstance(captured, str):
        date = captured[:10]
    else:
        date = ""
    heading = img.get("compass_angle")
    if heading is None:
        heading = 0
    return {
        "id": f"mapillary_{mid}",
        "lat": lat,
        "lon": lon,
        "heading": float(heading),
        "date": date,
        "caption": f"Street view ({date})" if date else "Street view",
        "file": f"mapillary_{mid}.jpg",
        "width_m": 6.0,
        "height_m": 4.0,
        "attribution": ATTRIBUTION,
        "_url": url,
        "_lat": lat,
        "_lon": lon,
    }


def download_images(sampled, photo_dir):
    """Download thumbnails; return list of photo dicts (without _url/_lat/_lon)."""
    results = []
    for img, lat, lon in sampled:
        photo = _to_photo(img, lat, lon)
        url = photo.pop("_url")
        photo.pop("_lat")
        photo.pop("_lon")
        if not url:
            continue
        dest = photo_dir / photo["file"]
        if not dest.exists():
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                dest.write_bytes(r.content)
            except requests.RequestException as e:
                log.warning("Failed to download %s: %s", url, e)
                continue
        results.append(photo)
    return results


def fetch_streetview(bbox, token, max_images=40, min_spacing_m=80, dry_run=False):
    """Fetch + cache street-level images; return list of photo dicts."""
    if not token:
        log.info("No Mapillary token provided — skipping street-view fetch.")
        return []
    log.info("Querying Mapillary for bbox %s (max %d)…", bbox, max_images)
    try:
        raw = query_mapillary(bbox, token)
    except requests.RequestException as e:
        log.warning("Mapillary query failed: %s", e)
        return []
    log.info("Mapillary returned %d images", len(raw))
    sampled = _sample(raw, min_spacing_m, max_images)
    log.info("Sampled %d images (spacing >= %d m)", len(sampled), min_spacing_m)
    if dry_run:
        return [_to_photo(i, la, lo) for i, la, lo in sampled]
    photo_dir = _photo_dir()
    photos = download_images(sampled, photo_dir)
    _write_attribution()
    log.info("Downloaded %d street-view images → %s", len(photos), photo_dir)
    return photos


def merge_manifest(new_photos):
    """Merge new photos into photo_manifest.json, dedup by id."""
    manifest = load_manifest()
    existing = {p.get("id") for p in manifest.get("photos", [])}
    added = 0
    for p in new_photos:
        if p["id"] in existing:
            continue
        # strip internal keys
        clean = {k: v for k, v in p.items() if not k.startswith("_")}
        manifest.setdefault("photos", []).append(clean)
        existing.add(p["id"])
        added += 1
    save_manifest(manifest)
    log.info("Merged %d new photos into manifest (%d total)", added, len(manifest["photos"]))
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Fetch Mapillary street-level imagery for an ng-drive map")
    ap.add_argument("--token", default=os.environ.get("MAPILLARY_ACCESS_TOKEN") or os.environ.get("MAPILLARY_TOKEN"),
                    help="Mapillary access token (or set MAPILLARY_ACCESS_TOKEN)")
    ap.add_argument("--max", type=int, default=40, help="Max images to fetch")
    ap.add_argument("--min-spacing", type=int, default=80, help="Min metres between photos")
    ap.add_argument("--dry-run", action="store_true", help="Query only, don't download")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
    if not args.token:
        print("ERROR: provide --token or set MAPILLARY_ACCESS_TOKEN", file=__import__("sys").stderr)
        raise SystemExit(2)

    photos = fetch_streetview(BBOX, args.token, max_images=args.max, min_spacing_m=args.min_spacing, dry_run=args.dry_run)
    if args.dry_run:
        print(f"Would fetch {len(photos)} images")
        return
    if photos:
        merge_manifest(photos)
        _write_attribution()
    print(f"Done: {len(photos)} images")


if __name__ == "__main__":
    main()
