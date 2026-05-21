#!/usr/bin/env python3
"""
ng-drive map generator — configurable for any map in the network.

Reads map identity + geography from map_config.py in the repo root.
All generation logic lives in tools/ and is shared across all ng-drive repos.

Usage
─────
    python3 tools/generate.py                     # offline (cached OSM + SRTM)
    python3 tools/generate.py --satellite         # download aerial imagery
    python3 tools/generate.py --online --satellite --zip
    python3 tools/generate.py --zoom 18           # higher-res satellite tiles

Pipeline steps
──────────────
    1. Elevation  : SRTM-30m cache → live API → synthetic procedural
    2. Satellite  : ESRI WorldImagery tiles (optional, --satellite flag)
    3. Terrain    : binary .ter + 16-bit PNG heightmap + RGB preview
    4. OSM        : parse roads, buildings, water, vegetation
    5. 3D meshes  : building COLLADA files
    6. Railway    : extract this map's section of the shared network trackbed
    7. Photos     : place historical photo billboards from photo_manifest.json
    8. Level JSON : assemble main.level.json scene graph
    9. Package    : <map>.zip mod archive (--zip)
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from tools.constants       import MAP_NAME, MAP_VERSION, LEVELS_DIR, TERRAIN_DIR
from tools.heightmap       import build_elevation, elev_at_world
from tools.terrain_file    import write_ter, write_heightmap_png, write_preview_png
from tools.osm_roads       import build_roads
from tools.osm_parse       import load as parse_osm
from tools.buildings       import build_building_objects
from tools.buildings3d     import build_building_meshes
from tools.water           import build_water_objects
from tools.vegetation      import build_vegetation_objects
from tools.railway_network import build_railway_objects
from tools.photo_spots     import build_photo_objects, write_billboard_dae
from tools.level_builder   import build_level

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

LEVEL_JSON = LEVELS_DIR / "main.level.json"
TER_FILE   = TERRAIN_DIR / f"{MAP_NAME}.ter"
HEIGHT_PNG = TERRAIN_DIR / f"{MAP_NAME}_height.png"
PREVIEW    = LEVELS_DIR  / "preview.png"
MOD_ZIP    = ROOT / f"{MAP_NAME}.zip"


def main(
    online:         bool = False,
    pack_zip:       bool = False,
    do_satellite:   bool = False,
    satellite_zoom: int  = 17,
) -> None:
    log.info("=== ng-drive map generator — %s v%s ===", MAP_NAME, MAP_VERSION)

    # 1. Elevation
    log.info("Step 1/9 – build elevation (SRTM-30m + fBm)")
    elev    = build_elevation(online=online)
    elev_fn = lambda wx, wy: elev_at_world(elev, wx, wy)

    # 2. Satellite texture
    sat_path = LEVELS_DIR / "art" / "terrain" / "satellite" / "satellite.png"
    has_sat  = sat_path.exists()
    if do_satellite and not has_sat:
        log.info("Step 2/9 – download satellite imagery (zoom=%d)", satellite_zoom)
        from tools.satellite import build_satellite_texture, write_satellite_material
        build_satellite_texture(zoom=satellite_zoom)
        write_satellite_material()
        has_sat = sat_path.exists()
    else:
        log.info("Step 2/9 – satellite: %s", "cached" if has_sat else "skipped (use --satellite)")

    # 3. Terrain files
    log.info("Step 3/9 – write terrain files")
    TERRAIN_DIR.mkdir(parents=True, exist_ok=True)
    write_ter(TER_FILE, elev, use_satellite=has_sat)
    write_heightmap_png(HEIGHT_PNG, elev)
    write_preview_png(PREVIEW, elev)

    # 4. OSM data
    log.info("Step 4/9 – parse OSM data")
    osm = parse_osm()

    # 5. Scene objects
    log.info("Step 5/9 – build scene objects")
    roads      = build_roads(elevation_fn=elev_fn, online=online)
    buildings  = build_building_objects(osm, elev_fn)
    water      = build_water_objects(osm, elev_fn)
    vegetation = build_vegetation_objects(osm, elev_fn)

    # 6. 3D building meshes
    log.info("Step 6/9 – generate 3D building meshes")
    bld_meshes = build_building_meshes(osm, elev_fn)

    # 7. Railway (shared network, filtered to this map's extent)
    log.info("Step 7/9 – extract railway network for this map")
    railway = build_railway_objects(elev_fn)

    # 8. Historical photo billboards
    log.info("Step 8/9 – place historical photo billboards")
    write_billboard_dae()
    photos = build_photo_objects(elev_fn)

    # 9. Level JSON
    log.info("Step 9/9 – write main.level.json")
    build_level(
        roads           = roads,
        buildings       = buildings,
        water           = water,
        vegetation      = vegetation,
        building_meshes = bld_meshes,
        railway         = railway,
        photos          = photos,
        out_path        = LEVEL_JSON,
    )

    log.info("Generation complete — %s v%s", MAP_NAME, MAP_VERSION)
    if pack_zip:
        _pack_zip()


def _pack_zip() -> None:
    log.info("Packaging → %s", MOD_ZIP)
    files = [f for f in (ROOT / "levels").rglob("*") if f.is_file()]
    with zipfile.ZipFile(MOD_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.relative_to(ROOT))
    log.info("Zip written (%.1f MB)", MOD_ZIP.stat().st_size / 1_048_576)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ng-drive map generator")
    ap.add_argument("--online",    action="store_true")
    ap.add_argument("--satellite", action="store_true")
    ap.add_argument("--zoom",      type=int, default=17)
    ap.add_argument("--zip",       action="store_true")
    args = ap.parse_args()
    main(online=args.online, pack_zip=args.zip,
         do_satellite=args.satellite, satellite_zoom=args.zoom)
