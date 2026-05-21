"""
Shared railway network loader — ng-drive network.

Reads data/railway_network.json (maintained in maldon-ng-drive, symlinked or
copied to each repo) and provides:

  - get_trackbed_for_map()  → [wx, wy, z, width] nodes within this map's world
  - get_stations_for_map()  → station dicts with world coords
  - get_all_stations()      → every station on the full network
  - build_railway_objects() → BeamNG scene objects for this map

The railway_network.json lives in maldon-ng-drive/data/ as the authoritative
source.  Each other repo either symlinks or copies it to their own data/.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

_NETWORK_FILENAME = "railway_network.json"


def _find_network_json() -> Path:
    """Search this repo's data/ first, then maldon-ng-drive's data/."""
    from tools.constants import DATA_DIR, ROOT
    local = DATA_DIR / _NETWORK_FILENAME
    if local.exists():
        return local
    # Fall back to sibling maldon-ng-drive repo
    sibling = ROOT.parent / "maldon-ng-drive" / "data" / _NETWORK_FILENAME
    if sibling.exists():
        log.info("Using railway_network.json from %s", sibling)
        return sibling
    raise FileNotFoundError(
        f"railway_network.json not found in {DATA_DIR} or {sibling}. "
        "Copy or symlink maldon-ng-drive/data/railway_network.json here."
    )


def load_network() -> dict:
    path = _find_network_json()
    return json.loads(path.read_text())


def get_trackbed_for_map(margin_m: float = 100.0) -> list[list[float]]:
    """
    Return all trackbed nodes (from all lines) that fall within this map's
    world extent, as [wx, wy, z, width] lists (world coordinates).

    Nodes are sorted by their cumulative distance along each line, then
    concatenated.  A small margin allows the trackbed to exit map edges cleanly.
    """
    from tools.constants import WORLD_HALF, gps_to_world

    network = load_network()
    result: list[list[float]] = []
    lim = WORLD_HALF + margin_m

    for line in network["lines"]:
        line_nodes = []
        for node in line["nodes"]:
            lat, lon, z, width = node
            wx, wy = gps_to_world(lat, lon)
            if -lim <= wx <= lim and -lim <= wy <= lim:
                line_nodes.append([wx, wy, z, width])
        if len(line_nodes) >= 2:
            result.extend(line_nodes)

    return result


def get_stations_for_map(margin_m: float = 200.0) -> list[dict]:
    """Return station dicts (with world coords added) within the map extent."""
    from tools.constants import WORLD_HALF, gps_to_world

    network = load_network()
    seen_ids: set[str] = set()
    result = []
    lim = WORLD_HALF + margin_m

    for line in network["lines"]:
        for st in line.get("stations", []):
            if st["id"] in seen_ids:
                continue
            wx, wy = gps_to_world(st["lat"], st["lon"])
            if -lim <= wx <= lim and -lim <= wy <= lim:
                st = dict(st)
                st["wx"] = wx
                st["wy"] = wy
                result.append(st)
                seen_ids.add(st["id"])

    return result


def get_all_stations() -> list[dict]:
    """All stations in the network (for documentation / UI)."""
    network = load_network()
    seen: set[str] = set()
    result = []
    for line in network["lines"]:
        for st in line.get("stations", []):
            if st["id"] not in seen:
                result.append(st)
                seen.add(st["id"])
    return result


def build_railway_objects(
    elevation_fn: Callable[[float, float], float],
) -> list[dict]:
    """
    Generate BeamNG scene objects for this map's section of railway:
      - DecalRoad  : trackbed (gravel/ballast)
      - TSStatic   : station site markers
    """
    objects: list[dict] = []

    # ── Trackbed ──────────────────────────────────────────────────────────────
    nodes = get_trackbed_for_map()
    if len(nodes) >= 2:
        # Blend hardcoded z with real terrain (60% hardcoded, 40% terrain)
        blended = []
        for n in nodes:
            wx, wy, z_hist, w = n
            z_terrain = elevation_fn(wx, wy)
            z = z_hist * 0.6 + z_terrain * 0.4
            blended.append([round(wx, 2), round(wy, 2), round(z, 2), w])

        objects.append({
            "class":        "DecalRoad",
            "name":         "railway_trackbed",
            "material":     "road_gravel_01",
            "overObjects":  False,
            "breakAngle":   3.0,
            "renderPriority": 10,
            "persistentId": str(uuid.uuid4()),
            "nodes":        blended,
        })
        log.info("Railway: %d trackbed nodes in this map", len(blended))

    # ── Station markers ───────────────────────────────────────────────────────
    for st in get_stations_for_map():
        wx, wy = st["wx"], st["wy"]
        z = elevation_fn(wx, wy)
        name_slug = st["id"].replace("-", "_")
        objects.append({
            "class":        "TSStatic",
            "name":         f"station_{name_slug}",
            "shapeName":    f"levels/{_map_name()}/art/shapes/station_{name_slug}.dae",
            "position":     [round(wx, 2), round(wy, 2), round(z, 2)],
            "scale":        [1.0, 1.0, 1.0],
            "rotation":     [0.0, 0.0, 0.0, 1.0],
            "persistentId": str(uuid.uuid4()),
            "_station_data": {
                "name":       st["name"],
                "opened":     st.get("opened"),
                "closed":     st.get("closed"),
                "era":        "1900–1952",
                "note":       st.get("note", ""),
            },
        })

    log.info(
        "Railway objects: %d trackbed + %d stations",
        1 if len(nodes) >= 2 else 0,
        len(get_stations_for_map()),
    )
    return objects


def _map_name() -> str:
    try:
        from tools.constants import MAP_NAME
        return MAP_NAME
    except Exception:
        return "map"
