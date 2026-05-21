"""
Water features → BeamNG scene objects — v3.0.

Sources
-------
• OSM waterway=river/stream and natural=water polygons
• Hardcoded River Chelmer and Stebbing Brook positions derived from OSM trace

BeamNG objects used
-------------------
• WaterBlock  — for named rivers and large water areas
• DecalRoad   — for streams and drains (visual trace on terrain)
"""

from __future__ import annotations

import logging
from typing import Any

from tools.constants import gps_to_world, WORLD_HALF
from tools.osm_parse import OsmWater, OsmData

log = logging.getLogger(__name__)

# Hard-coded River Chelmer world positions (from OSM trace, ~6 nodes for the
# stretch that crosses the southern portion of the map).
# Z values from SRTM: river thalweg ~40–45 m in this area.
_CHELMER_NODES: list[list[float]] = [
    [-1024, -650, 40.5, 8.0],
    [ -800, -680, 40.5, 9.0],
    [ -580, -720, 40.0, 10.0],
    [ -350, -750, 40.0, 11.0],
    [ -100, -770, 40.5, 10.0],
    [  150, -760, 41.0, 9.0],
    [  400, -740, 41.5, 8.5],
    [  650, -700, 42.0, 8.0],
    [  900, -650, 42.5, 7.5],
    [ 1024, -620, 43.0, 7.0],
]

# Stebbing Brook runs through the western side
_STEBBING_BROOK_NODES: list[list[float]] = [
    [-320, -1024, 45.0, 4.0],
    [-310,  -850, 46.0, 4.0],
    [-290,  -700, 48.0, 4.5],
    [-275,  -580, 50.0, 4.0],
    [-262,  -450, 54.0, 3.5],
    [-258,  -300, 60.0, 3.0],
    [-255,  -200, 64.0, 3.0],
]


def _uid(name: str) -> str:
    import uuid
    NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(NS, f"felsted.water.{name}"))


def _in_world(wx: float, wy: float, margin: float = 50.0) -> bool:
    return (-WORLD_HALF - margin <= wx <= WORLD_HALF + margin and
            -WORLD_HALF - margin <= wy <= WORLD_HALF + margin)


def _water_decal_road(name: str, osm_id: int,
                      gps_nodes: list, elevation_fn,
                      width: float = 5.0) -> dict | None:
    """Stream / drain as a DecalRoad using a water-surface material."""
    nodes = []
    for lat, lon in gps_nodes:
        wx, wy = gps_to_world(lat, lon)
        if not _in_world(wx, wy):
            continue
        wz = elevation_fn(wx, wy) - 0.5   # sink slightly into terrain
        nodes.append([round(wx,2), round(wy,2), round(wz,2), width])
    if len(nodes) < 2:
        return None
    return {
        "class":          "DecalRoad",
        "name":           f"stream_{name}_{osm_id}",
        "persistentId":   _uid(f"stream_{osm_id}"),
        "material":       "water",
        "renderPriority": 20,
        "textureLength":  8,
        "drivability":    0,
        "nodes":          nodes,
    }


def _water_block(name: str, x: float, y: float, z: float,
                 sx: float = 200.0, sy: float = 50.0) -> dict[str, Any]:
    """WaterBlock for a river or lake area."""
    return {
        "class":        "WaterBlock",
        "name":         name,
        "persistentId": _uid(name),
        "position":     [round(x,1), round(y,1), round(z,1)],
        "rotation":     [1, 0, 0, 0],
        "scale":        [sx, sy, 4.0],
        "fullReflect":  False,
        "useOcclusionQuery": True,
        "baseColor":    [0.15, 0.35, 0.55, 0.85],
        "clarity":      0.5,
        "density":      0.6,
    }


def build_water_objects(osm_data: OsmData, elevation_fn) -> list[dict]:
    """Return a list of BeamNG water scene objects."""
    objects: list[dict] = []

    # ── Hardcoded main rivers ─────────────────────────────────────────────────
    # River Chelmer DecalRoad
    objects.append({
        "class":          "DecalRoad",
        "name":           "river_chelmer",
        "persistentId":   _uid("chelmer"),
        "material":       "water",
        "renderPriority": 20,
        "textureLength":  20,
        "drivability":    0,
        "nodes":          _CHELMER_NODES,
    })
    # WaterBlock sitting over the Chelmer trace
    objects.append(_water_block("waterblock_chelmer", -150, -720, 40.0,
                                sx=2200, sy=60))

    # Stebbing Brook
    objects.append({
        "class":          "DecalRoad",
        "name":           "stream_stebbing_brook",
        "persistentId":   _uid("stebbing_brook"),
        "material":       "water",
        "renderPriority": 20,
        "textureLength":  8,
        "drivability":    0,
        "nodes":          _STEBBING_BROOK_NODES,
    })

    # ── OSM streams and drains ────────────────────────────────────────────────
    for wf in osm_data.water:
        if wf.is_area:
            # Natural water polygon → WaterBlock at centroid
            from tools.osm_parse import _centroid, _area_m2
            clat, clon  = _centroid(wf.gps_nodes)
            cx, cy      = gps_to_world(clat, clon)
            if not _in_world(cx, cy):
                continue
            cz   = elevation_fn(cx, cy) - 0.3
            area = _area_m2(wf.gps_nodes)
            side = max(10.0, (area ** 0.5))
            objects.append(
                _water_block(f"pond_{wf.osm_id}", cx, cy, cz, side, side)
            )
        elif wf.ww_type in ("stream", "drain"):
            w  = 3.0 if wf.ww_type == "stream" else 1.5
            rd = _water_decal_road(wf.ww_type, wf.osm_id,
                                   wf.gps_nodes, elevation_fn, w)
            if rd:
                objects.append(rd)

    log.info("Water: %d objects generated", len(objects))
    return objects
