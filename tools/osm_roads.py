"""
Road network → BeamNG DecalRoad node lists — v3.0.

Sources (tried in order):
  1. Cached OSM data  (data/felsted_osm.json)  — real geometry
  2. Static fallback  — hardcoded 10-road network from satellite imagery

Each road dict:
    {"name": str, "material": str, "width": float, "nodes": [[x,y,z], …]}
"""

from __future__ import annotations

import logging
from typing import Callable

from tools.constants import (
    BBOX, WORLD_HALF, gps_to_world, OSM_CACHE,
)
from tools.osm_parse import OsmRoad

log = logging.getLogger(__name__)


# ── Converter ─────────────────────────────────────────────────────────────────

def _road_to_dict(road: OsmRoad,
                  elevation_fn: Callable[[float, float], float]) -> dict:
    nodes = []
    for lat, lon in road.gps_nodes:
        wx, wy = gps_to_world(lat, lon)
        wz     = elevation_fn(wx, wy)
        nodes.append([round(wx, 2), round(wy, 2), round(wz, 2)])
    if len(nodes) < 2:
        return {}
    return {
        "name":     f"road_{road.osm_id}",
        "material": road.material,
        "width":    road.width,
        "nodes":    nodes,
    }


def _within_world(nodes: list) -> bool:
    margin = 100   # allow roads to extend slightly beyond terrain edge
    return all(
        -WORLD_HALF - margin <= n[0] <= WORLD_HALF + margin and
        -WORLD_HALF - margin <= n[1] <= WORLD_HALF + margin
        for n in nodes
    )


# ── Static fallback network ────────────────────────────────────────────────────
# (kept for --offline runs and as regression baseline for tests)

_STATIC_ROADS: list[dict] = [
    {
        "name": "road_stebbing_s",   "material": "road_rubber_sticky", "width": 5.5,
        "nodes": [[-260,-1024,62.0],[-260,-900,62.8],[-258,-780,63.8],
                  [-256,-660,65.0],[-254,-540,66.5],[-252,-420,68.2],
                  [-250,-300,70.0],[-248,-200,72.0]],
    },
    {
        "name": "road_stebbing_n",   "material": "road_rubber_sticky", "width": 5.5,
        "nodes": [[-248,-200,72.0],[-246,-80,73.5],[-244,50,74.8],
                  [-242,180,76.0],[-240,340,77.5],[-238,500,79.0],
                  [-235,660,80.5],[-232,820,81.2],[-228,1024,82.0]],
    },
    {
        "name": "road_entrance_drive","material": "road_rubber_sticky","width": 5.0,
        "nodes": [[-248,-200,72.0],[-210,-196,72.6],[-170,-188,73.4],
                  [-130,-175,74.0],[-90,-140,74.5],[-50,-100,75.0],
                  [-10,-55,75.4],[30,-5,75.8]],
    },
    {
        "name": "road_chapel_court", "material": "road_rubber_sticky","width": 4.0,
        "nodes": [[-170,-188,73.4],[-160,-220,73.2],[-140,-235,73.0],[-115,-228,73.2]],
    },
    {
        "name": "road_campus_loop",  "material": "road_rubber_sticky","width": 4.5,
        "nodes": [[30,-5,75.8],[120,0,76.0],[185,50,76.0],[200,140,75.8],
                  [195,240,75.4],[170,320,75.0],[110,370,74.6],[30,390,74.4],
                  [-60,370,74.6],[-140,320,75.0],[-185,230,75.4],[-195,130,75.8],
                  [-180,30,76.0],[-130,-20,76.0],[-70,-30,75.9],[30,-5,75.8]],
    },
    {
        "name": "road_carpark_access","material":"road_rubber_sticky","width": 5.5,
        "nodes": [[-248,-200,72.0],[-248,-290,72.0],[-252,-360,72.0]],
    },
    {
        "name": "road_carpark_row_a","material":"road_rubber_sticky","width": 6.0,
        "nodes": [[-252,-360,72.0],[-340,-360,72.0],[-340,-430,72.0],
                  [-252,-430,72.0],[-252,-360,72.0]],
    },
    {
        "name": "road_sports_access","material":"road_rubber_sticky","width": 4.0,
        "nodes": [[195,240,75.4],[270,310,74.8],[360,400,74.2],[440,480,73.6],[500,550,73.2]],
    },
    {
        "name": "road_braintree_e",  "material":"road_rubber_sticky","width": 5.5,
        "nodes": [[1024,-40,74.5],[860,-45,74.5],[680,-55,74.5],[500,-70,74.5],
                  [320,-90,74.5],[160,-120,74.5],[40,-155,74.0],
                  [-100,-175,73.5],[-170,-188,73.4]],
    },
    {
        "name": "road_service_rear","material":"road_rubber_sticky","width": 3.5,
        "nodes": [[30,-5,75.8],[40,-80,75.6],[20,-160,75.2],
                  [-20,-180,74.8],[-80,-200,74.5],[-130,-175,74.0]],
    },
]


# ── Public API ─────────────────────────────────────────────────────────────────

def build_roads(elevation_fn: Callable[[float, float], float] | None = None,
                online: bool = False) -> list[dict]:
    """
    Return road dicts for the level builder.

    Uses cached OSM data when available; falls back to static network.
    `elevation_fn(wx, wy)` sets each node's Z value from the terrain.
    """
    if elevation_fn is None:
        # Flat elevation — only used in offline/static mode
        elevation_fn = lambda wx, wy: 76.0

    if OSM_CACHE.exists():
        from tools.osm_parse import load as parse_osm
        osm = parse_osm()
        roads = []
        for r in osm.roads:
            d = _road_to_dict(r, elevation_fn)
            if d and _within_world(d["nodes"]):
                roads.append(d)
        if roads:
            log.info("Using OSM road network: %d roads", len(roads))
            return roads
        log.warning("OSM roads parsed to 0 valid ways; using static network.")

    log.info("Using static road network (%d roads).", len(_STATIC_ROADS))
    return _STATIC_ROADS
