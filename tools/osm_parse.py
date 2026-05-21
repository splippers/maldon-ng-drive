"""
Parse the cached OSM JSON into typed Python structures ready for BeamNG conversion.

All ways are indexed by their OSM tags.  Node (lat, lon) pairs are resolved once.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tools.constants import OSM_CACHE, gps_to_world

log = logging.getLogger(__name__)

# Width lookup for highway types (metres)
HIGHWAY_WIDTH: dict[str, float] = {
    "motorway":      12.0, "trunk":          9.0, "primary":        7.5,
    "secondary":      6.5, "tertiary":        5.5, "unclassified":   5.0,
    "residential":    4.5, "living_street":   4.0, "service":        3.5,
    "track":          3.0, "bridleway":       2.5, "cycleway":       2.0,
    "footway":        2.0, "path":            1.5, "pedestrian":     3.0,
    "steps":          1.5,
}

HIGHWAY_MATERIAL: dict[str, str] = {
    "motorway":      "road_rubber_sticky", "trunk":      "road_rubber_sticky",
    "primary":       "road_rubber_sticky", "secondary":  "road_rubber_sticky",
    "tertiary":      "road_rubber_sticky", "unclassified":"road_rubber_sticky",
    "residential":   "road_rubber_sticky", "living_street":"road_rubber_sticky",
    "service":       "road_rubber_sticky", "track":      "dirt",
    "bridleway":     "dirt",               "cycleway":   "sidewalk",
    "footway":       "sidewalk",           "path":       "sidewalk",
    "pedestrian":    "sidewalk",           "steps":      "sidewalk",
}


@dataclass
class OsmRoad:
    osm_id:   int
    name:     str
    hw_type:  str
    width:    float
    material: str
    gps_nodes: list[tuple[float, float]]   # (lat, lon)


@dataclass
class OsmBuilding:
    osm_id:    int
    name:      str
    bld_type:  str
    gps_nodes: list[tuple[float, float]]   # polygon vertices (lat, lon)
    centroid:  tuple[float, float]         # (lat, lon)


@dataclass
class OsmWater:
    osm_id:    int
    name:      str
    ww_type:   str   # river, stream, drain, lake …
    gps_nodes: list[tuple[float, float]]
    is_area:   bool  # True for natural=water polygon


@dataclass
class OsmLanduse:
    osm_id:    int
    lu_type:   str   # forest, grass, farmland …
    gps_nodes: list[tuple[float, float]]
    centroid:  tuple[float, float]
    area_m2:   float


@dataclass
class OsmLeisure:
    osm_id:    int
    le_type:   str   # pitch, swimming_pool, playground …
    name:      str
    sport:     str
    gps_nodes: list[tuple[float, float]]
    centroid:  tuple[float, float]


@dataclass
class OsmData:
    roads:    list[OsmRoad]    = field(default_factory=list)
    buildings:list[OsmBuilding]= field(default_factory=list)
    water:    list[OsmWater]   = field(default_factory=list)
    landuse:  list[OsmLanduse] = field(default_factory=list)
    leisure:  list[OsmLeisure] = field(default_factory=list)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _centroid(gps: list[tuple[float,float]]) -> tuple[float, float]:
    if not gps:
        return (0.0, 0.0)
    return (sum(p[0] for p in gps) / len(gps),
            sum(p[1] for p in gps) / len(gps))


def _area_m2(gps: list[tuple[float,float]]) -> float:
    """Shoelace area in m² using equirectangular projection."""
    if len(gps) < 3:
        return 0.0
    pts = [gps_to_world(la, lo) for la, lo in gps]
    n   = len(pts)
    a   = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i+1) % n]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0


# ── Main parser ───────────────────────────────────────────────────────────────

def load(path: Path | str | None = None) -> OsmData:
    """
    Parse cached OSM JSON into OsmData.
    Returns an empty OsmData if cache is absent.
    """
    p = Path(path) if path else OSM_CACHE
    if not p.exists():
        log.warning("OSM cache not found at %s; returning empty data.", p)
        return OsmData()

    raw = json.loads(p.read_text())
    nodes_by_id: dict[int, tuple[float,float]] = {
        e["id"]: (e["lat"], e["lon"])
        for e in raw["elements"] if e["type"] == "node"
    }

    data = OsmData()
    seen_ids: set[int] = set()

    for el in raw["elements"]:
        if el["type"] != "way":
            continue
        wid = el["id"]
        if wid in seen_ids:
            continue
        seen_ids.add(wid)

        tags     = el.get("tags", {})
        gps      = [nodes_by_id[n] for n in el["nodes"] if n in nodes_by_id]
        if len(gps) < 2:
            continue

        if "highway" in tags:
            hw = tags["highway"]
            if hw in ("proposed", "construction", "platform"):
                continue
            data.roads.append(OsmRoad(
                osm_id   = wid,
                name     = tags.get("name", f"road_{wid}"),
                hw_type  = hw,
                width    = float(tags.get("width",  HIGHWAY_WIDTH.get(hw, 4.0))),
                material = HIGHWAY_MATERIAL.get(hw, "road_rubber_sticky"),
                gps_nodes= gps,
            ))

        elif "building" in tags:
            ctr = _centroid(gps)
            data.buildings.append(OsmBuilding(
                osm_id   = wid,
                name     = tags.get("name", ""),
                bld_type = tags.get("building", "yes"),
                gps_nodes= gps,
                centroid = ctr,
            ))

        elif "waterway" in tags or tags.get("natural") == "water":
            ww   = tags.get("waterway", "lake" if tags.get("natural")=="water" else "")
            data.water.append(OsmWater(
                osm_id   = wid,
                name     = tags.get("name", ""),
                ww_type  = ww,
                gps_nodes= gps,
                is_area  = (tags.get("natural") == "water"),
            ))

        elif "landuse" in tags or tags.get("natural") in ("wood", "scrub"):
            lu_type = tags.get("landuse") or tags.get("natural", "unknown")
            ctr = _centroid(gps)
            data.landuse.append(OsmLanduse(
                osm_id   = wid,
                lu_type  = lu_type,
                gps_nodes= gps,
                centroid = ctr,
                area_m2  = _area_m2(gps),
            ))

        elif "leisure" in tags:
            ctr = _centroid(gps)
            data.leisure.append(OsmLeisure(
                osm_id   = wid,
                le_type  = tags["leisure"],
                name     = tags.get("name", ""),
                sport    = tags.get("sport", ""),
                gps_nodes= gps,
                centroid = ctr,
            ))

    log.info("OSM: %d roads, %d buildings, %d water, %d landuse, %d leisure",
             len(data.roads), len(data.buildings),
             len(data.water), len(data.landuse), len(data.leisure))
    return data
