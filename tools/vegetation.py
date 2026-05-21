"""
Vegetation / forest areas → BeamNG scene objects — v3.0.

Approach
--------
• OSM landuse=forest and natural=wood areas → Forest group objects
  (BeamNG places trees procedurally using a density mask)
• OSM leisure=pitch → flat ground paint using DecalRoad with grass material
• Leisure=swimming_pool → noted as TSStatic placeholders

BeamNG Forest objects require a ForestItemData prototype and a density bitmap;
this module generates placeholder Forest groups with correct world positions
so the editor can attach density maps.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from tools.constants import gps_to_world, WORLD_HALF
from tools.osm_parse import OsmData, OsmLanduse, OsmLeisure, _centroid, _area_m2

log = logging.getLogger(__name__)


def _uid(name: str) -> str:
    import uuid
    NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(NS, f"felsted.veg.{name}"))


def _in_world(wx: float, wy: float, margin: float = 100.0) -> bool:
    return (-WORLD_HALF - margin <= wx <= WORLD_HALF + margin and
            -WORLD_HALF - margin <= wy <= WORLD_HALF + margin)


def _forest_group(lu: OsmLanduse, elevation_fn) -> dict | None:
    """Return a ForestItemData + Forest pair for a woodland polygon."""
    clat, clon = lu.centroid
    cx, cy     = gps_to_world(clat, clon)
    if not _in_world(cx, cy):
        return None
    cz    = elevation_fn(cx, cy)
    side  = math.sqrt(lu.area_m2)
    tag   = f"forest_{lu.osm_id}"
    return {
        "class":        "SimGroup",
        "name":         tag,
        "persistentId": _uid(tag),
        "children": [
            {
                "class":        "ForestItemData",
                "name":         f"{tag}_item",
                "persistentId": _uid(f"{tag}_item"),
                "shapeFile":    "art/shapes/trees/defaulttree.dae",
                "colorsHaveAlpha": True,
                "maxScale":     8.0,
                "minScale":     4.0,
                "windScale":    0.3,
            },
            {
                "class":        "Forest",
                "name":         f"{tag}_inst",
                "persistentId": _uid(f"{tag}_inst"),
                "position":     [round(cx,1), round(cy,1), round(cz,1)],
                "scale":        [round(side,1), round(side,1), 1.0],
                "dataBlocks":   [f"{tag}_item"],
                "_area_m2":     round(lu.area_m2),
            },
        ],
    }


def _pitch_decal(le: OsmLeisure, elevation_fn) -> dict | None:
    """Sports pitch as a DecalRoad rectangle (shows line paint on ground)."""
    nodes = []
    for lat, lon in le.gps_nodes:
        wx, wy = gps_to_world(lat, lon)
        if not _in_world(wx, wy, 200):
            continue
        wz = elevation_fn(wx, wy) + 0.02
        nodes.append([round(wx,2), round(wy,2), round(wz,2), 0.3])
    if len(nodes) < 3:
        return None
    if nodes[0][:2] != nodes[-1][:2]:
        nodes.append(nodes[0])
    tag = f"pitch_{le.osm_id}"
    return {
        "class":          "DecalRoad",
        "name":           tag,
        "persistentId":   _uid(tag),
        "material":       "road_white",
        "renderPriority": 8,
        "textureLength":  5,
        "drivability":    1,
        "nodes":          nodes,
    }


def build_vegetation_objects(osm_data: OsmData, elevation_fn) -> list[dict]:
    """Return BeamNG scene objects for vegetation and leisure pitches."""
    objects: list[dict] = []

    # Forest / woodland (landuse=forest + natural=wood/scrub)
    for lu in osm_data.landuse:
        if lu.lu_type in ("forest", "wood", "scrub") and lu.area_m2 > 200:
            fg = _forest_group(lu, elevation_fn)
            if fg:
                objects.append(fg)

    for lu in osm_data.landuse:
        if lu.lu_type == "grass" and lu.area_m2 > 200:
            clat, clon = lu.centroid
            cx, cy = gps_to_world(clat, clon)
            # Grass areas are handled by terrain material; just log
            log.debug("Grass area %.0f m² at (%.0f,%.0f)", lu.area_m2, cx, cy)

    for nat in []:   # natural=wood handled via osm_data.landuse above
        pass

    # Sports pitches
    pitch_count = 0
    for le in osm_data.leisure:
        if le.le_type in ("pitch", "playing_field"):
            pd = _pitch_decal(le, elevation_fn)
            if pd:
                objects.append(pd)
                pitch_count += 1

    log.info("Vegetation: %d forest groups, %d pitch outlines",
             sum(1 for o in objects if o.get("class") == "SimGroup"),
             pitch_count)
    return objects
