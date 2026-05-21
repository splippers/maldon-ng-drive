"""
Building footprints → BeamNG scene objects — v3.0.

Two representations are generated:

1. TSStatic markers — one per named building at its centroid.
   shapeName is left blank (fill in via World Editor with an appropriate mesh).

2. DecalRoad outlines — a closed loop around each building footprint.
   Uses the 'concrete' ground material so the outline is faintly visible
   on the terrain even without 3D meshes.

Key school buildings mapped from OSM + campus-map overlay:
  Felsted School Chapel (building=chapel)
  Felsted School Sports Centre
  Music School
  Grignon Hall
  Lord Riche Hall
  Bourchiers (historic building)
"""

from __future__ import annotations

import logging
from typing import Any

from tools.constants import gps_to_world, WORLD_HALF
from tools.osm_parse import OsmBuilding, OsmData, _centroid

log = logging.getLogger(__name__)

# ── Known named school buildings ──────────────────────────────────────────────
# Fallback positions derived from campus-map PDF + satellite overlay (world m).
# Used when OSM has the building but its centroid is outside the map.
_NAMED_FALLBACKS: dict[str, tuple[float, float]] = {
    "Felsted School Chapel":        (-165.0, -175.0),
    "Felsted School Sports Centre": ( 190.0,  110.0),
    "Music School":                 (  60.0,   80.0),
    "Grignon Hall":                 ( -30.0,   20.0),
    "Lord Riche Hall":              (-100.0,   60.0),
    "Bourchiers":                   (  80.0,  -60.0),
    "Memorial Hall":                (-150.0,   40.0),
    "Rookwoods":                    (-200.0,  200.0),
    "Felsted School":               (   0.0,    0.0),
}


def _uid(name: str) -> str:
    import uuid
    NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(NS, f"felsted.bld.{name}"))


def _in_world(wx: float, wy: float, margin: float = 50.0) -> bool:
    return (-WORLD_HALF - margin <= wx <= WORLD_HALF + margin and
            -WORLD_HALF - margin <= wy <= WORLD_HALF + margin)


def _ts_static(name: str, wx: float, wy: float, wz: float,
               bld_type: str = "yes") -> dict[str, Any]:
    """TSStatic placeholder at the building centroid."""
    shape = (
        "art/shapes/buildings/chapel.dae"    if "chapel" in bld_type.lower() else
        "art/shapes/buildings/school.dae"    if "school" in bld_type.lower() else
        "art/shapes/buildings/building.dae"
    )
    return {
        "class":        "TSStatic",
        "name":         f"bld_{name.replace(' ','_')[:32]}",
        "persistentId": _uid(name),
        "position":     [round(wx,2), round(wy,2), round(wz,2)],
        "rotation":     [0, 0, 0, 1],
        "scale":        [1, 1, 1],
        "shapeName":    shape,
        "_note":        f"OSM building: {name!r} — set shapeName in editor",
    }


def _footprint_road(osm_id: int, gps_nodes: list, elevation_fn) -> dict | None:
    """Closed DecalRoad loop tracing the building footprint."""
    nodes = []
    for lat, lon in gps_nodes:
        wx, wy = gps_to_world(lat, lon)
        if not _in_world(wx, wy):
            return None
        wz = elevation_fn(wx, wy) + 0.05   # just above terrain
        nodes.append([round(wx,2), round(wy,2), round(wz,2), 0.4])

    if len(nodes) < 3:
        return None
    # Close the loop
    if nodes[0] != nodes[-1]:
        nodes.append(nodes[0])

    return {
        "class":    "DecalRoad",
        "name":     f"footprint_{osm_id}",
        "persistentId": _uid(f"fp{osm_id}"),
        "material": "road_white",
        "renderPriority": 5,
        "textureLength": 2,
        "drivability": 0,
        "nodes": nodes,
    }


def build_building_objects(osm_data: OsmData,
                           elevation_fn) -> tuple[list[dict], list[dict]]:
    """
    Return (markers, footprints) — two lists of BeamNG scene objects.

    Parameters
    ----------
    osm_data     : parsed OSM data
    elevation_fn : callable(wx, wy) → float  terrain elevation
    """
    markers    : list[dict] = []
    footprints : list[dict] = []
    seen_names : set[str]   = set()

    for bld in osm_data.buildings:
        lat, lon = bld.centroid
        wx, wy   = gps_to_world(lat, lon)

        # If OSM centroid is outside map, try the fallback
        if not _in_world(wx, wy):
            if bld.name in _NAMED_FALLBACKS:
                wx, wy = _NAMED_FALLBACKS[bld.name]
            else:
                continue

        wz = elevation_fn(wx, wy)

        # Marker
        label = bld.name or f"building_{bld.osm_id}"
        if label not in seen_names:
            markers.append(_ts_static(label, wx, wy, wz, bld.bld_type))
            seen_names.add(label)

        # Footprint outline
        fp = _footprint_road(bld.osm_id, bld.gps_nodes, elevation_fn)
        if fp:
            footprints.append(fp)

    # Ensure all named fallbacks have markers even if OSM is missing them
    for name, (wx, wy) in _NAMED_FALLBACKS.items():
        if name not in seen_names:
            wz = elevation_fn(wx, wy)
            markers.append(_ts_static(name, wx, wy, wz, "school"))
            seen_names.add(name)

    log.info("Buildings: %d markers, %d footprint outlines",
             len(markers), len(footprints))
    return markers, footprints
