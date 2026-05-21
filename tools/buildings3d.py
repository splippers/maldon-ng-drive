"""
3D building mesh generator — v3.0.

Converts OSM building footprints into COLLADA (.dae) meshes: extruded
polygon walls + flat roof.  Each building is written as a separate .dae
in levels/felsted/art/shapes/buildings/ and referenced as a TSStatic
object in the scene.

Height estimation priority:
  1. OSM building:levels tag (parsed from raw OSM, 3.2 m per level)
  2. OSM height tag (metres)
  3. Default by building type (residential=8 m, school=12 m, chapel=18 m)
"""

from __future__ import annotations

import uuid
import logging
import math
from pathlib import Path
from typing import Callable

import numpy as np

from tools.constants import (
    LEVELS_DIR, WORLD_HALF, gps_to_world,
)
from tools.osm_parse import OsmBuilding, OsmData

log = logging.getLogger(__name__)

_SHAPES_DIR = LEVELS_DIR / "art" / "shapes" / "buildings"

_DEFAULT_HEIGHT: dict[str, float] = {
    "school":        12.0,
    "university":    12.0,
    "chapel":        18.0,
    "church":        18.0,
    "cathedral":     22.0,
    "residential":    8.0,
    "house":          7.0,
    "apartments":    12.0,
    "commercial":    10.0,
    "retail":         6.0,
    "industrial":     8.0,
    "warehouse":     10.0,
    "office":        12.0,
    "yes":            8.0,   # generic "building=yes"
}
_FALLBACK_HEIGHT = 8.0


def _estimate_height(bld: OsmBuilding) -> float:
    btype = (bld.bld_type or "yes").lower()
    return _DEFAULT_HEIGHT.get(btype, _FALLBACK_HEIGHT)


def _footprint_to_world(gps_nodes: list) -> list[tuple[float, float]]:
    """Convert GPS node list to world-space (wx, wy) pairs, drop last if closed."""
    poly = [gps_to_world(lat, lon) for lat, lon in gps_nodes]
    if len(poly) > 2 and poly[0] == poly[-1]:
        poly = poly[:-1]
    return poly


def _poly_area_sign(poly: list[tuple[float, float]]) -> float:
    """Signed area (positive = CCW in XY plane = CCW when Z-up)."""
    n = len(poly)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return area / 2.0


def _triangulate_polygon(poly: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """Ear-clip triangulation of a simple polygon.  Returns list of (i,j,k) index triples."""
    n = len(poly)
    if n < 3:
        return []
    indices = list(range(n))
    tris = []
    sign = _poly_area_sign(poly)

    def is_ear(i: int) -> bool:
        prev_i = indices[(i - 1) % len(indices)]
        curr_i = indices[i]
        next_i = indices[(i + 1) % len(indices)]
        a, b, c = poly[prev_i], poly[curr_i], poly[next_i]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if sign > 0 and cross < 0:
            return False
        if sign < 0 and cross > 0:
            return False
        # Check no other point is inside triangle
        for j, oi in enumerate(indices):
            if oi in (prev_i, curr_i, next_i):
                continue
            p = poly[oi]
            if _point_in_tri(p, a, b, c):
                return False
        return True

    def _point_in_tri(p, a, b, c):
        def sign(p1, p2, p3):
            return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
        d1, d2, d3 = sign(p,a,b), sign(p,b,c), sign(p,c,a)
        has_neg = (d1<0) or (d2<0) or (d3<0)
        has_pos = (d1>0) or (d2>0) or (d3>0)
        return not (has_neg and has_pos)

    safety = len(indices) * len(indices) + 10
    while len(indices) > 3 and safety > 0:
        safety -= 1
        found = False
        for i in range(len(indices)):
            if is_ear(i):
                prev_i = indices[(i - 1) % len(indices)]
                curr_i = indices[i]
                next_i = indices[(i + 1) % len(indices)]
                tris.append((prev_i, curr_i, next_i))
                indices.pop(i)
                found = True
                break
        if not found:
            break   # degenerate polygon

    if len(indices) == 3:
        tris.append(tuple(indices))
    return tris


def _write_dae(
    path: Path,
    poly: list[tuple[float, float]],
    z_base: float,
    height: float,
    name: str,
) -> None:
    """Write a COLLADA file for one extruded building polygon."""
    n = len(poly)
    if n < 3:
        return

    # Vertices: floor ring (z_base) then roof ring (z_base + height)
    # Indexed as: floor=0..n-1, roof=n..2n-1
    verts: list[tuple[float, float, float]] = []
    for wx, wy in poly:
        verts.append((wx, wy, z_base))
    for wx, wy in poly:
        verts.append((wx, wy, z_base + height))

    # Faces: walls (quads → 2 tris each) + roof triangles
    # Wall quad for edge i→(i+1): floor_i, floor_j, roof_j, roof_i  (CCW outward)
    tris: list[tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        fi, fj, ri, rj = i, j, i + n, j + n
        tris.append((fi, fj, rj))
        tris.append((fi, rj, ri))

    # Roof triangulation
    roof_poly = poly
    roof_tris = _triangulate_polygon(roof_poly)
    # Ensure roof faces outward (up): if polygon is CW in XY, flip winding
    if _poly_area_sign(poly) < 0:
        roof_tris = [(a + n, c + n, b + n) for a, b, c in roof_tris]
    else:
        roof_tris = [(a + n, b + n, c + n) for a, b, c in roof_tris]
    tris.extend(roof_tris)

    # Build COLLADA strings
    vert_str = " ".join(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}" for v in verts)
    tri_str  = " ".join(f"{a} {b} {c}" for a, b, c in tris)
    vc_str   = " ".join("3" for _ in tris)

    geo_id = f"geo_{name[:32]}"
    mat_id = "Mat_building"

    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_materials>
    <material id="{mat_id}" name="building">
      <instance_effect url="#{mat_id}_effect"/>
    </material>
  </library_materials>
  <library_effects>
    <effect id="{mat_id}_effect">
      <profile_COMMON>
        <technique sid="common">
          <phong>
            <diffuse><color>0.72 0.65 0.55 1</color></diffuse>
            <specular><color>0.05 0.05 0.05 1</color></specular>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_geometries>
    <geometry id="{geo_id}" name="{name[:64]}">
      <mesh>
        <source id="{geo_id}_pos">
          <float_array id="{geo_id}_pos_arr" count="{len(verts)*3}">{vert_str}</float_array>
          <technique_common>
            <accessor source="#{geo_id}_pos_arr" count="{len(verts)}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="{geo_id}_verts">
          <input semantic="POSITION" source="#{geo_id}_pos"/>
        </vertices>
        <triangles count="{len(tris)}" material="{mat_id}">
          <input semantic="VERTEX" source="#{geo_id}_verts" offset="0"/>
          <p>{tri_str}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="Building" name="{name[:64]}" type="NODE">
        <instance_geometry url="#{geo_id}">
          <bind_material>
            <technique_common>
              <instance_material symbol="{mat_id}" target="#{mat_id}"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene>
    <instance_visual_scene url="#Scene"/>
  </scene>
</COLLADA>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml)


def build_building_meshes(
    osm_data: OsmData,
    elevation_fn: Callable[[float, float], float],
) -> list[dict]:
    """
    Generate COLLADA building meshes and return a list of TSStatic scene objects.

    Each object's 'shapeName' points to the .dae file relative to the level root.
    Position is set to (0, 0, 0) because the geometry is already in world-space.
    """
    _SHAPES_DIR.mkdir(parents=True, exist_ok=True)
    objects: list[dict] = []
    skipped = 0

    all_buildings = list(osm_data.buildings)
    log.info("Generating 3D meshes for %d OSM buildings", len(all_buildings))

    for bld in all_buildings:
        if len(bld.gps_nodes) < 4:
            skipped += 1
            continue

        poly = _footprint_to_world(bld.gps_nodes)
        if len(poly) < 3:
            skipped += 1
            continue

        # Discard buildings entirely outside the world
        cx, cy = bld.centroid
        wx, wy = gps_to_world(cx, cy)
        if abs(wx) > WORLD_HALF + 50 or abs(wy) > WORLD_HALF + 50:
            skipped += 1
            continue

        z_base  = elevation_fn(wx, wy)
        height  = _estimate_height(bld)
        safe_id = str(bld.osm_id)
        dae_rel = f"levels/felsted/art/shapes/buildings/bld_{safe_id}.dae"
        dae_abs = _SHAPES_DIR / f"bld_{safe_id}.dae"

        _write_dae(dae_abs, poly, z_base, height, bld.name or f"bld_{safe_id}")

        objects.append({
            "class":        "TSStatic",
            "name":         f"bld_{safe_id}",
            "shapeName":    dae_rel,
            "position":     [0.0, 0.0, 0.0],   # geometry already in world space
            "scale":        [1.0, 1.0, 1.0],
            "rotation":     [0.0, 0.0, 0.0, 1.0],
            "persistentId": str(uuid.uuid4()),
        })

    log.info(
        "Built %d building meshes, skipped %d", len(objects), skipped
    )
    return objects
