"""
OSM tags → BeamNG game elements — v1.0.

OpenStreetMap already carries the road *semantics* a driving game needs:
carriageway direction, lane counts, dual-carriageway splits, medians,
barriers, bridges, tunnels, surfaces, speed limits, lighting, footways
and cycleways.  This module reads those tags and derives the matching
BeamNG scene elements automatically, so every map in the network gets
road furniture from the data instead of hand-placement.

Elements produced
─────────────────
  linear (DecalRoad-shaped dicts, appended to the Roads group)
    • central_reserve   — median strip between dual carriageways
    • guard_rail        — barrier=* ways + motorway/trunk medians
    • sidewalk          — footways alongside roads (sidewalk=* tags)
    • cycleway          — cycle lanes/tracks (cycleway=* tags)

  point (TSStatic placeholders in the GameElements group)
    • speed_limit       — maxspeed limits
    • street_lamp       — lit=yes ways, at fixed spacing
    • bridge / tunnel   — deck/portal markers

Materials
─────────
BeamNG resolves ``material`` by name; the ng-drive art pack ships the
median_*/guard_rail/sidewalk materials referenced here.  Until an art
pack is installed the elements still appear (grey) and carry their OSM
tags in ``_note`` for the World Editor.

Geometry note
─────────────
Dual carriageways are almost always mapped as *two* one-way ways a few
metres apart.  We pair them by proximity + opposite direction and lay
the median along the mid-line; explicitly tagged ``dual_carriageway=yes``
ways just get a median down their own centre-line.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tools.constants import gps_to_world
from tools.osm_parse import OsmRoad

log = logging.getLogger(__name__)

ElevFn = Callable[[float, float], float]

# ── Tuning tables ─────────────────────────────────────────────────────────────

# Classes that can be one half of a dual carriageway.
DUAL_CLASSES = {"motorway", "trunk", "primary", "secondary"}

LANE_WIDTH: dict[str, float] = {
    "motorway": 3.7, "trunk": 3.5, "primary": 3.3, "secondary": 3.2,
    "tertiary": 3.0, "unclassified": 3.0, "residential": 3.0,
    "living_street": 2.8, "service": 2.8,
}

# Median width (m) when the way carries no explicit median tag.
DEFAULT_MEDIAN: dict[str, float] = {
    "motorway": 3.5, "trunk": 3.0, "primary": 2.5, "secondary": 2.0,
}

MEDIAN_MATERIAL = {"motorway": "median_concrete", "trunk": "median_concrete",
                   "primary": "median_concrete"}
DEFAULT_MEDIAN_MATERIAL = "median_grass"

SIDEWALK_WIDTH = 2.0
CYCLEWAY_WIDTH = 1.8
SIDEWALK_MATERIAL = "sidewalk"
CYCLEWAY_MATERIAL = "cycleway"
GUARDRAIL_MATERIAL = "guard_rail"

LAMP_SPACING_M = 35.0
SPEED_SIGN_INTERVAL_M = 400.0

BARRIER_KINDS = {"guard_rail", "crash_barrier", "jersey_barrier",
                 "concrete_barrier", "barrier_board", "kerb"}

MAX_PAIR_GAP_M = 30.0     # carriageways further apart than this aren't a dual
MIN_PAIR_GAP_M = 1.0      # closer than this and they're the same carriageway
MIN_PAIR_COVER = 0.5      # ≥50 % of nodes must find an opposite partner
MAX_PAIR_MIDPOINT_M = 120.0   # cheap prefilter before precise distance


# ── Tag semantics ─────────────────────────────────────────────────────────────

@dataclass
class RoadSemantics:
    oneway:    int = 0            # 0 two-way, +1 forward, -1 reverse
    lanes:     int | None = None
    dual:      bool = False
    bridge:    bool = False
    tunnel:    bool = False
    surface:   str = ""
    maxspeed:  float | None = None
    roundabout: bool = False
    sidewalk:  str = ""           # no | both | left | right | yes | separate
    cycleway:  str = ""           # lane | track | shared | no | opposite_*
    lit:       bool = False
    barrier:   str = ""
    median:    float | None = None
    name:      str = ""
    hw:        str = ""


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).split()[0].rstrip("m").strip())
    except (ValueError, IndexError):
        return None


def _mph_to_kmh(value: str | None) -> float | None:
    if not value:
        return None
    v = str(value).strip().lower()
    if v in ("none", "signals", "variable"):
        return None
    f = _as_float(v)
    if f is None:
        return None
    if "mph" in v:
        return round(f * 1.609344)
    return round(f)


def road_semantics(tags: dict[str, str]) -> RoadSemantics:
    """Interpret the OSM tags of one highway way."""
    ow = tags.get("oneway", "").strip().lower()
    oneway = 1 if ow in ("yes", "true", "1") else (-1 if ow in ("-1", "reverse") else 0)
    roundabout = tags.get("junction", "").lower() in ("roundabout", "circular")
    if roundabout:
        oneway = 1

    lanes = None
    for key in ("lanes", "lanes:forward", "lanes:both_ways"):
        v = tags.get(key, "")
        if v.strip().isdigit():
            lanes = int(v)
            break

    median = None
    for key in ("median", "median:width", "central_reservation",
                "central_reservation:width"):
        median = _as_float(tags.get(key))
        if median:
            break

    barrier = tags.get("barrier", "").lower()
    return RoadSemantics(
        oneway     = oneway,
        lanes      = lanes,
        dual       = tags.get("dual_carriageway", "").lower() in ("yes", "true", "1"),
        bridge     = tags.get("bridge", "").lower() not in ("", "no"),
        tunnel     = tags.get("tunnel", "").lower() not in ("", "no"),
        surface    = tags.get("surface", "").lower(),
        maxspeed   = _mph_to_kmh(tags.get("maxspeed")),
        roundabout = roundabout,
        sidewalk   = tags.get("sidewalk", "").lower(),
        cycleway   = (tags.get("cycleway", "") or tags.get("cycleway:left", "")
                      or tags.get("cycleway:right", "")).lower(),
        lit        = tags.get("lit", "").lower() in ("yes", "true", "1"),
        barrier    = barrier if barrier in BARRIER_KINDS else "",
        median     = median,
        name       = tags.get("name", ""),
        hw         = tags.get("highway", ""),
    )


def effective_width(road: OsmRoad) -> float:
    """Best estimate of the *whole* carriageway width in metres."""
    sem = road_semantics(road.tags)
    explicit = _as_float(road.tags.get("width"))
    if explicit:
        return explicit
    if sem.lanes:
        return sem.lanes * LANE_WIDTH.get(sem.hw, 3.0)
    return road.width


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _world_pts(road: OsmRoad) -> list[tuple[float, float]]:
    return [gps_to_world(la, lo) for la, lo in road.gps_nodes]


def _dist_point_to_segment(px, py, ax, ay, bx, by) -> tuple[float, float, float]:
    """Return (distance, closest_x, closest_y) from P to segment AB."""
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-12:
        return math.hypot(px - ax, py - ay), ax, ay
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy), cx, cy


def _nearest_on_polyline(px: float, py: float,
                         poly: list[tuple[float, float]]) -> tuple[float, float, float]:
    best = (float("inf"), poly[0][0], poly[0][1])
    for (ax, ay), (bx, by) in zip(poly, poly[1:]):
        d, cx, cy = _dist_point_to_segment(px, py, ax, ay, bx, by)
        if d < best[0]:
            best = (d, cx, cy)
    return best


def _offset_polyline(poly: list[tuple[float, float]], dist: float
                     ) -> list[tuple[float, float]]:
    """Offset a polyline sideways by ``dist`` (positive = left of travel)."""
    n = len(poly)
    out: list[tuple[float, float]] = []
    for i in range(n):
        if i == 0:
            ax, ay = poly[0]
            bx, by = poly[1]
        elif i == n - 1:
            ax, ay = poly[-2]
            bx, by = poly[-1]
        else:
            ax, ay = poly[i - 1]
            bx, by = poly[i + 1]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        out.append((poly[i][0] + nx * dist, poly[i][1] + ny * dist))
    return out


def _dedupe(poly: list[tuple[float, float]], min_gap: float = 1.0
            ) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in poly:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_gap:
            out.append(p)
    return out


def _linear(name: str, material: str, width: float,
            nodes: list[list[float]], **extra) -> dict:
    return {"name": name, "material": material, "width": round(width, 2),
            "nodes": nodes, **extra}


def _road_nodes(poly: list[tuple[float, float]], elev_fn: ElevFn,
                bridge: bool = False) -> list[list[float]]:
    zs = [elev_fn(x, y) for x, y in poly]
    if bridge and len(zs) >= 2:
        # Level deck between the abutment heights.
        z0, z1 = zs[0], zs[-1]
        n = len(zs) - 1
        zs = [z0 + (z1 - z0) * (i / n) for i in range(n + 1)]
    return [[round(x, 2), round(y, 2), round(z, 2)]
            for (x, y), z in zip(poly, zs)]


def _marker(kind: str, name: str, x: float, y: float, z: float,
            yaw_deg: float = 0.0, **props) -> dict:
    half = math.radians(yaw_deg) / 2
    return {
        "class":       "TSStatic",
        "name":        name,
        "shapeName":   "",
        "position":    [round(x, 2), round(y, 2), round(z + 0.2, 2)],
        "rotation":    [0.0, 0.0, round(math.sin(half), 4), round(math.cos(half), 4)],
        "scale":       [1, 1, 1],
        "gameElement": kind,
        "_note":       f"{kind} — set shapeName in editor" + (
            f" ({props['note']})" if props.get("note") else ""),
        **{k: v for k, v in props.items() if k != "note"},
    }


# ── Central reservations ──────────────────────────────────────────────────────

def _median_material(hw: str) -> str:
    return MEDIAN_MATERIAL.get(hw, DEFAULT_MEDIAN_MATERIAL)


def _median_width(sem: RoadSemantics, hw: str) -> float:
    return sem.median or DEFAULT_MEDIAN.get(hw, 2.0)


def pair_carriageways(roads: list[OsmRoad]
                      ) -> list[tuple[OsmRoad, OsmRoad]]:
    """
    Pair opposing one-way ways that form a dual carriageway.

    Returns a greedy, distance-sorted list of (way_a, way_b) pairs where
    each way is used at most once.
    """
    cand: list[tuple[OsmRoad, RoadSemantics, list[tuple[float, float]]]] = []
    for r in roads:
        sem = road_semantics(r.tags)
        # Roundabout arcs are one-way but are not carriageway halves.
        if (sem.oneway and not sem.roundabout
                and r.hw_type in DUAL_CLASSES and len(r.gps_nodes) >= 2):
            cand.append((r, sem, _world_pts(r)))

    def _mid(poly):
        return poly[len(poly) // 2]

    def _net_dir(poly) -> tuple[float, float]:
        dx, dy = poly[-1][0] - poly[0][0], poly[-1][1] - poly[0][1]
        L = math.hypot(dx, dy)
        return (dx / L, dy / L) if L > 1e-6 else (0.0, 0.0)

    scored: list[tuple[float, int, int]] = []
    for i in range(len(cand)):
        ri, si, pi = cand[i]
        mi = _mid(pi)
        di = _net_dir(pi)
        for j in range(i + 1, len(cand)):
            rj, sj, pj = cand[j]
            if ri.hw_type != rj.hw_type:
                continue
            mj = _mid(pj)
            if math.hypot(mi[0] - mj[0], mi[1] - mj[1]) > MAX_PAIR_MIDPOINT_M:
                continue
            # Carriageway halves run in opposite directions.
            dj = _net_dir(pj)
            if di[0] * dj[0] + di[1] * dj[1] > 0.0:
                continue
            ds_all = [_nearest_on_polyline(x, y, pj)[0] for x, y in pi]
            ds_all += [_nearest_on_polyline(x, y, pi)[0] for x, y in pj]
            cover = sum(1 for d in ds_all if d < MAX_PAIR_GAP_M) / max(len(ds_all), 1)
            med = float(np.median(ds_all)) if ds_all else float("inf")
            if cover >= MIN_PAIR_COVER and MIN_PAIR_GAP_M <= med <= MAX_PAIR_GAP_M:
                scored.append((med, i, j))

    scored.sort(key=lambda t: t[0])
    used: set[int] = set()
    pairs: list[tuple[OsmRoad, OsmRoad]] = []
    for _med, i, j in scored:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        pairs.append((cand[i][0], cand[j][0]))
    return pairs


def _median_polyline(a: list[tuple[float, float]],
                     b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Mid-line between two roughly parallel carriageways."""
    ref, other = (a, b) if len(a) >= len(b) else (b, a)
    mids = []
    for x, y in ref:
        _, cx, cy = _nearest_on_polyline(x, y, other)
        mids.append(((x + cx) / 2.0, (y + cy) / 2.0))
    return _dedupe(mids, 2.0)


def build_game_elements(osm_roads: list[OsmRoad], elev_fn: ElevFn,
                        world_half: float | None = None
                        ) -> dict[str, list[dict]]:
    """
    Derive all game elements from parsed OSM roads.

    Returns ``{"linear": [...road dicts...], "objects": [...markers...]}``.
    """
    if world_half is None:
        from tools.constants import WORLD_HALF
        world_half = WORLD_HALF

    linear: list[dict] = []
    objects: list[dict] = []

    def _in_world(pts) -> bool:
        m = 120
        return all(-world_half - m <= x <= world_half + m and
                   -world_half - m <= y <= world_half + m for x, y in pts)

    # 1. Dual carriageways mapped as a single way.
    for r in osm_roads:
        sem = road_semantics(r.tags)
        if not sem.dual or r.hw_type not in DUAL_CLASSES:
            continue
        poly = _dedupe(_world_pts(r), 2.0)
        if len(poly) < 2 or not _in_world(poly):
            continue
        w = _median_width(sem, r.hw_type)
        linear.append(_linear(f"median_{r.osm_id}", _median_material(r.hw_type), w,
                              _road_nodes(poly, elev_fn),
                              gameElement="central_reserve",
                              osm_id=r.osm_id, source="dual_carriageway=yes"))

    # 2. Paired opposing one-way carriageways.
    for a, b in pair_carriageways(osm_roads):
        sem = road_semantics(a.tags)
        poly = _median_polyline(_world_pts(a), _world_pts(b))
        if len(poly) < 2 or not _in_world(poly):
            continue
        w = _median_width(sem, a.hw_type)
        linear.append(_linear(f"median_{a.osm_id}_{b.osm_id}",
                              _median_material(a.hw_type), w,
                              _road_nodes(poly, elev_fn),
                              gameElement="central_reserve",
                              osm_id=a.osm_id, source="paired oneway ways"))

    # 3. Barriers + sidewalks + cycleways drawn along each way.
    for r in osm_roads:
        if r.hw_type not in (DUAL_CLASSES | {"tertiary", "unclassified",
                                             "residential", "service",
                                             "living_street"}):
            continue
        sem = road_semantics(r.tags)
        poly = _dedupe(_world_pts(r), 2.0)
        if len(poly) < 2 or not _in_world(poly):
            continue
        half = effective_width(r) / 2.0

        if sem.barrier:
            # Draw the barrier on each side the way's tags imply.
            sides = _barrier_sides(r.tags)
            for side in sides:
                off = _offset_polyline(poly, side * (half + 0.3))
                linear.append(_linear(f"barrier_{r.osm_id}_{'L' if side > 0 else 'R'}",
                                      GUARDRAIL_MATERIAL, 0.6,
                                      _road_nodes(off, elev_fn),
                                      gameElement="guard_rail",
                                      barrier=sem.barrier, osm_id=r.osm_id))

        sw = sem.sidewalk
        if sw in ("both", "left", "right", "yes", "separate"):
            sides = {"both": (1, -1), "yes": (1, -1), "separate": (1, -1),
                     "left": (1,), "right": (-1,)}[sw]
            for side in sides:
                off = _offset_polyline(poly, side * (half + SIDEWALK_WIDTH / 2))
                linear.append(_linear(f"sidewalk_{r.osm_id}_{'L' if side > 0 else 'R'}",
                                      SIDEWALK_MATERIAL, SIDEWALK_WIDTH,
                                      _road_nodes(off, elev_fn),
                                      gameElement="sidewalk", osm_id=r.osm_id))

        cw = sem.cycleway
        if cw in ("lane", "track", "shared", "opposite_lane", "opposite_track",
                  "opposite", "yes"):
            off = _offset_polyline(poly, half + CYCLEWAY_WIDTH / 2 + 0.2)
            linear.append(_linear(f"cycleway_{r.osm_id}", CYCLEWAY_MATERIAL,
                                  CYCLEWAY_WIDTH, _road_nodes(off, elev_fn),
                                  gameElement="cycleway", osm_id=r.osm_id))

        # Point furniture along the way.
        curv = _cumulative(poly)
        if sem.maxspeed:
            for j in _sample_indices(curv, SPEED_SIGN_INTERVAL_M):
                x, y = poly[j]
                objects.append(_marker("speed_limit", f"speed_{r.osm_id}_{j}",
                                       x, y, elev_fn(x, y),
                                       yaw_deg=_yaw(poly, j, sem.oneway),
                                       limit_kmh=sem.maxspeed))
        if sem.lit:
            for j in _sample_indices(curv, LAMP_SPACING_M):
                side = _offset_polyline(poly, half + 0.5)[j]
                objects.append(_marker("street_lamp", f"lamp_{r.osm_id}_{j}",
                                       side[0], side[1], elev_fn(side[0], side[1]),
                                       yaw_deg=_yaw(poly, j, sem.oneway)))
        if sem.bridge:
            x, y = poly[len(poly) // 2]
            objects.append(_marker("bridge", f"bridge_{r.osm_id}", x, y,
                                   max(elev_fn(px, py) for px, py in poly),
                                   note=sem.name or r.hw_type))
        if sem.tunnel:
            x, y = poly[len(poly) // 2]
            objects.append(_marker("tunnel", f"tunnel_{r.osm_id}", x, y,
                                   elev_fn(x, y), note=sem.name or r.hw_type))

    log.info("Game elements: %d linear (%d medians), %d point objects",
             len(linear), sum(1 for e in linear if e.get("gameElement") == "central_reserve"),
             len(objects))
    return {"linear": linear, "objects": objects}


# ── Small geometry/noise helpers ──────────────────────────────────────────────

def _cumulative(poly: list[tuple[float, float]]) -> list[float]:
    out = [0.0]
    for (ax, ay), (bx, by) in zip(poly, poly[1:]):
        out.append(out[-1] + math.hypot(bx - ax, by - ay))
    return out


def _sample_indices(curv: list[float], spacing: float) -> list[int]:
    if len(curv) < 2 or spacing <= 0:
        return []
    total = curv[-1]
    if total <= 0:
        return []
    targets = np.arange(0.0, total + 1e-6, spacing)
    return sorted({int(np.argmin(np.abs(np.asarray(curv) - t))) for t in targets})


def _yaw(poly: list[tuple[float, float]], i: int, oneway: int) -> float:
    j = min(i + 1, len(poly) - 1)
    k = max(i - 1, 0)
    dx, dy = poly[j][0] - poly[k][0], poly[j][1] - poly[k][1]
    if oneway < 0:
        dx, dy = -dx, -dy
    return math.degrees(math.atan2(dy, dx))


def _barrier_sides(tags: dict[str, str]) -> tuple[int, ...]:
    """Which side(s) of the way a barrier sits on."""
    return (1, -1)
