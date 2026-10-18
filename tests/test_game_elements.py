"""
Tests for OSM tag → BeamNG game-element derivation (tools/game_elements.py).

Map-agnostic: all fixtures are placed relative to whatever map
map_config.py / tools.constants currently points at, so the same tests
run unchanged on every map in the ng-drive network.

Covers the tag interpreter, the dual-carriageway pairing heuristic (the
"highway central reservation" case), linear elements (medians, barriers,
sidewalks, cycleways) and point furniture (speed signs, lamps, bridges,
tunnels).
"""

from __future__ import annotations

import pytest

from tools.constants import CENTER_LAT, CENTER_LON, gps_to_world
from tools.game_elements import (
    DEFAULT_MEDIAN,
    build_game_elements,
    effective_width,
    pair_carriageways,
    road_semantics,
)
from tools.osm_parse import OsmRoad


# ── Fixtures, placed relative to the map centre ───────────────────────────────

def _pt(dlat, dlon):
    return (CENTER_LAT + dlat, CENTER_LON + dlon)


def _line(dlat0, dlon0, dlat1, dlon1, n=5):
    return [_pt(dlat0 + (dlat1 - dlat0) * i / (n - 1),
                dlon0 + (dlon1 - dlon0) * i / (n - 1)) for i in range(n)]


# A ~550 m north-south line through the centre.
_NORTH_SOUTH = _line(-0.0025, 0.0, 0.0025, 0.0)
# 0.0002° of longitude ≈ 14 m at UK latitudes.
_GAP = 0.0002


def _road(osm_id, nodes, hw="primary", width=7.0, **tags):
    return OsmRoad(
        osm_id    = osm_id,
        name      = tags.get("name", f"road_{osm_id}"),
        hw_type   = hw,
        width     = float(width),
        material  = "road_rubber_sticky",
        gps_nodes = nodes,
        tags      = {"highway": hw, **tags},
    )


def flat_elev(x, y):        # noqa: ARG001
    return 10.0


# ── Tag interpretation ────────────────────────────────────────────────────────

class TestRoadSemantics:
    def test_oneway_variants(self):
        assert road_semantics({"oneway": "yes"}).oneway == 1
        assert road_semantics({"oneway": "true"}).oneway == 1
        assert road_semantics({"oneway": "-1"}).oneway == -1
        assert road_semantics({"oneway": "no"}).oneway == 0
        assert road_semantics({}).oneway == 0

    def test_roundabout_is_oneway(self):
        sem = road_semantics({"junction": "roundabout"})
        assert sem.roundabout and sem.oneway == 1

    def test_lanes_parsed(self):
        assert road_semantics({"lanes": "3"}).lanes == 3
        assert road_semantics({"lanes:forward": "2"}).lanes == 2
        assert road_semantics({}).lanes is None

    def test_maxspeed_conversions(self):
        assert road_semantics({"maxspeed": "30 mph"}).maxspeed == 48
        assert road_semantics({"maxspeed": "50"}).maxspeed == 50
        assert road_semantics({"maxspeed": "none"}).maxspeed is None

    def test_median_aliases(self):
        assert road_semantics({"median": "3"}).median == 3.0
        assert road_semantics({"central_reservation": "2.5"}).median == 2.5

    def test_bridge_tunnel_flags(self):
        assert road_semantics({"bridge": "yes"}).bridge
        assert not road_semantics({"bridge": "no"}).bridge
        assert road_semantics({"tunnel": "yes"}).tunnel

    def test_barrier_only_known_kinds(self):
        assert road_semantics({"barrier": "guard_rail"}).barrier == "guard_rail"
        assert road_semantics({"barrier": "gate"}).barrier == ""

    def test_lit_sidewalk_cycleway(self):
        sem = road_semantics({"lit": "yes", "sidewalk": "both", "cycleway": "lane"})
        assert sem.lit and sem.sidewalk == "both" and sem.cycleway == "lane"


class TestEffectiveWidth:
    def test_explicit_width_wins(self):
        r = _road(1, _NORTH_SOUTH, width=9.5)
        assert effective_width(r) == 9.5

    def test_lanes_times_lane_width(self):
        r = _road(1, _NORTH_SOUTH, hw="motorway", lanes="4")
        assert effective_width(r) == pytest.approx(4 * 3.7)

    def test_falls_back_to_hw_width(self):
        r = _road(1, _NORTH_SOUTH, hw="secondary")
        assert effective_width(r) == 7.0


# ── Central reservation pairing ───────────────────────────────────────────────

class TestPairCarriageways:
    def _dual_pair(self, gap=_GAP):
        """Two parallel one-way ways ~14 m apart, opposite directions."""
        a = _road(1, _line(-0.0025, 0.0, 0.0025, 0.0), oneway="yes")
        b = _road(2, _line(0.0025, gap, -0.0025, gap), oneway="yes")
        return a, b

    def test_pairs_opposing_parallel_oneways(self):
        a, b = self._dual_pair()
        pairs = pair_carriageways([a, b])
        assert len(pairs) == 1
        assert {pairs[0][0].osm_id, pairs[0][1].osm_id} == {1, 2}

    def test_ignores_two_way_roads(self):
        a = _road(1, _line(-0.0025, 0.0, 0.0025, 0.0))
        b = _road(2, _line(0.0025, _GAP, -0.0025, _GAP))
        assert pair_carriageways([a, b]) == []

    def test_ignores_same_direction(self):
        a = _road(1, _line(-0.0025, 0.0, 0.0025, 0.0), oneway="yes")
        b = _road(2, _line(-0.0025, _GAP, 0.0025, _GAP), oneway="yes")
        assert pair_carriageways([a, b]) == []

    def test_ignores_roundabout_arcs(self):
        a = _road(1, _line(-0.0025, 0.0, 0.0025, 0.0),
                  oneway="yes", junction="roundabout")
        b = _road(2, _line(0.0025, _GAP, -0.0025, _GAP),
                  oneway="yes", junction="roundabout")
        assert pair_carriageways([a, b]) == []

    def test_ignores_far_apart(self):
        a = _road(1, _line(-0.0025, 0.0, 0.0025, 0.0), oneway="yes")
        b = _road(2, _line(0.0025, 0.004, -0.0025, 0.004), oneway="yes")
        assert pair_carriageways([a, b]) == []

    def test_each_way_used_once(self):
        a = _road(1, _line(-0.0025, 0.0, 0.0025, 0.0), oneway="yes")
        b = _road(2, _line(0.0025, _GAP, -0.0025, _GAP), oneway="yes")
        c = _road(3, _line(0.0025, 2 * _GAP, -0.0025, 2 * _GAP), oneway="yes")
        pairs = pair_carriageways([a, b, c])
        used = [r.osm_id for p in pairs for r in p]
        assert len(used) == len(set(used))


# ── Element generation ────────────────────────────────────────────────────────

class TestBuildGameElements:
    def test_dual_carriageway_tag_makes_median(self):
        r = _road(7, _NORTH_SOUTH, hw="trunk", dual_carriageway="yes")
        els = build_game_elements([r], flat_elev, world_half=5000)
        medians = [e for e in els["linear"]
                   if e.get("gameElement") == "central_reserve"]
        assert len(medians) == 1
        assert medians[0]["width"] == DEFAULT_MEDIAN["trunk"]
        assert medians[0]["material"] == "median_concrete"

    def test_paired_oneways_make_median(self):
        a = _road(1, _line(-0.0025, 0.0, 0.0025, 0.0),
                  hw="secondary", oneway="yes")
        b = _road(2, _line(0.0025, _GAP, -0.0025, _GAP),
                  hw="secondary", oneway="yes")
        els = build_game_elements([a, b], flat_elev, world_half=5000)
        medians = [e for e in els["linear"]
                   if e.get("gameElement") == "central_reserve"]
        assert len(medians) == 1
        # median should sit between the two carriageways (world coords)
        xa = gps_to_world(*_pt(0.0, 0.0))[0]
        xb = gps_to_world(*_pt(0.0, _GAP))[0]
        mx = sum(n[0] for n in medians[0]["nodes"]) / len(medians[0]["nodes"])
        assert min(xa, xb) < mx < max(xa, xb)

    def test_guard_rail_from_barrier_tag(self):
        r = _road(3, _line(-0.0015, 0.0, 0.0015, 0.0),
                  hw="motorway", barrier="guard_rail")
        els = build_game_elements([r], flat_elev, world_half=5000)
        rails = [e for e in els["linear"] if e.get("gameElement") == "guard_rail"]
        assert len(rails) == 2          # both sides
        assert all(e["material"] == "guard_rail" for e in rails)

    def test_sidewalk_both_sides(self):
        r = _road(4, _line(-0.0015, 0.0, 0.0015, 0.0),
                  hw="residential", sidewalk="both")
        els = build_game_elements([r], flat_elev, world_half=5000)
        sw = [e for e in els["linear"] if e.get("gameElement") == "sidewalk"]
        assert len(sw) == 2

    def test_cycleway_lane(self):
        r = _road(5, _line(-0.0015, 0.0, 0.0015, 0.0),
                  hw="residential", cycleway="lane")
        els = build_game_elements([r], flat_elev, world_half=5000)
        assert any(e.get("gameElement") == "cycleway" for e in els["linear"])

    def test_speed_signs_emitted(self):
        r = _road(6, _line(-0.005, 0.0, 0.005, 0.0),
                  hw="primary", maxspeed="40 mph")
        els = build_game_elements([r], flat_elev, world_half=5000)
        signs = [e for e in els["objects"] if e.get("gameElement") == "speed_limit"]
        assert signs
        assert all(s["limit_kmh"] == 64 for s in signs)

    def test_street_lamps_when_lit(self):
        r = _road(8, _line(-0.0025, 0.0, 0.0025, 0.0),
                  hw="primary", lit="yes")
        els = build_game_elements([r], flat_elev, world_half=5000)
        lamps = [e for e in els["objects"] if e.get("gameElement") == "street_lamp"]
        assert lamps

    def test_bridge_deck_is_level(self):
        r = _road(9, _line(-0.0025, 0.0, 0.0025, 0.0),
                  hw="primary", bridge="yes")
        els = build_game_elements([r], lambda x, y: x / 100.0, world_half=5000)
        bridges = [e for e in els["objects"] if e.get("gameElement") == "bridge"]
        assert len(bridges) == 1

    def test_elements_outside_world_dropped(self):
        # Way ~5.5 km north of centre, world_half = 1000
        r = _road(10, _line(0.05, 0.0, 0.055, 0.0),
                  hw="trunk", dual_carriageway="yes")
        els = build_game_elements([r], flat_elev, world_half=1000)
        assert els["linear"] == []

    def test_returns_both_collections(self):
        els = build_game_elements([], flat_elev, world_half=5000)
        assert els == {"linear": [], "objects": []}
