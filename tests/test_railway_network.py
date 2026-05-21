"""Tests for the shared railway network — maldon-ng-drive perspective."""

import json
import pytest
from pathlib import Path

from tools.railway_network import (
    load_network,
    get_trackbed_for_map,
    get_stations_for_map,
    get_all_stations,
    build_railway_objects,
)
from tools.constants import WORLD_HALF, CENTER_LAT, CENTER_LON


_ELEV_FN = lambda wx, wy: 10.0   # flat Maldon estuary mock


class TestNetworkJson:
    def test_loads(self):
        net = load_network()
        assert isinstance(net, dict)

    def test_has_lines(self):
        net = load_network()
        assert "lines" in net
        assert len(net["lines"]) >= 1

    def test_maldon_branch_present(self):
        net = load_network()
        ids = {l["id"] for l in net["lines"]}
        assert "maldon_branch" in ids

    def test_dunmow_branch_present(self):
        net = load_network()
        ids = {l["id"] for l in net["lines"]}
        assert "dunmow_branch" in ids

    def test_all_nodes_four_elements(self):
        net = load_network()
        for line in net["lines"]:
            for node in line["nodes"]:
                assert len(node) == 4, f"Node {node} must be [lat, lon, z, width]"

    def test_node_latlon_plausible(self):
        net = load_network()
        for line in net["lines"]:
            for node in line["nodes"]:
                lat, lon = node[0], node[1]
                assert 51.5 < lat < 52.2, f"Lat {lat} outside Essex"
                assert 0.3 < lon < 0.8, f"Lon {lon} outside Essex"

    def test_node_elevation_plausible(self):
        net = load_network()
        for line in net["lines"]:
            for node in line["nodes"]:
                z = node[2]
                assert 0.0 < z < 120.0, f"z={z} outside 0–120 m"

    def test_station_data(self):
        net = load_network()
        for line in net["lines"]:
            for st in line.get("stations", []):
                assert "id" in st
                assert "lat" in st and "lon" in st
                assert "z_asl" in st


class TestTrackbedForMap:
    def test_returns_list(self):
        nodes = get_trackbed_for_map()
        assert isinstance(nodes, list)

    def test_maldon_has_trackbed_nodes(self):
        """Maldon East station is within this map's world extent."""
        nodes = get_trackbed_for_map()
        assert len(nodes) >= 2, "At least 2 trackbed nodes should be in Maldon map"

    def test_nodes_within_world_plus_margin(self):
        margin = 100
        nodes = get_trackbed_for_map(margin_m=margin)
        lim = WORLD_HALF + margin
        for n in nodes:
            assert -lim <= n[0] <= lim, f"wx={n[0]} outside world+margin"
            assert -lim <= n[1] <= lim, f"wy={n[1]} outside world+margin"

    def test_nodes_have_four_elements(self):
        for n in get_trackbed_for_map():
            assert len(n) == 4

    def test_trackbed_elevation_plausible_for_maldon(self):
        """Maldon is low-lying; all trackbed nodes should be below 30 m."""
        for n in get_trackbed_for_map():
            assert n[2] < 35.0, f"z={n[2]} too high for Maldon coastal area"


class TestStationsForMap:
    def test_returns_list(self):
        st = get_stations_for_map()
        assert isinstance(st, list)

    def test_maldon_east_present(self):
        st = get_stations_for_map()
        ids = {s["id"] for s in st}
        assert "maldon_east" in ids, f"maldon_east not in {ids}"

    def test_stations_have_world_coords(self):
        for st in get_stations_for_map():
            assert "wx" in st and "wy" in st
            assert isinstance(st["wx"], float)
            assert isinstance(st["wy"], float)

    def test_stations_within_map(self):
        margin = 200
        lim = WORLD_HALF + margin
        for st in get_stations_for_map(margin_m=margin):
            assert -lim <= st["wx"] <= lim
            assert -lim <= st["wy"] <= lim

    def test_felsted_not_in_maldon_map(self):
        """Felsted is 29 km away — should not appear in Maldon map."""
        st = get_stations_for_map()
        ids = {s["id"] for s in st}
        assert "felsted" not in ids, "Felsted station should be outside Maldon map"


class TestAllStations:
    def test_all_stations_has_both_endpoints(self):
        sts = get_all_stations()
        ids = {s["id"] for s in sts}
        assert "maldon_east" in ids
        assert "felsted" in ids

    def test_no_duplicates(self):
        sts = get_all_stations()
        ids = [s["id"] for s in sts]
        assert len(ids) == len(set(ids))

    def test_witham_appears_once(self):
        """Witham is shared between both lines; should appear once."""
        sts = get_all_stations()
        witham_count = sum(1 for s in sts if s["id"] == "witham")
        assert witham_count == 1


class TestBuildRailwayObjects:
    def test_returns_list(self):
        objs = build_railway_objects(_ELEV_FN)
        assert isinstance(objs, list)

    def test_trackbed_present(self):
        objs = build_railway_objects(_ELEV_FN)
        decals = [o for o in objs if o.get("class") == "DecalRoad"]
        assert len(decals) >= 1

    def test_station_markers_present(self):
        objs = build_railway_objects(_ELEV_FN)
        markers = [o for o in objs if o.get("class") == "TSStatic"]
        assert len(markers) >= 1   # at least Maldon East

    def test_persistent_ids_unique(self):
        objs = build_railway_objects(_ELEV_FN)
        ids = [o["persistentId"] for o in objs if "persistentId" in o]
        assert len(ids) == len(set(ids))

    def test_trackbed_nodes_blended_elevation(self):
        """Nodes should be blended between historical z and elev_fn output."""
        objs = build_railway_objects(_ELEV_FN)
        decals = [o for o in objs if o.get("class") == "DecalRoad"]
        for d in decals:
            for n in d["nodes"]:
                z = n[2]
                # elev_fn returns 10.0, historical z is ~8–30 for Maldon
                # blend: 40% × 10 + 60% × historical
                assert 4.0 < z < 30.0, f"z={z} outside expected blend range"
