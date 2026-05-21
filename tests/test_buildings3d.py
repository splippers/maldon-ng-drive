"""Tests for 3D building mesh generation — v3.0."""

import pytest
from pathlib import Path

from tools.buildings3d import (
    build_building_meshes,
    _footprint_to_world,
    _poly_area_sign,
    _triangulate_polygon,
    _estimate_height,
    _write_dae,
)
from tools.osm_parse import OsmBuilding, OsmData
from tools.constants import WORLD_HALF


_ELEV_FN = lambda wx, wy: 76.0


# ── Polygon helpers ───────────────────────────────────────────────────────────

class TestPolyHelpers:
    def test_area_sign_ccw(self):
        # CCW square
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert _poly_area_sign(poly) > 0

    def test_area_sign_cw(self):
        poly = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        assert _poly_area_sign(poly) < 0

    def test_area_sign_magnitude(self):
        # Shoelace formula: 4×3 rectangle → signed area = 12.0
        poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)]
        assert abs(_poly_area_sign(poly)) == pytest.approx(12.0)

    def test_triangulate_triangle(self):
        poly = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        tris = _triangulate_polygon(poly)
        assert len(tris) == 1

    def test_triangulate_square(self):
        poly = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        tris = _triangulate_polygon(poly)
        assert len(tris) == 2

    def test_triangulate_pentagon(self):
        import math
        poly = [(math.cos(2*math.pi*i/5), math.sin(2*math.pi*i/5)) for i in range(5)]
        tris = _triangulate_polygon(poly)
        assert len(tris) == 3


class TestFootprintConvert:
    def test_basic_convert(self):
        # GPS nodes near campus centre
        nodes = [(51.8588, 0.4371), (51.8590, 0.4371),
                 (51.8590, 0.4375), (51.8588, 0.4375)]
        poly = _footprint_to_world(nodes)
        assert len(poly) == 4
        for wx, wy in poly:
            assert isinstance(wx, float)
            assert isinstance(wy, float)

    def test_closed_ring_stripped(self):
        nodes = [(51.8588, 0.4371), (51.8590, 0.4371),
                 (51.8590, 0.4375), (51.8588, 0.4371)]
        poly = _footprint_to_world(nodes)
        assert len(poly) == 3   # last == first → stripped


class TestHeightEstimate:
    def test_chapel_tall(self):
        bld = OsmBuilding(osm_id=1, name="Chapel", bld_type="chapel",
                          gps_nodes=[], centroid=(51.86, 0.44))
        assert _estimate_height(bld) >= 16.0

    def test_house_short(self):
        bld = OsmBuilding(osm_id=2, name="", bld_type="house",
                          gps_nodes=[], centroid=(51.86, 0.44))
        assert _estimate_height(bld) <= 9.0

    def test_generic_fallback(self):
        bld = OsmBuilding(osm_id=3, name="", bld_type="yes",
                          gps_nodes=[], centroid=(51.86, 0.44))
        assert _estimate_height(bld) > 0


class TestWriteDae:
    def test_creates_file(self, tmp_path):
        poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        path = tmp_path / "test.dae"
        _write_dae(path, poly, z_base=76.0, height=10.0, name="test")
        assert path.exists()

    def test_valid_xml(self, tmp_path):
        import xml.etree.ElementTree as ET
        poly = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        path = tmp_path / "test.dae"
        _write_dae(path, poly, z_base=76.0, height=10.0, name="test_bld")
        tree = ET.parse(path)   # raises if invalid XML
        root = tree.getroot()
        assert "COLLADA" in root.tag

    def test_dae_has_geometry(self, tmp_path):
        poly = [(0.0, 0.0), (20.0, 0.0), (20.0, 15.0), (0.0, 15.0)]
        path = tmp_path / "geo.dae"
        _write_dae(path, poly, z_base=50.0, height=8.0, name="geo_test")
        content = path.read_text()
        assert "float_array" in content
        assert "triangles" in content

    def test_dae_contains_correct_z(self, tmp_path):
        poly = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        path = tmp_path / "z.dae"
        _write_dae(path, poly, z_base=76.0, height=12.0, name="z_test")
        content = path.read_text()
        assert "76.000" in content   # base z
        assert "88.000" in content   # 76 + 12

    def test_degenerate_polygon_no_crash(self, tmp_path):
        poly = [(0.0, 0.0), (1.0, 0.0)]   # only 2 points
        path = tmp_path / "degen.dae"
        _write_dae(path, poly, z_base=76.0, height=8.0, name="degen")
        assert not path.exists()   # skipped silently


class TestBuildBuildingMeshes:
    def _make_osm(self, n_buildings=3):
        osm = OsmData()
        for i in range(n_buildings):
            # Small square footprint near Maldon East station
            lat0, lon0 = 51.7310 + i * 0.001, 0.663
            bld = OsmBuilding(
                osm_id   = 100 + i,
                name     = f"TestBld{i}",
                bld_type = "school",
                gps_nodes= [
                    (lat0,        lon0),
                    (lat0 + 5e-4, lon0),
                    (lat0 + 5e-4, lon0 + 5e-4),
                    (lat0,        lon0 + 5e-4),
                    (lat0,        lon0),   # closed
                ],
                centroid = (lat0 + 2.5e-4, lon0 + 2.5e-4),
            )
            osm.buildings.append(bld)
        return osm

    def test_returns_list(self, tmp_path, monkeypatch):
        import tools.buildings3d as b3d
        monkeypatch.setattr(b3d, "_SHAPES_DIR", tmp_path)
        osm = self._make_osm(3)
        result = build_building_meshes(osm, _ELEV_FN)
        assert isinstance(result, list)

    def test_correct_count(self, tmp_path, monkeypatch):
        import tools.buildings3d as b3d
        monkeypatch.setattr(b3d, "_SHAPES_DIR", tmp_path)
        osm = self._make_osm(3)
        result = build_building_meshes(osm, _ELEV_FN)
        assert len(result) == 3

    def test_objects_are_ts_static(self, tmp_path, monkeypatch):
        import tools.buildings3d as b3d
        monkeypatch.setattr(b3d, "_SHAPES_DIR", tmp_path)
        osm = self._make_osm(2)
        result = build_building_meshes(osm, _ELEV_FN)
        for obj in result:
            assert obj["class"] == "TSStatic"

    def test_dae_files_created(self, tmp_path, monkeypatch):
        import tools.buildings3d as b3d
        monkeypatch.setattr(b3d, "_SHAPES_DIR", tmp_path)
        osm = self._make_osm(2)
        build_building_meshes(osm, _ELEV_FN)
        daes = list(tmp_path.glob("*.dae"))
        assert len(daes) == 2

    def test_out_of_world_skipped(self, tmp_path, monkeypatch):
        import tools.buildings3d as b3d
        monkeypatch.setattr(b3d, "_SHAPES_DIR", tmp_path)
        osm = OsmData()
        far_bld = OsmBuilding(
            osm_id=999, name="Far", bld_type="yes",
            gps_nodes=[(52.0, 0.0), (52.001, 0.0), (52.001, 0.001), (52.0, 0.0)],
            centroid=(52.0005, 0.0005),
        )
        osm.buildings.append(far_bld)
        result = build_building_meshes(osm, _ELEV_FN)
        assert len(result) == 0   # out of world bounds

    def test_empty_osm_ok(self, tmp_path, monkeypatch):
        import tools.buildings3d as b3d
        monkeypatch.setattr(b3d, "_SHAPES_DIR", tmp_path)
        result = build_building_meshes(OsmData(), _ELEV_FN)
        assert result == []

    def test_persistent_ids_unique(self, tmp_path, monkeypatch):
        import tools.buildings3d as b3d
        monkeypatch.setattr(b3d, "_SHAPES_DIR", tmp_path)
        osm = self._make_osm(4)
        result = build_building_meshes(osm, _ELEV_FN)
        ids = [o["persistentId"] for o in result]
        assert len(ids) == len(set(ids))
