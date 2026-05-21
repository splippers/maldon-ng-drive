"""Tests for satellite imagery tile compositor — v3.0."""

import math
import numpy as np
import pytest

from tools.satellite import (
    _deg2tile, _tile_lon_left, _tile_merc_top,
    _gps_to_canvas_px, _merc_y, _world_bbox,
    write_satellite_material,
)
from tools.constants import CENTER_LAT, CENTER_LON, WORLD_HALF, LEVELS_DIR


class TestTileMath:
    def test_deg2tile_known(self):
        """Zoom 0 has exactly one tile covering the world."""
        tx, ty = _deg2tile(0.0, 0.0, zoom=0)
        assert tx == 0 and ty == 0

    def test_deg2tile_north_pole_zero(self):
        tx, ty = _deg2tile(85.0, 0.0, zoom=1)
        assert ty == 0

    def test_deg2tile_center(self):
        """Map centre at zoom 17 should land in a known x/y range."""
        tx, ty = _deg2tile(CENTER_LAT, CENTER_LON, zoom=17)
        assert 65760 < tx < 65800   # Maldon ~51.73°N, 0.66°E
        assert 43440 < ty < 43470

    def test_tile_lon_left_zero(self):
        assert _tile_lon_left(0, zoom=0) == pytest.approx(-180.0)

    def test_tile_lon_left_half(self):
        n = 2 ** 10
        assert _tile_lon_left(n // 2, zoom=10) == pytest.approx(0.0, abs=0.01)

    def test_merc_y_equator(self):
        assert _merc_y(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_merc_y_positive_north(self):
        assert _merc_y(51.0) > 0

    def test_tile_merc_top_monotone(self):
        """Increasing ty → decreasing Mercator latitude (moving south)."""
        m0 = _tile_merc_top(100, 17)
        m1 = _tile_merc_top(101, 17)
        assert m0 > m1


class TestWorldBbox:
    def test_returns_four_values(self):
        bbox = _world_bbox()
        assert len(bbox) == 4

    def test_lat_ordering(self):
        lat_s, lon_w, lat_n, lon_e = _world_bbox()
        assert lat_s < lat_n

    def test_lon_ordering(self):
        lat_s, lon_w, lat_n, lon_e = _world_bbox()
        assert lon_w < lon_e

    def test_bbox_near_center(self):
        lat_s, lon_w, lat_n, lon_e = _world_bbox()
        assert lat_s < CENTER_LAT < lat_n
        assert lon_w < CENTER_LON < lon_e

    def test_bbox_size_approx_2km(self):
        lat_s, lon_w, lat_n, lon_e = _world_bbox()
        ns_m = (lat_n - lat_s) * 111139
        assert 1900 < ns_m < 2200, f"N-S span {ns_m:.0f} m, expected ~2048 m"


class TestCanvasPixelMapping:
    """Tests for _gps_to_canvas_px using the actual Maldon world tile range."""

    _ZOOM  = 17
    _TX_MIN, _TY_MIN = 65772, 43448   # Maldon NW corner
    _TX_MAX, _TY_MAX = 65782, 43458   # Maldon SE corner
    _NX = _TX_MAX - _TX_MIN + 1   # 11
    _NY = _TY_MAX - _TY_MIN + 1   # 11
    _CW = _NX * 256
    _CH = _NY * 256

    def _px(self, lat, lon):
        return _gps_to_canvas_px(
            lat, lon,
            self._TX_MIN, self._TY_MIN, self._TX_MAX, self._TY_MAX,
            self._ZOOM, self._CH, self._CW,
        )

    def test_nw_corner_is_top_left(self):
        lat_nw, lon_nw = _world_bbox()[2], _world_bbox()[1]
        row, col = self._px(lat_nw, lon_nw)
        assert row < self._CH // 2
        assert col < self._CW // 2

    def test_se_corner_is_bottom_right(self):
        lat_se, lon_se = _world_bbox()[0], _world_bbox()[3]
        row, col = self._px(lat_se, lon_se)
        assert row > self._CH // 2
        assert col > self._CW // 2

    def test_centre_near_canvas_middle(self):
        row, col = self._px(CENTER_LAT, CENTER_LON)
        assert abs(row - self._CH // 2) < self._CH * 0.4
        assert abs(col - self._CW // 2) < self._CW * 0.4

    def test_clamped_within_canvas(self):
        # Extreme lat/lon outside the tile range
        row, col = self._px(90.0, 180.0)
        assert 0 <= row < self._CH
        assert 0 <= col < self._CW


class TestSatelliteMaterial:
    def test_write_creates_json(self, tmp_path, monkeypatch):
        import tools.satellite as sat_mod
        from pathlib import Path
        # Redirect _SAT_DIR to tmp_path
        monkeypatch.setattr(sat_mod, "_SAT_DIR", tmp_path)
        result = write_satellite_material()
        assert result.exists()

    def test_material_json_valid(self, tmp_path, monkeypatch):
        import json, tools.satellite as sat_mod
        monkeypatch.setattr(sat_mod, "_SAT_DIR", tmp_path)
        path = write_satellite_material()
        data = json.loads(path.read_text())
        assert "felsted_satellite" in data
        mat = data["felsted_satellite"]
        assert mat["detailScale"] == 2048.0
        assert "satellite.png" in mat["diffuseMap"]

    def test_flat_normal_created(self, tmp_path, monkeypatch):
        import tools.satellite as sat_mod
        monkeypatch.setattr(sat_mod, "_SAT_DIR", tmp_path)
        write_satellite_material()
        assert (tmp_path / "flat_n.png").exists()
