"""Tests for coordinate projection utilities — updated for v3.0 (2 m terrain)."""

import math
import pytest
from tools.constants import (
    gps_to_world, world_to_gps, world_to_hm, hm_to_world,
    CENTER_LAT, CENTER_LON,
    BLOCK_SIZE, GRID_SIZE, SQUARE_SIZE, WORLD_HALF,
)


class TestGpsToWorld:
    def test_centre_maps_to_origin(self):
        x, y = gps_to_world(CENTER_LAT, CENTER_LON)
        assert abs(x) < 0.01
        assert abs(y) < 0.01

    def test_north_is_positive_y(self):
        _, y = gps_to_world(CENTER_LAT + 0.001, CENTER_LON)
        assert y > 0

    def test_east_is_positive_x(self):
        x, _ = gps_to_world(CENTER_LAT, CENTER_LON + 0.001)
        assert x > 0

    def test_roundtrip(self):
        lat, lon = 51.858, 0.440
        x, y = gps_to_world(lat, lon)
        lat2, lon2 = world_to_gps(x, y)
        assert abs(lat2 - lat) < 1e-7
        assert abs(lon2 - lon) < 1e-7

    def test_scale_1km_north(self):
        """0.009° latitude ≈ 1 km north."""
        _, y = gps_to_world(CENTER_LAT + 0.009, CENTER_LON)
        assert 950 < y < 1050

    def test_scale_1km_east(self):
        """East scale at ~52°N: 1 km ≈ 0.0145–0.0155°."""
        import math
        lon_per_km = 1000 / (111139 * math.cos(math.radians(CENTER_LAT)))
        x, _ = gps_to_world(CENTER_LAT, CENTER_LON + lon_per_km)
        assert 950 < x < 1050

    def test_world_fits_inside_bbox(self):
        """All four world corners must lie inside BBOX."""
        from tools.constants import BBOX
        s, w, n, e = BBOX
        for wx, wy in [
            (-WORLD_HALF, -WORLD_HALF), (-WORLD_HALF, WORLD_HALF),
            ( WORLD_HALF, -WORLD_HALF), ( WORLD_HALF, WORLD_HALF),
        ]:
            lat, lon = world_to_gps(wx, wy)
            assert s <= lat <= n
            assert w <= lon <= e

    def test_map_centre_within_world(self):
        """Map centre should be at or very near world origin."""
        x, y = gps_to_world(CENTER_LAT, CENTER_LON)
        assert abs(x) < 1.0
        assert abs(y) < 1.0


class TestTerrainResolution:
    def test_block_size_is_1024(self):
        assert BLOCK_SIZE == 1024

    def test_square_size_is_2m(self):
        assert SQUARE_SIZE == 2.0

    def test_world_is_2048m(self):
        assert BLOCK_SIZE * SQUARE_SIZE == 2048.0

    def test_world_half_is_1024(self):
        assert WORLD_HALF == 1024.0

    def test_grid_size_is_1025(self):
        assert GRID_SIZE == BLOCK_SIZE + 1 == 1025


class TestWorldToHm:
    def test_centre_maps_to_middle(self):
        r, c = world_to_hm(0, 0)
        mid = BLOCK_SIZE // 2
        assert abs(r - mid) <= 1
        assert abs(c - mid) <= 1

    def test_north_is_low_row(self):
        r_n, _ = world_to_hm(0,  500)
        r_s, _ = world_to_hm(0, -500)
        assert r_n < r_s

    def test_east_is_high_col(self):
        _, c_e = world_to_hm( 500, 0)
        _, c_w = world_to_hm(-500, 0)
        assert c_e > c_w

    def test_clamped_within_grid(self):
        for wx, wy in [(-9999, 0), (9999, 0), (0, -9999), (0, 9999)]:
            r, c = world_to_hm(wx, wy)
            assert 0 <= r <= BLOCK_SIZE
            assert 0 <= c <= BLOCK_SIZE

    def test_resolution_2m_per_pixel(self):
        """Two points 2 m apart should map to adjacent grid cells."""
        r0, c0 = world_to_hm(0, 0)
        r1, c1 = world_to_hm(SQUARE_SIZE, 0)   # 2 m east
        assert c1 == c0 + 1

    def test_1km_span(self):
        """1 000 m east should shift column by ~500 cells."""
        _, c0 = world_to_hm(0, 0)
        _, c1 = world_to_hm(1000, 0)
        assert abs((c1 - c0) - 500) <= 1


class TestHmRoundtrip:
    def test_centre_roundtrip(self):
        mid = BLOCK_SIZE // 2
        wx, wy = hm_to_world(mid, mid)
        r, c = world_to_hm(wx, wy)
        assert abs(r - mid) <= 1
        assert abs(c - mid) <= 1
