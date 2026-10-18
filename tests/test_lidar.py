"""
Tests for Environment Agency LiDAR DTM integration — ng-drive network.

Map-agnostic: all raster fixtures are georeferenced to whatever map
map_config.py currently points at, so the same tests run for Maldon,
Felsted and every other England map in the network.

Covers:
  • WGS84 → OSGB36 → EPSG:27700 transforms (against the OS published
    worked example and independently verified PROJ values)
  • GeoTIFF reading (georeferencing tags, nodata handling)
  • Bilinear sampling and nodata infill
  • World-grid resampling alignment (synthetic rasters with known
    georeferencing)
  • WCS request construction and error handling
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from tools import lidar
from tools.lidar import (
    fill_nodata,
    lidar_base_grid,
    osgb36_to_grid,
    read_lidar_geotiff,
    sample_bilinear,
    wgs84_to_osgb36,
)
from tools.constants import (
    BBOX, CENTER_LAT, CENTER_LON, GRID_SIZE,
    LAT_SCALE, LON_SCALE, SQUARE_SIZE, WORLD_HALF,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_test_geotiff(path, grid, x0, y0, px=2.0, py=2.0):
    """Write a north-up float32 GeoTIFF with a ModelTransformation tag."""
    matrix = (px, 0.0, 0.0, x0,
              0.0, -py, 0.0, y0,
              0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 1.0)
    tifffile.imwrite(path, np.asarray(grid, dtype=np.float32),
                     extratags=[(34264, "d", 16, matrix, True)],
                     photometric="minisblack")


def _raster_covering_world(px=4.0, pad_m=500.0):
    """Return (x0, y0, rows, cols) of a north-up raster covering the world."""
    s, w, n, e = BBOX
    lats = np.array([s, s, n, n])
    lons = np.array([w, e, w, e])
    ee, nn = wgs84_to_osgb36(lats, lons)
    x0 = np.floor((ee.min() - pad_m) / px) * px
    y0 = np.ceil((nn.max() + pad_m) / px) * px
    cols = int(np.ceil((ee.max() + pad_m - x0) / px)) + 1
    rows = int(np.ceil((y0 - (nn.min() - pad_m)) / px)) + 1
    return float(x0), float(y0), rows, cols, px


@pytest.fixture
def plane_tif(tmp_path):
    """Raster covering the whole world area, elevation = easting / 1000."""
    x0, y0, rows, cols, px = _raster_covering_world()
    east = x0 + (np.arange(cols) + 0.5) * px
    grid = np.tile(east, (rows, 1)) / 1000.0
    path = tmp_path / "plane.tif"
    _write_test_geotiff(path, grid, x0, y0, px, px)
    return path, dict(x0=x0, y0=y0, px=px, py=px, rows=rows, cols=cols)


def _world_centre_latlon(wx, wy):
    return CENTER_LAT + wy / LAT_SCALE, CENTER_LON + wx / LON_SCALE


# ── Coordinate transforms ─────────────────────────────────────────────────────

class TestOsgb36ToGrid:
    def test_os_worked_example(self):
        """OS Guide Annex C: 52°39'27.2531\"N 1°43'4.5177\"E."""
        lat = 52 + 39 / 60 + 27.2531 / 3600
        lon = 1 + 43 / 60 + 4.5177 / 3600
        e, n = osgb36_to_grid(np.array([lat]), np.array([lon]))
        assert abs(e[0] - 651409.903) < 0.01
        assert abs(n[0] - 313177.270) < 0.01

    def test_origin(self):
        """The true origin projects to the projection's false origin."""
        e, n = osgb36_to_grid(np.array([49.0]), np.array([-2.0]))
        assert abs(e[0] - 400_000.0) < 0.01
        assert abs(n[0] - (-100_000.0)) < 0.01

    def test_easting_increases_with_longitude(self):
        e1, _ = osgb36_to_grid(np.array([52.0]), np.array([-1.0]))
        e2, _ = osgb36_to_grid(np.array([52.0]), np.array([0.0]))
        assert e2[0] > e1[0]

    def test_northing_increases_with_latitude(self):
        _, n1 = osgb36_to_grid(np.array([52.0]), np.array([0.0]))
        _, n2 = osgb36_to_grid(np.array([53.0]), np.array([0.0]))
        assert n2[0] > n1[0]

    def test_vectorised(self):
        lats = np.array([51.5, 52.0, 53.0])
        lons = np.array([-0.1, 0.0, 0.1])
        e, n = osgb36_to_grid(lats, lons)
        assert e.shape == n.shape == (3,)


class TestWgs84ToOsgb36:
    # Reference values computed with PROJ (EPSG:4326 → EPSG:27700).
    # Reference values computed with PROJ (EPSG:4326 → EPSG:27700).
    # "Map centre" is the configured CENTER_LAT/CENTER_LON anchor.
    REFERENCE = [
        ("Felsted School", 51.8588, 0.4371, 567923.87, 220608.88),
        ("Norwich",        52.6309, 1.2974, 623243.56, 308787.53),
        ("Maldon centre",  51.7314, 0.6633, 584016.03, 206988.34),
    ]

    @pytest.mark.parametrize("name,lat,lon,e_ref,n_ref", REFERENCE)
    def test_matches_reference(self, name, lat, lon, e_ref, n_ref):
        e, n = wgs84_to_osgb36(np.array([lat]), np.array([lon]))
        dist = np.hypot(e[0] - e_ref, n[0] - n_ref)
        assert dist < 2.5, f"{name}: {dist:.2f} m off reference"

    def test_map_centre_inside_world_bbox(self):
        """The configured centre must project into a plausible BNG box."""
        e, n = wgs84_to_osgb36(np.array([CENTER_LAT]), np.array([CENTER_LON]))
        assert 0 < e[0] < 700_000
        assert 0 < n[0] < 1_300_000


# ── GeoTIFF reading ───────────────────────────────────────────────────────────

class TestReadGeotiff:
    def test_reads_grid_and_transform(self, plane_tif):
        path, tr = plane_tif
        grid, transform = read_lidar_geotiff(path)
        assert grid.shape == (tr["rows"], tr["cols"])
        assert transform["x0"] == pytest.approx(tr["x0"])
        assert transform["y0"] == pytest.approx(tr["y0"])
        assert transform["px"] == pytest.approx(tr["px"])
        assert transform["py"] == pytest.approx(tr["py"])

    def test_nodata_becomes_nan(self, tmp_path):
        grid = np.full((10, 10), 5.0, dtype=np.float32)
        grid[0, 0] = -3.4028235e38
        path = tmp_path / "nodata.tif"
        _write_test_geotiff(path, grid, 0.0, 0.0)
        read, _ = read_lidar_geotiff(path)
        assert np.isnan(read[0, 0])
        assert np.isfinite(read[1, 1])

    def test_missing_georeferencing_raises(self, tmp_path):
        path = tmp_path / "plain.tif"
        tifffile.imwrite(path, np.ones((4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="georeferencing"):
            read_lidar_geotiff(path)


class TestSampleBilinear:
    def test_plane_is_exact(self, plane_tif):
        path, tr = plane_tif
        grid, transform = read_lidar_geotiff(path)
        # Sample two points near the map centre
        lat, lon = _world_centre_latlon(0.0, 0.0)
        e, n = wgs84_to_osgb36(np.array([lat, lat]), np.array([lon, lon + 0.004]))
        out = sample_bilinear(grid, transform, e, n)
        np.testing.assert_allclose(out, e / 1000.0, atol=1e-3)

    def test_clamps_outside_extent(self, plane_tif):
        path, tr = plane_tif
        grid, transform = read_lidar_geotiff(path)
        e = np.array([0.0, 10_000_000.0])
        n = np.array([0.0, 10_000_000.0])
        out = sample_bilinear(grid, transform, e, n)
        assert np.all(np.isfinite(out))

    def test_preserves_shape(self, plane_tif):
        path, tr = plane_tif
        grid, transform = read_lidar_geotiff(path)
        lat, lon = _world_centre_latlon(0.0, 0.0)
        e, n = wgs84_to_osgb36(np.full((7, 9), lat), np.full((7, 9), lon))
        assert sample_bilinear(grid, transform, e, n).shape == (7, 9)


class TestFillNodata:
    def test_fills_void_with_neighbour(self):
        grid = np.ones((5, 5), dtype=np.float32)
        grid[2, 2] = np.nan
        filled = fill_nodata(grid)
        assert np.all(np.isfinite(filled))
        assert filled[2, 2] == pytest.approx(1.0)

    def test_no_nan_is_noop(self):
        grid = np.arange(9, dtype=np.float32).reshape(3, 3)
        np.testing.assert_array_equal(fill_nodata(grid), grid)

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="entirely nodata"):
            fill_nodata(np.full((3, 3), np.nan, dtype=np.float32))


# ── World-grid alignment ──────────────────────────────────────────────────────

class TestWorldGridAlignment:
    def test_plane_samples_match_projection(self, plane_tif):
        """Resampled world grid must equal easting/1000 from the transform."""
        path, _ = plane_tif
        grid = lidar_base_grid(path)
        assert grid.shape == (GRID_SIZE, GRID_SIZE)

        idx = np.arange(GRID_SIZE)
        wx = -WORLD_HALF + idx * SQUARE_SIZE
        wy = WORLD_HALF - idx * SQUARE_SIZE
        xx, yy = np.meshgrid(wx, wy)
        lat = CENTER_LAT + yy / LAT_SCALE
        lon = CENTER_LON + xx / LON_SCALE
        e, n = wgs84_to_osgb36(lat, lon)
        np.testing.assert_allclose(grid, e / 1000.0, atol=5e-3)

    def test_feature_lands_at_expected_world_position(self, tmp_path):
        """A single-cell marker should appear at its true world location."""
        x0, y0, rows, cols, px = _raster_covering_world()
        grid = np.zeros((rows, cols), dtype=np.float32)

        # Marker near the map centre: world (0, 0) → BNG → raster cell
        lat, lon = _world_centre_latlon(0.0, 0.0)
        ce, cn = wgs84_to_osgb36(np.array([lat]), np.array([lon]))
        col = int((ce[0] - x0) / px)
        row = int((y0 - cn[0]) / px)
        grid[row, col] = 99.0
        path = tmp_path / "marker.tif"
        _write_test_geotiff(path, grid, x0, y0, px, px)

        marker_e = x0 + (col + 0.5) * px
        marker_n = y0 - (row + 0.5) * px

        sampled = lidar_base_grid(path)
        r, c = np.unravel_index(np.argmax(sampled), sampled.shape)
        wx = c * SQUARE_SIZE - WORLD_HALF + SQUARE_SIZE / 2
        wy = WORLD_HALF - r * SQUARE_SIZE - SQUARE_SIZE / 2

        mlat = CENTER_LAT + wy / LAT_SCALE
        mlon = CENTER_LON + wx / LON_SCALE
        e, n = wgs84_to_osgb36(np.array([mlat]), np.array([mlon]))
        assert np.hypot(e[0] - marker_e, n[0] - marker_n) < 2 * px


# ── WCS download ──────────────────────────────────────────────────────────────

class TestWcsFetch:
    def test_builds_valid_request_and_caches(self, monkeypatch, tmp_path):
        captured = {}

        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return self.payload

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            buf = tmp_path / "payload.tif"
            _write_test_geotiff(buf, np.ones((10, 10), dtype=np.float32),
                                560_000.0, 230_000.0)
            return FakeResponse(buf.read_bytes())

        monkeypatch.setattr(lidar.urllib.request, "urlopen", fake_urlopen)
        out = tmp_path / "lidar" / "tile.tif"
        lidar.fetch_lidar_tile(out)

        assert out.exists()
        url = captured["url"]
        assert "GetCoverage" in url
        assert "subset=E(" in url and "subset=N(" in url
        assert lidar._WCS_COVERAGE in url
        assert "geotiff" in url

    def test_raises_on_xml_exception(self, monkeypatch, tmp_path):
        class FakeResponse:
            def read(self):
                return b'<?xml version="1.0"?><ExceptionReport/>'

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(lidar.urllib.request, "urlopen",
                            lambda req, timeout=None: FakeResponse())
        with pytest.raises(RuntimeError, match="WCS exception"):
            lidar.fetch_lidar_tile(tmp_path / "tile.tif")


# ── Availability behaviour ────────────────────────────────────────────────────

class TestBuildLidarBase:
    def test_returns_none_offline_without_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(lidar, "LIDAR_GRID_CACHE", tmp_path / "none.npy")
        monkeypatch.setattr(lidar, "LIDAR_TIF_CACHE", tmp_path / "none.tif")
        assert lidar.build_lidar_base(online=False) is None

    def test_uses_cached_grid(self, monkeypatch, tmp_path):
        cache = tmp_path / "grid.npy"
        np.save(cache, np.full((GRID_SIZE, GRID_SIZE), 5.0, dtype=np.float32))
        monkeypatch.setattr(lidar, "LIDAR_GRID_CACHE", cache)
        out = lidar.build_lidar_base(online=False)
        assert out is not None and out.shape == (GRID_SIZE, GRID_SIZE)
        assert out[0, 0] == pytest.approx(5.0)
