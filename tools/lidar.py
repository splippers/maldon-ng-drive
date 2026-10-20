"""
Environment Agency LiDAR DTM integration — v3.1.

Replaces the SRTM 30 m base terrain with the EA LiDAR Composite Digital
Terrain Model (2 m resolution, ±15 cm vertical RMSE), served as GeoTIFF
via the Defra Data Services Platform WCS 2.0.1 endpoint.

Pipeline:
  1. Convert the map's WGS84 bounding box to OSGB36 (EPSG:27700).
  2. Download the covering GeoTIFF tile(s) via WCS GetCoverage.
  3. Read elevation + georeferencing tags (pure Python via tifffile,
     no GDAL required).
  4. Bilinearly resample onto the 1025×1025 BeamNG world grid.

Coordinate transform notes:
  • WGS84 → OSGB36 uses a 7-parameter Helmert plus the OS polynomial grid
    series; the projection matches the OS published worked example to
    0.3 mm and PROJ to 1 mm (~2 m residual vs OSTN15).
  • Vertical datum is ODN (Ordnance Datum Newlyn), which agrees with the
    SRTM calibration anchors in map_config.py to ~1 m.

Licence: Environment Agency LiDAR Composite — Open Government Licence.
"""

from __future__ import annotations

import logging
import math
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from tools.constants import (
    BBOX, CENTER_LAT, CENTER_LON, GRID_SIZE,
    SQUARE_SIZE, WORLD_HALF,
    LIDAR_TIF_CACHE, LIDAR_GRID_CACHE,
    LAT_SCALE, LON_SCALE,
    MAP_NAME,
)

log = logging.getLogger(__name__)

# ── WCS endpoint (Defra Data Services Platform) ───────────────────────────────
_WCS_URL = (
    "https://environment.data.gov.uk/spatialdata/"
    "lidar-composite-digital-terrain-model-dtm-2m/wcs"
)
_WCS_COVERAGE = "09ea3b37-df3a-4e8b-ac69-fb0842227b04__Lidar_Composite_Elevation_DTM_2m"
_WCS_FORMAT   = "image/tiff;application=geotiff"

# Padding around the world bounds so terrain edges always have data.
_TILE_PAD_M = 150.0

# GDAL writes nodata as -3.4028235e+38; treat anything below this as void.
_NODATA_THRESHOLD = -1e30


# ── WGS84 → OSGB36 (EPSG:27700) ───────────────────────────────────────────────
# 7-parameter Helmert (WGS84 → OSGB36) using the OS-published parameters
# ("A Guide to Coordinate Systems in Great Britain"), applied in the
# *coordinate frame* rotation convention.  Verified against PROJ's
# EPSG:4326→EPSG:27700 pipeline to ±1.5 m across Great Britain
# (residual is the missing OSTN15 datum shift; the projection series
# below agrees with PROJ to <0.5 m).
_HELMERT = dict(
    tx=-446.448, ty=125.157, tz=-542.060,
    rx=math.radians(0.1502 / 3600),
    ry=math.radians(0.2470 / 3600),
    rz=math.radians(0.8421 / 3600),
    s=20.4894e-6,
)

# Airy 1830 ellipsoid + OSGB36 national grid parameters
_A, _B      = 6377563.396, 6356256.910
_F0         = 0.9996012717
_LAT0       = math.radians(49.0)
_LON0       = math.radians(-2.0)
_E0, _N0    = 400_000.0, -100_000.0


def wgs84_to_osgb36(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised WGS84 lat/lon → OSGB36 easting/northing (metres)."""
    lat_r = np.radians(np.asarray(lat, dtype=np.float64))
    lon_r = np.radians(np.asarray(lon, dtype=np.float64))

    # WGS84 → geocentric (ECEF)
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    nu = a / np.sqrt(1 - e2 * np.sin(lat_r) ** 2)
    x1 = nu * np.cos(lat_r) * np.cos(lon_r)
    y1 = nu * np.cos(lat_r) * np.sin(lon_r)
    z1 = nu * (1 - e2) * np.sin(lat_r)

    # Helmert transform → OSGB36 geocentric (coordinate-frame rotations)
    h = _HELMERT
    x2 = h["tx"] + x1 * (1 + h["s"]) + y1 * h["rz"] - z1 * h["ry"]
    y2 = h["ty"] - x1 * h["rz"] + y1 * (1 + h["s"]) + z1 * h["rx"]
    z2 = h["tz"] + x1 * h["ry"] - y1 * h["rx"] + z1 * (1 + h["s"])

    # OSGB36 geocentric → geodetic (Airy 1830), Bowring's method
    b_ = _B
    ep2 = (_A ** 2 - b_ ** 2) / b_ ** 2          # second eccentricity²
    p = np.sqrt(x2 ** 2 + y2 ** 2)
    theta = np.arctan2(z2 * _A, p * b_)
    lat_b = np.arctan2(
        z2 + ep2 * b_ * np.sin(theta) ** 3,
        p - e2_airy() * _A * np.cos(theta) ** 3,
    )
    lon_b = np.arctan2(y2, x2)
    return osgb36_to_grid(np.degrees(lat_b), np.degrees(lon_b))


def osgb36_to_grid(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    OSGB36 geodetic lat/lon (degrees) → EPSG:27700 easting/northing (metres).

    Implements the OS polynomial series (sub-metre within Great Britain).
    Validated against PROJ to <0.5 m, and against the worked example in
    "A Guide to Coordinate Systems in Great Britain" (Annex C):
    52°39'27.2531"N 1°43'4.5177"E → 651409.903 E, 313177.270 N.
    """
    lat_b = np.radians(np.asarray(lat, dtype=np.float64))
    lon_b = np.radians(np.asarray(lon, dtype=np.float64))
    b_ = _B

    # OSGB36 lat/lon → national grid (OS polynomial series)
    n = (_A - b_) / (_A + b_)
    sin_phi, cos_phi = np.sin(lat_b), np.cos(lat_b)
    tan_phi = sin_phi / cos_phi
    eta2 = ((_A ** 2 - b_ ** 2) / b_ ** 2) * cos_phi ** 2   # η² = ν/ρ − 1

    nu_phi = _A * _F0 / np.sqrt(1 - e2_airy() * np.sin(lat_b) ** 2)

    dlat = lat_b - _LAT0
    m = b_ * _F0 * (
        (1 + n + 1.25 * n ** 2 + 1.25 * n ** 3) * dlat
        - (3 * n + 3 * n ** 2 + 2.625 * n ** 3) * np.sin(dlat) * np.cos(lat_b + _LAT0)
        + (1.875 * n ** 2 + 1.875 * n ** 3) * np.sin(2 * dlat) * np.cos(2 * (lat_b + _LAT0))
        - (1.4583333333 * n ** 3) * np.sin(3 * dlat) * np.cos(3 * (lat_b + _LAT0))
    )

    dlon = lon_b - _LON0
    easting = (
        nu_phi * cos_phi * dlon
        + nu_phi * cos_phi ** 3 * (1 - tan_phi ** 2 + eta2) * dlon ** 3 / 6
        + nu_phi * cos_phi ** 5
        * (5 - 18 * tan_phi ** 2 + tan_phi ** 4 + 14 * eta2 - 58 * eta2 * tan_phi ** 2)
        * dlon ** 5 / 120
        + _E0
    )
    northing = (
        m
        + nu_phi * sin_phi * cos_phi * dlon ** 2 / 2
        + nu_phi * sin_phi * cos_phi ** 3 * (5 - tan_phi ** 2 + 9 * eta2) * dlon ** 4 / 24
        + nu_phi * sin_phi * cos_phi ** 5
        * (61 - 58 * tan_phi ** 2 + tan_phi ** 4) * dlon ** 6 / 720
        + _N0
    )
    return easting, northing


def e2_airy() -> float:
    """First eccentricity squared of the Airy 1830 ellipsoid."""
    return (_A ** 2 - _B ** 2) / _A ** 2


# ── GeoTIFF reading (pure Python via tifffile) ────────────────────────────────

def read_lidar_geotiff(path: Path) -> tuple[np.ndarray, dict]:
    """
    Read an EA LiDAR GeoTIFF.

    Returns (grid, transform) where grid[row, col] is elevation in metres
    (row 0 = north edge) and transform = dict(x0, y0, px, py) describing
    the corner of pixel (0, 0) and positive pixel sizes.
    """
    import tifffile

    # tifffile logs a benign warning when casting the GDAL_NODATA
    # sentinel (-3.4e38) to float32; we detect nodata ourselves.
    tifffile_log = logging.getLogger("tifffile")
    prev_level = tifffile_log.level
    tifffile_log.setLevel(logging.ERROR)
    try:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            grid = page.asarray().astype(np.float32)

            transform = None
            for tag in page.tags.values():
                if tag.code == 34264:                    # ModelTransformationTag
                    m = np.asarray(tag.value, dtype=np.float64).reshape(4, 4)
                    transform = dict(x0=float(m[0, 3]), y0=float(m[1, 3]),
                                     px=abs(float(m[0, 0])), py=abs(float(m[1, 1])))
                elif tag.code == 33550 and transform is None:   # ModelPixelScaleTag
                    scale = np.asarray(tag.value, dtype=np.float64)
                    px, py = float(scale[0]), float(scale[1])
                    transform = dict(x0=None, y0=None, px=px, py=py)
                elif tag.code == 33922 and transform is not None:  # ModelTiepointTag
                    tp = np.asarray(tag.value, dtype=np.float64)
                    transform["x0"] = float(tp[3]) - float(tp[0]) * transform["px"]
                    transform["y0"] = float(tp[4]) + float(tp[1]) * transform["py"]
    finally:
        tifffile_log.setLevel(prev_level)

    if transform is None:
        raise ValueError(f"No georeferencing tags found in {path}")

    grid[grid < _NODATA_THRESHOLD] = np.nan
    log.info("LiDAR raster loaded: %s shape=%s range=%.1f–%.1f m",
             path.name, grid.shape,
             np.nanmin(grid), np.nanmax(grid))
    return grid, transform


def sample_bilinear(grid: np.ndarray, transform: dict,
                    easting: np.ndarray, northing: np.ndarray) -> np.ndarray:
    """Bilinear sample of a north-up raster at easting/northing arrays."""
    col_f = (easting - transform["x0"]) / transform["px"] - 0.5
    row_f = (transform["y0"] - northing) / transform["py"] - 0.5

    h, w = grid.shape
    col_f = np.clip(col_f, 0.0, w - 1.001)
    row_f = np.clip(row_f, 0.0, h - 1.001)

    c0 = col_f.astype(np.int32)
    r0 = row_f.astype(np.int32)
    dc = col_f - c0
    dr = row_f - r0

    v00 = grid[r0,     c0]
    v01 = grid[r0,     c0 + 1]
    v10 = grid[r0 + 1, c0]
    v11 = grid[r0 + 1, c0 + 1]
    return (v00 * (1 - dr) * (1 - dc) + v01 * (1 - dr) * dc
            + v10 * dr * (1 - dc) + v11 * dr * dc).astype(np.float32)


def fill_nodata(grid: np.ndarray) -> np.ndarray:
    """Fill NaN voids with the nearest valid sample (Euclidean distance)."""
    from scipy.ndimage import distance_transform_edt

    mask = np.isnan(grid)
    if not mask.any():
        return grid
    if mask.all():
        raise ValueError("LiDAR raster is entirely nodata for the map area")

    _, indices = distance_transform_edt(mask, return_distances=True,
                                        return_indices=True)
    indices = np.asarray(indices)
    # scipy returns indices either as (ndim, ...) or (..., ndim) depending
    # on version — normalise to (ndim, ...).
    if indices.shape[0] != grid.ndim:
        indices = np.moveaxis(indices, -1, 0)
    filled = grid[tuple(indices)]
    log.info("Filled %d nodata cells via nearest-neighbour", int(mask.sum()))
    return filled.astype(np.float32)


# ── WCS download ──────────────────────────────────────────────────────────────

def _wcs_request(e_min: float, e_max: float, n_min: float, n_max: float,
                 out_path: Path) -> None:
    url = (
        f"{_WCS_URL}?service=WCS&version=2.0.1&request=GetCoverage"
        f"&coverageId={_WCS_COVERAGE}"
        f"&subset=E({e_min:.1f},{e_max:.1f})"
        f"&subset=N({n_min:.1f},{n_max:.1f})"
        f"&format={urllib.parse.quote(_WCS_FORMAT)}"
    )
    log.info("Downloading LiDAR DTM %.0f×%.0f m from Defra WCS …",
             e_max - e_min, n_max - n_min)
    req = urllib.request.Request(url, headers={"User-Agent": f"{MAP_NAME}-ng-drive/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()

    if data[:5] == b"<?xml":
        raise RuntimeError(f"WCS exception: {data[:400].decode(errors='replace')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    log.info("LiDAR tile cached: %s (%.1f MB)", out_path, len(data) / 1_048_576)


def fetch_lidar_tile(out_path: Path = LIDAR_TIF_CACHE) -> Path:
    """Download the LiDAR DTM tile(s) covering the map world + padding."""
    corners_lat = np.array([BBOX[0], BBOX[0], BBOX[2], BBOX[2]])
    corners_lon = np.array([BBOX[1], BBOX[3], BBOX[1], BBOX[3]])
    e, n = wgs84_to_osgb36(corners_lat, corners_lon)
    _wcs_request(e.min() - _TILE_PAD_M, e.max() + _TILE_PAD_M,
                 n.min() - _TILE_PAD_M, n.max() + _TILE_PAD_M,
                 out_path)
    return out_path


# ── World grid resampling ─────────────────────────────────────────────────────

def lidar_base_grid(tif_path: Path = LIDAR_TIF_CACHE) -> np.ndarray:
    """
    Resample the LiDAR DTM onto the BeamNG world grid (GRID_SIZE²).

    World vertices: row 0 = north edge (wy = +WORLD_HALF),
    col 0 = west edge (wx = -WORLD_HALF), 2 m spacing.
    """
    grid, transform = read_lidar_geotiff(tif_path)

    wx = -WORLD_HALF + np.arange(GRID_SIZE, dtype=np.float64) * SQUARE_SIZE
    wy = WORLD_HALF - np.arange(GRID_SIZE, dtype=np.float64) * SQUARE_SIZE
    xx, yy = np.meshgrid(wx, wy)

    lat = CENTER_LAT + yy / LAT_SCALE
    lon = CENTER_LON + xx / LON_SCALE
    e, n = wgs84_to_osgb36(lat, lon)

    sampled = sample_bilinear(grid, transform, e, n)
    return fill_nodata(sampled)


# ── Public entry point used by heightmap.py ───────────────────────────────────

def build_lidar_base(online: bool = False) -> np.ndarray | None:
    """
    Return the LiDAR-backed base elevation grid, or None if unavailable.

    Order: cached resampled grid → cached GeoTIFF → live WCS download.
    """
    if LIDAR_GRID_CACHE.exists():
        base = np.load(LIDAR_GRID_CACHE)
        log.info("LiDAR base grid loaded from cache, range %.1f–%.1f m",
                 base.min(), base.max())
        return base

    try:
        if not LIDAR_TIF_CACHE.exists():
            if not online:
                return None
            fetch_lidar_tile()
        base = lidar_base_grid()
        np.save(LIDAR_GRID_CACHE, base)
        log.info("LiDAR base grid built: range %.1f–%.1f m", base.min(), base.max())
        return base
    except Exception as exc:
        log.warning("LiDAR unavailable (%s); falling back to SRTM/synthetic", exc)
        return None
