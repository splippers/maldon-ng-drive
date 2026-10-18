"""
Terrain heightmap generator — v3.1.

Priority order for elevation source:
  1. Environment Agency LiDAR DTM 2 m  (data/<map>_lidar_1025.npy)   ← best
  2. Cached SRTM-30m grid  (data/<map>_srtm_32x32.json)
  3. Live OpenTopoData fetch (--online)
  4. Synthetic procedural model tuned to the map's spot heights

In all cases the final grid is:
  • 1025 × 1025 float32 array (BLOCK_SIZE=1024, 2 m/cell → 2 048 m world)
  • Blended with fBm fine-detail noise
  • Road grade cuts + level-of-interest flattening (non-LiDAR sources only)

When the LiDAR base is used the synthetic shaping is skipped entirely —
the real survey already contains every road cut, kerb, camber, and the
estuary — and only ±0.3 m of micro-detail noise is added.
"""

from __future__ import annotations

import json
import logging
import urllib.request

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from tools.constants import (
    BLOCK_SIZE, GRID_SIZE, SQUARE_SIZE, WORLD_HALF,
    CAMPUS_ELEV, VALLEY_ELEV, HILL_N_ELEV,
    CENTER_LAT, CENTER_LON, BBOX,
    SRTM_CACHE, world_to_hm,
    LAT_SCALE, LON_SCALE,
)

log = logging.getLogger(__name__)

_OPENTOPODATA = "https://api.opentopodata.org/v1/srtm30m"
_BATCH        = 100


# ── fBm noise ─────────────────────────────────────────────────────────────────

def _value_noise(size: int, coarse: int, seed: int) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    pts    = rng.standard_normal((coarse, coarse)).astype(np.float32)
    factor = size / coarse
    zoomed = zoom(pts, factor, order=3)
    # Trim/pad to exact size
    r, c = zoomed.shape
    if r > size: zoomed = zoomed[:size, :]
    if c > size: zoomed = zoomed[:, :size]
    if r < size: zoomed = np.pad(zoomed, ((0, size-r), (0, 0)), mode="edge")
    if c < size: zoomed = np.pad(zoomed, ((0, 0), (0, size-c)), mode="edge")
    return zoomed


def _fbm(size: int, seed: int = 42, octaves: int = 9,
         base_coarse: int = 4, lacunarity: float = 2.0,
         persistence: float = 0.5) -> np.ndarray:
    """Fractional Brownian Motion via layered smooth noise (numpy only)."""
    result = np.zeros((size, size), dtype=np.float32)
    amp    = 1.0
    total  = 0.0
    coarse = base_coarse

    for i in range(octaves):
        layer = _value_noise(size, coarse, seed + i * 7919)
        result += layer * amp
        total  += amp
        amp    *= persistence
        coarse  = min(size // 2, max(2, int(coarse * lacunarity)))

    return result / total


# ── Terrain shaping utilities ─────────────────────────────────────────────────

def _gaussian_bump(grid: np.ndarray, row: int, col: int,
                   radius_m: float, delta: float) -> None:
    """Add a Gaussian hill (+) or depression (−) in-place."""
    sigma = radius_m / SQUARE_SIZE
    rr    = np.arange(GRID_SIZE, dtype=np.float32)[:, None]
    cc    = np.arange(GRID_SIZE, dtype=np.float32)[None, :]
    grid += delta * np.exp(-((rr - row)**2 + (cc - col)**2) / (2 * sigma**2))


def _road_cut(grid: np.ndarray,
              wx0: float, wy0: float, wx1: float, wy1: float,
              z0: float, z1: float, blend_m: float = 18.0) -> None:
    """Blend terrain toward a linear grade along a road segment."""
    r0, c0 = world_to_hm(wx0, wy0)
    r1, c1 = world_to_hm(wx1, wy1)
    steps  = max(abs(r1 - r0), abs(c1 - c0), 1)
    sigma  = blend_m / SQUARE_SIZE

    for i in range(steps + 1):
        t = i / steps
        r = int(r0 + (r1 - r0) * t)
        c = int(c0 + (c1 - c0) * t)
        z = z0 + (z1 - z0) * t

        pad = int(sigma * 3)
        rlo, rhi = max(0, r - pad), min(GRID_SIZE, r + pad + 1)
        clo, chi = max(0, c - pad), min(GRID_SIZE, c + pad + 1)
        rr       = np.arange(rlo, rhi, dtype=np.float32)[:, None]
        cc       = np.arange(clo, chi, dtype=np.float32)[None, :]
        blend    = np.exp(-((rr - r)**2 + (cc - c)**2) / (2 * sigma**2))
        grid[rlo:rhi, clo:chi] = (
            grid[rlo:rhi, clo:chi] * (1 - blend) + z * blend
        )


# ── Latitude/longitude grid sampling ──────────────────────────────────────────

def world_latlon() -> tuple[np.ndarray, np.ndarray]:
    """Latitude/longitude arrays for every world terrain vertex."""
    idx = np.arange(GRID_SIZE, dtype=np.float64)
    wx  = -WORLD_HALF + idx * SQUARE_SIZE
    wy  =  WORLD_HALF - idx * SQUARE_SIZE
    xx, yy = np.meshgrid(wx, wy)
    return (CENTER_LAT + yy / LAT_SCALE, CENTER_LON + xx / LON_SCALE)


def sample_latlon_grid(lats: np.ndarray, lons: np.ndarray,
                       values: np.ndarray,
                       lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Bilinearly sample a regular lat/lon grid onto arbitrary coordinates.

    ``lats``/``lons`` must be ascending 1-D arrays; ``values`` has shape
    (len(lats), len(lons)).  Coordinates outside the grid are clamped.
    """
    fi = np.interp(lat, lats, np.arange(len(lats), dtype=np.float64))
    fj = np.interp(lon, lons, np.arange(len(lons), dtype=np.float64))
    i0 = np.clip(fi.astype(np.int64), 0, len(lats) - 2)
    j0 = np.clip(fj.astype(np.int64), 0, len(lons) - 2)
    ti = (fi - i0).astype(np.float32)
    tj = (fj - j0).astype(np.float32)

    v = (values[i0,     j0    ] * (1 - ti) * (1 - tj)
         + values[i0,     j0 + 1] * (1 - ti) * tj
         + values[i0 + 1, j0    ] * ti * (1 - tj)
         + values[i0 + 1, j0 + 1] * ti * tj)
    return v.astype(np.float32)


# ── SRTM / elevation sources ──────────────────────────────────────────────────

def _load_srtm_cache() -> np.ndarray | None:
    """
    Load the cached 32×32 SRTM grid and resample it onto the world grid.

    The cache spans BBOX (wider than the 2 km world), so it must be
    *sampled* by lat/lon — never stretched/fitted — otherwise terrain
    features land at the wrong scale relative to OSM geometry.
    """
    if not SRTM_CACHE.exists():
        return None
    d    = json.loads(SRTM_CACHE.read_text())
    if "lats" not in d or "lons" not in d:
        log.warning("SRTM cache %s lacks lat/lon georeferencing — ignoring", SRTM_CACHE.name)
        return None
    lats = np.asarray(d["lats"], dtype=np.float64)
    lons = np.asarray(d["lons"], dtype=np.float64)
    raw  = np.asarray(d["elevations"], dtype=np.float32)   # (nlat, nlon), lats ascending

    lat, lon = world_latlon()
    full = sample_latlon_grid(lats, lons, raw, lat, lon)
    full = gaussian_filter(full, sigma=1.0)
    log.info("SRTM cache loaded, range %.1f–%.1f m", full.min(), full.max())
    return full.astype(np.float32)


def _fetch_srtm_live(coarse: int = 32) -> np.ndarray:
    """Download a coarse SRTM grid and resample it onto the world grid."""
    import time
    s, w, n, e = BBOX
    lats = np.linspace(s, n, coarse)
    lons = np.linspace(w, e, coarse).tolist()
    flat_la = [la for la in lats for _ in lons]
    flat_lo = [lo for _ in lats for lo in lons]

    elevs: list[float] = []
    for i in range(0, len(flat_la), _BATCH):
        loc = "|".join(f"{la:.6f},{lo:.6f}"
                       for la, lo in zip(flat_la[i:i+_BATCH], flat_lo[i:i+_BATCH]))
        url = f"{_OPENTOPODATA}?locations={loc}"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=25) as r:
                    data = json.loads(r.read())
                elevs.extend(pt["elevation"] for pt in data["results"])
                time.sleep(1.1)
                break
            except Exception as exc:
                log.warning("SRTM batch %d attempt %d: %s", i//_BATCH, attempt, exc)
                time.sleep(8 * (attempt + 1))
        else:
            elevs.extend([CAMPUS_ELEV] * min(_BATCH, len(flat_la) - i))

    raw  = np.array(elevs[:coarse*coarse], dtype=np.float32).reshape(coarse, coarse)
    lat, lon = world_latlon()
    full = sample_latlon_grid(lats, np.asarray(lons), raw, lat, lon)
    return gaussian_filter(full, sigma=1.0).astype(np.float32)


# ── Base terrain generation ───────────────────────────────────────────────────

def _make_synthetic_base() -> np.ndarray:
    """Pure procedural base, calibrated to Felsted spot heights."""
    grid = np.full((GRID_SIZE, GRID_SIZE), CAMPUS_ELEV, dtype=np.float32)

    # Large-scale rolling Essex landscape
    xs = np.linspace(0, 1, GRID_SIZE, dtype=np.float32)
    ys = np.linspace(0, 1, GRID_SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    import math
    grid += 7.0  * np.sin(xx * math.pi * 2.2 + 0.4) * np.cos(yy * math.pi * 1.8 + 0.6)
    grid += 4.5  * np.cos(xx * math.pi * 3.9 + 1.0) * np.sin(yy * math.pi * 3.3 + 0.3)
    grid += 2.0  * np.sin(xx * math.pi * 7.5 + 2.1) * np.cos(yy * math.pi * 5.8 + 1.7)
    grid += 0.8  * np.cos(xx * math.pi * 13.1)       * np.sin(yy * math.pi * 10.4 + 0.5)

    # River valley (Chelmer / Stebbing Brook) through southern quarter
    river_row = int(0.74 * GRID_SIZE)
    r_arr     = np.arange(GRID_SIZE, dtype=np.float32)[:, None]
    sigma_px  = 80.0 / SQUARE_SIZE
    grid -= (CAMPUS_ELEV - VALLEY_ELEV) * np.exp(-((r_arr - river_row)**2) / (2 * sigma_px**2))

    # North-west ridge
    r, c = world_to_hm(-380, 620)
    _gaussian_bump(grid, r, c, 450, HILL_N_ELEV - CAMPUS_ELEV)
    r, c = world_to_hm(430, 700)
    _gaussian_bump(grid, r, c, 300, (HILL_N_ELEV - 5) - CAMPUS_ELEV)

    return grid


def _apply_campus_blend(grid: np.ndarray) -> None:
    """Blend campus plateau toward ~76 m, in-place."""
    r_cam, c_cam = world_to_hm(0, 0)
    rr = np.arange(GRID_SIZE, dtype=np.float32)[:, None]
    cc = np.arange(GRID_SIZE, dtype=np.float32)[None, :]
    sigma = 300.0 / SQUARE_SIZE
    blend = np.exp(-((rr - r_cam)**2 + (cc - c_cam)**2) / (2 * sigma**2))
    np.copyto(grid, grid * (1 - blend * 0.88) + CAMPUS_ELEV * (blend * 0.88))


def _apply_road_cuts(grid: np.ndarray) -> None:
    """Grade-cut the terrain along key road alignments."""
    # Stebbing Road: 62 m (south) → 72 m (campus gate) → 80 m (north)
    _road_cut(grid, -260, -1024, -260,  -200,  62.0, 72.0)
    _road_cut(grid, -260,  -200, -240,   800,  72.0, 80.0)
    # Entrance drive: 72 m → 76 m
    _road_cut(grid, -260, -200,    0,   -50,   72.0, 75.5)
    # Braintree Road approach: roughly constant ~75 m
    _road_cut(grid,  800,  -50,   -170, -188,  75.0, 73.5)


def build_elevation(online: bool = False, lidar: bool = False) -> np.ndarray:
    """
    Return float32 (GRID_SIZE, GRID_SIZE) elevation array in metres.

    Strategy:
      1. LiDAR DTM 2 m base (cached grid, cached GeoTIFF, or live WCS
         download when ``online``/``lidar`` is set).
      2. Cached SRTM grid if present.
      3. Live SRTM if ``--online`` requested.
      4. Synthetic procedural model.

    LiDAR bases skip the synthetic road cuts and level flattening.
    """
    base: np.ndarray | None = None
    source = "synthetic"

    # 1. LiDAR (best available data)
    try:
        from tools.lidar import build_lidar_base
        base = build_lidar_base(online=online or lidar)
    except Exception as exc:                       # pragma: no cover - defensive
        log.warning("LiDAR module unavailable: %s", exc)
        base = None

    if base is not None:
        source = "lidar"

    # 2. SRTM cache
    if base is None:
        base = _load_srtm_cache()
        if base is not None:
            source = "srtm-cache"

    # 3. Live SRTM
    if base is None and online:
        log.info("Fetching SRTM elevation online …")
        try:
            base = _fetch_srtm_live()
            source = "srtm-live"
            log.info("SRTM fetch OK, range %.1f–%.1f m", base.min(), base.max())
        except Exception as exc:
            log.warning("SRTM online fetch failed: %s", exc)

    # 4. Synthetic fallback
    if base is None:
        log.info("Generating synthetic base terrain …")
        base = _make_synthetic_base()
        source = "synthetic"

    if source == "lidar":
        # Real survey data: add only micro-detail, no synthetic shaping.
        noise = _fbm(GRID_SIZE, seed=42, octaves=9, base_coarse=4) * 0.6 - 0.3
        grid  = (base + noise.astype(np.float32)).astype(np.float32)
        grid  = gaussian_filter(grid, sigma=0.5).astype(np.float32)
    else:
        # Coarse/interpolated sources need detail noise and synthetic shaping.
        noise = _fbm(GRID_SIZE, seed=42, octaves=9, base_coarse=4) * 4.0 - 2.0
        grid  = (base + noise.astype(np.float32)).astype(np.float32)

        _apply_campus_blend(grid)
        _apply_road_cuts(grid)

        # Light smoothing to remove any seams
        grid = gaussian_filter(grid, sigma=0.8).astype(np.float32)

        # Re-apply level flatten after final smooth
        _apply_campus_blend(grid)

    log.info("Elevation ready (%s): range %.1f–%.1f m",
             source, grid.min(), grid.max())
    return grid


def elev_at_world(grid: np.ndarray, wx: float, wy: float) -> float:
    """Bilinear sample of the elevation grid at world coordinates."""
    cf = (wx + WORLD_HALF) / SQUARE_SIZE
    rf = (WORLD_HALF - wy)  / SQUARE_SIZE
    c0 = max(0, min(BLOCK_SIZE - 1, int(cf)))
    r0 = max(0, min(BLOCK_SIZE - 1, int(rf)))
    c1, r1 = min(BLOCK_SIZE, c0 + 1), min(BLOCK_SIZE, r0 + 1)
    tc, tr = cf - c0, rf - r0
    return float(
        grid[r0, c0] * (1-tr) * (1-tc) +
        grid[r0, c1] * (1-tr) *    tc  +
        grid[r1, c0] *    tr  * (1-tc) +
        grid[r1, c1] *    tr  *    tc
    )
