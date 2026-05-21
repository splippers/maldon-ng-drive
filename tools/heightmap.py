"""
Terrain heightmap generator — v3.0.

Priority order for elevation source:
  1. Cached SRTM-30m grid  (data/felsted_srtm_32x32.json)
  2. Live OpenTopoData fetch (--online)
  3. Synthetic procedural model tuned to known spot heights

In all cases the final grid is:
  • 1025 × 1025 float32 array (BLOCK_SIZE=1024, 2 m/cell → 2 048 m world)
  • Blended with fBm fine-detail noise
  • Road grade cuts applied along Stebbing Road
  • Campus plateau flattened to ~76 m ASL
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from tools.constants import (
    BLOCK_SIZE, GRID_SIZE, SQUARE_SIZE, WORLD_HALF,
    CAMPUS_ELEV, VALLEY_ELEV, HILL_N_ELEV,
    CENTER_LAT, CENTER_LON, BBOX,
    SRTM_CACHE, world_to_hm,
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


# ── SRTM / elevation sources ──────────────────────────────────────────────────

def _load_srtm_cache() -> np.ndarray | None:
    """Load cached 32×32 SRTM grid and zoom to GRID_SIZE. Returns None if missing."""
    if not SRTM_CACHE.exists():
        return None
    d    = json.loads(SRTM_CACHE.read_text())
    grid = np.array(d["elevations"], dtype=np.float32)   # shape (32, 32) S→N
    # Image convention: row 0 = north
    grid = np.flipud(grid)
    # Zoom to GRID_SIZE
    factor = GRID_SIZE / grid.shape[0]
    full   = zoom(grid, factor, order=3)[:GRID_SIZE, :GRID_SIZE]
    full   = gaussian_filter(full, sigma=3.0)
    log.info("SRTM cache loaded, range %.1f–%.1f m", full.min(), full.max())
    return full.astype(np.float32)


def _fetch_srtm_live(coarse: int = 32) -> np.ndarray:
    """Download SRTM grid, zoom to GRID_SIZE."""
    import time
    s, w, n, e = BBOX
    lats = np.linspace(s, n, coarse).tolist()
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

    raw = np.array(elevs[:coarse*coarse], dtype=np.float32).reshape(coarse, coarse)
    raw = np.flipud(raw)
    full = zoom(raw, GRID_SIZE / coarse, order=3)[:GRID_SIZE, :GRID_SIZE]
    return gaussian_filter(full, sigma=3.0).astype(np.float32)


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


def build_elevation(online: bool = False) -> np.ndarray:
    """
    Return float32 (GRID_SIZE, GRID_SIZE) elevation array in metres.

    Strategy:
      1. Use cached SRTM grid if present.
      2. Fetch live SRTM if --online requested.
      3. Fall back to synthetic procedural model.
    """
    base: np.ndarray | None = None

    # Try cache first
    base = _load_srtm_cache()

    if base is None and online:
        log.info("Fetching SRTM elevation online …")
        try:
            base = _fetch_srtm_live()
            log.info("SRTM fetch OK, range %.1f–%.1f m", base.min(), base.max())
        except Exception as exc:
            log.warning("SRTM online fetch failed: %s", exc)

    if base is None:
        log.info("Generating synthetic base terrain …")
        base = _make_synthetic_base()

    # Add fBm fine-detail noise (amplitude ±2 m, 9 octaves)
    noise = _fbm(GRID_SIZE, seed=42, octaves=9, base_coarse=4) * 4.0 - 2.0
    grid  = (base + noise.astype(np.float32)).astype(np.float32)

    _apply_campus_blend(grid)
    _apply_road_cuts(grid)

    # Light smoothing to remove any seams
    grid = gaussian_filter(grid, sigma=0.8).astype(np.float32)

    # Re-apply campus flatten after final smooth
    _apply_campus_blend(grid)

    log.info("Elevation ready: range %.1f–%.1f m", grid.min(), grid.max())
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
