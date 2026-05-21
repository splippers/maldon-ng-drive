"""
Shared geographic constants for any ng-drive map.

Reads map identity from map_config.py in the repo root, so the same
tools/ directory works for every map in the ng-drive network by just
changing map_config.py.

Network maps:
  maldon-ng-drive    (this repo)  — Maldon East terminus, GER Maldon branch
  witham-ng-drive                 — Witham junction (Maldon ↔ Dunmow branches)
  cressing-ng-drive               — Cressing / White Notley section
  braintree-ng-drive              — Braintree junction
  rayne-ng-drive                  — Rayne station
  felsted-ng-drive                — Felsted School, GER Dunmow branch
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

# ── Locate and load map_config.py ─────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    _cfg = importlib.import_module("map_config")
except ModuleNotFoundError:
    raise RuntimeError(
        "No map_config.py found in repo root.  "
        "Each ng-drive repo must have a map_config.py.  "
        "See maldon-ng-drive/map_config.py for a template."
    )

# ── Identity ───────────────────────────────────────────────────────────────────
MAP_NAME    = _cfg.MAP_NAME
MAP_TITLE   = _cfg.MAP_TITLE
MAP_VERSION = _cfg.MAP_VERSION
MAP_DESC    = _cfg.MAP_DESC
MAP_AUTHOR  = getattr(_cfg, "MAP_AUTHOR", "ng-drive-network")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = _ROOT
DATA_DIR   = ROOT / "data"
LEVELS_DIR = ROOT / "levels" / MAP_NAME
TERRAIN_DIR = LEVELS_DIR / "terrainGrid"

OSM_CACHE  = DATA_DIR / f"{MAP_NAME}_osm.json"
SRTM_CACHE = DATA_DIR / f"{MAP_NAME}_srtm_32x32.json"

# ── GPS anchor ────────────────────────────────────────────────────────────────
CENTER_LAT = _cfg.CENTER_LAT
CENTER_LON = _cfg.CENTER_LON
BBOX       = _cfg.BBOX   # (S, W, N, E)

# ── Projection ────────────────────────────────────────────────────────────────
_LAT_SCALE = 111_139.0
_LON_SCALE = 111_139.0 * math.cos(math.radians(CENTER_LAT))


def gps_to_world(lat: float, lon: float) -> tuple[float, float]:
    """GPS → BeamNG world (X east, Y north) in metres from map centre."""
    return (lon - CENTER_LON) * _LON_SCALE, (lat - CENTER_LAT) * _LAT_SCALE


def world_to_gps(x: float, y: float) -> tuple[float, float]:
    """BeamNG world (X east, Y north) → (lat, lon)."""
    return CENTER_LAT + y / _LAT_SCALE, CENTER_LON + x / _LON_SCALE


# ── Terrain grid ──────────────────────────────────────────────────────────────
BLOCK_SIZE  = getattr(_cfg, "BLOCK_SIZE",  1024)
SQUARE_SIZE = getattr(_cfg, "SQUARE_SIZE", 2.0)
GRID_SIZE   = BLOCK_SIZE + 1
WORLD_HALF  = BLOCK_SIZE * SQUARE_SIZE / 2   # ±1024 m by default

MAX_TERRAIN_HEIGHT = 200.0

# ── Key elevation anchors ─────────────────────────────────────────────────────
CAMPUS_ELEV = getattr(_cfg, "CAMPUS_ELEV", 76.0)   # main level of interest
VALLEY_ELEV = getattr(_cfg, "VALLEY_ELEV", 40.0)   # water / low ground
HILL_N_ELEV = getattr(_cfg, "HILL_N_ELEV", 84.0)   # local high point


# ── World ↔ heightmap helpers ─────────────────────────────────────────────────

def world_to_hm(wx: float, wy: float) -> tuple[int, int]:
    """World (X east, Y north) → heightmap (row, col). Row 0 = north edge."""
    col = int((wx + WORLD_HALF) / SQUARE_SIZE)
    row = int((WORLD_HALF - wy) / SQUARE_SIZE)
    return max(0, min(BLOCK_SIZE, row)), max(0, min(BLOCK_SIZE, col))


def hm_to_world(row: int, col: int) -> tuple[float, float]:
    wx = col * SQUARE_SIZE - WORLD_HALF + SQUARE_SIZE / 2
    wy = WORLD_HALF - row * SQUARE_SIZE - SQUARE_SIZE / 2
    return wx, wy
