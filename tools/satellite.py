"""
Satellite imagery downloader and compositor — v3.0.

Downloads ESRI WorldImagery tiles (free, no API key) and composites them
into a single PNG for use as the BeamNG terrain base texture.

Output: levels/felsted/art/terrain/satellite/satellite.png  (4096×4096 RGB)
The material is configured with detailScale=2048 so the image tiles exactly
once over the 2048 m × 2048 m world — giving true aerial-photo ground texture.
"""

from __future__ import annotations

import io
import logging
import math
import struct
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Optional

import numpy as np

from tools.constants import (
    CENTER_LAT, CENTER_LON, WORLD_HALF, DATA_DIR, LEVELS_DIR,
    world_to_gps,
)

log = logging.getLogger(__name__)

_TILE_CACHE = DATA_DIR / "tiles" / "esri"
_SAT_DIR    = LEVELS_DIR / "art" / "terrain" / "satellite"
_SAT_PNG    = _SAT_DIR / "satellite.png"
_TILE_PX    = 256

_ESRI_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services"
    "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


# ── GPS ↔ tile math ───────────────────────────────────────────────────────────

def _deg2tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    tx = int((lon + 180.0) / 360.0 * n)
    lat_r = math.radians(lat)
    ty = int((1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n)
    return tx, ty


def _tile_lon_left(tx: int, zoom: int) -> float:
    return tx / (2 ** zoom) * 360.0 - 180.0


def _tile_merc_top(ty: int, zoom: int) -> float:
    """Mercator Y (asinh(tan(lat))) for the top edge of tile ty."""
    return math.pi * (1.0 - 2.0 * ty / 2 ** zoom)


def _world_bbox() -> tuple[float, float, float, float]:
    """GPS bbox of the 2048 m × 2048 m world: (lat_s, lon_w, lat_n, lon_e)."""
    lat_n, lon_w = world_to_gps(-WORLD_HALF,  WORLD_HALF)
    lat_s, lon_e = world_to_gps( WORLD_HALF, -WORLD_HALF)
    return lat_s, lon_w, lat_n, lon_e


# ── Tile download ─────────────────────────────────────────────────────────────

def _download_tile(tx: int, ty: int, zoom: int) -> Optional[np.ndarray]:
    """Return (256, 256, 3) uint8 RGB tile, or None on failure."""
    cache = _TILE_CACHE / str(zoom) / str(ty) / f"{tx}.jpg"
    if cache.exists():
        data = cache.read_bytes()
    else:
        url = _ESRI_URL.format(z=zoom, y=ty, x=tx)
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "felsted-ng-drive/3.0"}
                )
                with urllib.request.urlopen(req, timeout=20) as r:
                    data = r.read()
                time.sleep(0.08)
                break
            except Exception as exc:
                log.warning("Tile %d/%d/%d attempt %d: %s", zoom, ty, tx, attempt, exc)
                time.sleep(1.5 * (attempt + 1))
        else:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        log.warning("Decode failed %d/%d/%d: %s", zoom, ty, tx, exc)
        return None


# ── Canvas crop helpers ───────────────────────────────────────────────────────

def _merc_y(lat_deg: float) -> float:
    return math.asinh(math.tan(math.radians(lat_deg)))


def _gps_to_canvas_px(
    lat: float, lon: float,
    tx_min: int, ty_min: int, tx_max: int, ty_max: int,
    zoom: int,
    canvas_h: int, canvas_w: int,
) -> tuple[int, int]:
    """Map a GPS point to pixel (row, col) within the stitched tile canvas."""
    lon_left  = _tile_lon_left(tx_min, zoom)
    lon_right = _tile_lon_left(tx_max + 1, zoom)
    col = int((lon - lon_left) / (lon_right - lon_left) * canvas_w)

    merc_top = _tile_merc_top(ty_min, zoom)
    merc_bot = _tile_merc_top(ty_max + 1, zoom)
    row = int((merc_top - _merc_y(lat)) / (merc_top - merc_bot) * canvas_h)

    return (
        max(0, min(canvas_h - 1, row)),
        max(0, min(canvas_w - 1, col)),
    )


# ── PNG writer ────────────────────────────────────────────────────────────────

def _write_png_rgb(arr: np.ndarray, path: Path) -> None:
    """Write uint8 (H, W, 3) array as PNG using stdlib only."""
    h, w, _ = arr.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    raw  = b"".join(b"\x00" + row.tobytes() for row in arr)
    idat = chunk(b"IDAT", zlib.compress(raw, 6))
    iend = chunk(b"IEND", b"")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend)


# ── Main entry ────────────────────────────────────────────────────────────────

def build_satellite_texture(
    zoom: int = 17,
    out_size: int = 4096,
    force: bool = False,
) -> Path:
    """
    Download ESRI WorldImagery tiles for the map area and composite into a
    single satellite.png.  Returns path to the output file.

    zoom=17 → ~0.74 m/px at 51.86°N (good enough for BeamNG driving view)
    zoom=18 → ~0.37 m/px            (higher quality, ~4× more tile downloads)
    """
    if _SAT_PNG.exists() and not force:
        log.info("Satellite texture already cached: %s", _SAT_PNG)
        return _SAT_PNG

    lat_s, lon_w, lat_n, lon_e = _world_bbox()
    log.info("World GPS bbox: %.4f–%.4f N, %.4f–%.4f E", lat_s, lat_n, lon_w, lon_e)

    tx_nw, ty_nw = _deg2tile(lat_n, lon_w, zoom)   # NW → smallest tx, ty
    tx_se, ty_se = _deg2tile(lat_s, lon_e, zoom)   # SE → largest  tx, ty

    tx_min, tx_max = min(tx_nw, tx_se), max(tx_nw, tx_se)
    ty_min, ty_max = min(ty_nw, ty_se), max(ty_nw, ty_se)

    n_x = tx_max - tx_min + 1
    n_y = ty_max - ty_min + 1
    log.info(
        "Tile grid: %d×%d = %d tiles at zoom %d", n_x, n_y, n_x * n_y, zoom
    )

    canvas_h = n_y * _TILE_PX
    canvas_w = n_x * _TILE_PX
    canvas = np.full((canvas_h, canvas_w, 3), 64, dtype=np.uint8)

    downloaded = 0
    total = n_x * n_y
    for iy, ty in enumerate(range(ty_min, ty_max + 1)):
        for ix, tx in enumerate(range(tx_min, tx_max + 1)):
            tile = _download_tile(tx, ty, zoom)
            if tile is not None:
                r0 = iy * _TILE_PX
                c0 = ix * _TILE_PX
                canvas[r0:r0 + _TILE_PX, c0:c0 + _TILE_PX] = tile[:_TILE_PX, :_TILE_PX]
                downloaded += 1
        if iy % 5 == 0:
            log.info("  Row %d/%d, tiles downloaded: %d/%d", iy + 1, n_y, downloaded, total)

    log.info("Downloaded %d/%d tiles", downloaded, total)

    # Crop canvas to exactly the world GPS extent
    row_n, col_w = _gps_to_canvas_px(
        lat_n, lon_w, tx_min, ty_min, tx_max, ty_max, zoom, canvas_h, canvas_w
    )
    row_s, col_e = _gps_to_canvas_px(
        lat_s, lon_e, tx_min, ty_min, tx_max, ty_max, zoom, canvas_h, canvas_w
    )
    row_n, row_s = min(row_n, row_s), max(row_n, row_s)
    col_w, col_e = min(col_w, col_e), max(col_w, col_e)
    cropped = canvas[row_n:row_s + 1, col_w:col_e + 1]
    log.info("Cropped to %d×%d px", cropped.shape[1], cropped.shape[0])

    # Resize to out_size × out_size
    from scipy.ndimage import zoom as spzoom
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        log.error("Crop is too small — check tile download and GPS constants")
        resized = np.full((out_size, out_size, 3), 80, dtype=np.uint8)
    else:
        fh = out_size / cropped.shape[0]
        fw = out_size / cropped.shape[1]
        channels = [
            np.clip(spzoom(cropped[:, :, c].astype(np.float32), (fh, fw), order=1), 0, 255)
            for c in range(3)
        ]
        resized = np.stack(channels, axis=2).astype(np.uint8)

    _write_png_rgb(resized, _SAT_PNG)
    log.info("Satellite texture → %s  (%d×%d)", _SAT_PNG, out_size, out_size)
    return _SAT_PNG


def write_satellite_material() -> Path:
    """Write the BeamNG terrain material JSON for the satellite layer."""
    mat_path = _SAT_DIR / "materials.json"
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    mat_path.write_text(
        '{\n'
        '  "felsted_satellite": {\n'
        '    "class": "TerrainMaterial",\n'
        '    "diffuseMap":  "levels/felsted/art/terrain/satellite/satellite.png",\n'
        '    "normalMap":   "levels/felsted/art/terrain/satellite/flat_n.png",\n'
        '    "detailScale": 2048.0,\n'
        '    "detailStrength": 0.0,\n'
        '    "useSideProjection": false,\n'
        '    "groundModel": "asphalt"\n'
        '  }\n'
        '}\n'
    )

    # Flat normal map (128, 128, 255) — 4×4 pixels, tiled; keeps normals neutral
    normal_path = _SAT_DIR / "flat_n.png"
    if not normal_path.exists():
        flat = np.zeros((4, 4, 3), dtype=np.uint8)
        flat[:, :, 0] = 128
        flat[:, :, 1] = 128
        flat[:, :, 2] = 255
        _write_png_rgb(flat, normal_path)

    return mat_path
