"""
Write Torque3D / BeamNG terrain binary files (.ter).

Format: version 2 of the Torque3D TerrainFile format.
Byte layout (little-endian):

  [0  –  3]  Magic        "TERR"
  [4  –  7]  uint32       version = 2
  [8  – 11]  uint32       blockSize   (cells per side; vertex grid = blockSize+1)
  [12 – 15]  uint32       numLayers
  [16 – …]   uint16×(blockSize+1)²  height values (0 = 0 m, 65535 = maxHeight m)
  […  – …]   uint8 ×(blockSize+1)²  per-vertex flags (all 0)
  For each layer:
    NUL-terminated UTF-8  internalName
    NUL-terminated UTF-8  texturePath
    uint8×(blockSize+1)²  coverage weights (0–255)

If the binary proves incompatible with the installed BeamNG version, fall back
to the PNG import workflow: open World Editor → Terrain → Import Heightmap and
point it at `terrainGrid/felsted_height.png`.
"""

from __future__ import annotations

import struct
import logging
from pathlib import Path

import numpy as np

from tools.constants import (
    BLOCK_SIZE, GRID_SIZE, MAX_TERRAIN_HEIGHT, SQUARE_SIZE,
)

log = logging.getLogger(__name__)

_MAGIC   = b"TERR"
_VERSION = 2

TERRAIN_LAYER_NAME     = "felsted_grass"
TERRAIN_LAYER_TEX      = "levels/felsted/art/terrains/felsted_grass/diffuse.png"
SATELLITE_LAYER_NAME   = "felsted_satellite"
SATELLITE_LAYER_TEX    = "levels/felsted/art/terrain/satellite/satellite.png"


def _norm_to_uint16(elevation: np.ndarray, max_h: float) -> np.ndarray:
    n = np.clip(elevation / max_h, 0.0, 1.0)
    return (n * 65535.0).astype(np.uint16)


def write_ter(path: Path | str,
              elevation: np.ndarray,
              max_height: float = MAX_TERRAIN_HEIGHT,
              use_satellite: bool = False) -> None:
    """
    Write a .ter file from an elevation array.

    Parameters
    ----------
    path       : destination file path
    elevation  : float32 array of shape (GRID_SIZE, GRID_SIZE) in metres.
                 Row 0 = north; col 0 = west (image convention).
    max_height : metres corresponding to uint16 value 65535.
    """
    path = Path(path)
    h, w = elevation.shape
    if h != GRID_SIZE or w != GRID_SIZE:
        raise ValueError(f"elevation must be ({GRID_SIZE}, {GRID_SIZE}), got ({h}, {w})")

    heights  = _norm_to_uint16(elevation, max_height)
    flags    = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    coverage = np.full((GRID_SIZE, GRID_SIZE), 255, dtype=np.uint8)

    layer_name = SATELLITE_LAYER_NAME if use_satellite else TERRAIN_LAYER_NAME
    layer_tex  = SATELLITE_LAYER_TEX  if use_satellite else TERRAIN_LAYER_TEX

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<I", _VERSION))
        f.write(struct.pack("<I", BLOCK_SIZE))
        f.write(struct.pack("<I", 1))          # numLayers

        f.write(heights.tobytes())
        f.write(flags.tobytes())

        f.write(layer_name.encode() + b"\x00")
        f.write(layer_tex.encode()  + b"\x00")
        f.write(coverage.tobytes())

    size_mb = path.stat().st_size / 1_048_576
    log.info("Wrote %s (%.1f MB)", path, size_mb)


def write_heightmap_png(path: Path | str,
                        elevation: np.ndarray,
                        max_height: float = MAX_TERRAIN_HEIGHT) -> None:
    """
    Save a 16-bit greyscale PNG heightmap using stdlib (no Pillow quirks).

    This is the fallback for manual import via the BeamNG World Editor:
      World Editor → Terrain → Import Heightmap  (max height = 200 m).
    """
    import struct, zlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    u16 = _norm_to_uint16(elevation, max_height)
    h, w = u16.shape

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFF_FFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)   # 16-bit grayscale

    # Filter byte 0 (None) prepended to each row; big-endian uint16
    raw = b"".join(b"\x00" + u16[r].astype(">u2").tobytes() for r in range(h))

    with path.open("wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(_chunk(b"IEND", b""))

    log.info("Wrote heightmap PNG %s", path)


def write_preview_png(path: Path | str, elevation: np.ndarray) -> None:
    """8-bit colourised preview of the terrain for info.json thumbnail."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lo, hi = elevation.min(), elevation.max()
    norm = ((elevation - lo) / max(hi - lo, 1.0) * 255).astype(np.uint8)

    # Colour map: blue (water) → green (grass) → grey (hill)
    r = np.clip(norm.astype(np.int16) - 60, 0, 255).astype(np.uint8)
    g = np.clip(200 - abs(norm.astype(np.int16) - 130), 80, 220).astype(np.uint8)
    b = np.where(norm < 100, np.clip(200 - norm, 0, 255), 80).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)

    img = Image.fromarray(rgb, mode="RGB").resize((512, 512), Image.BILINEAR)
    img.save(str(path))
    log.info("Wrote preview PNG %s", path)
