"""
Write BeamNG terrain binary files (.ter).

Format: BeamNG TerrainFile v9 (the current on-disk format; see
https://documentation.beamng.com/modding/levels/level_formats/terrain/).
Byte layout (little-endian):

  [0]       uint8        version = 9
  [1 – 4]   uint32       size (power-of-two samples per side; world width =
                         size * squareSize, squareSize set on TerrainBlock)
  [5 – …]   uint16×size²  heightmap (row 0 = south, col 0 = west)
  […  – …]   uint8 ×size²  layermap (material index per sample; row 0 = south;
                           255 = hole/empty terrain)
  […  – …]   uint32        materialCount
  for each material: uint8 length + UTF-8 internal name (length-prefixed)

The input elevation grid is (BLOCK_SIZE+1)² = GRID_SIZE² with row 0 = north
(image convention). The v9 file stores BLOCK_SIZE² samples with row 0 = south,
so the far-north row and far-east column of the input grid are dropped and the
grid is flipped south-to-north. This matches the TerrainBlock placement at
[-WORLD_HALF, -WORLD_HALF, 0]: sample (r, c) sits at world
(-WORLD_HALF + c*squareSize, -WORLD_HALF + r*squareSize).
"""

from __future__ import annotations

import json
import struct
import logging
from pathlib import Path

import numpy as np

from tools.constants import (
    BLOCK_SIZE, GRID_SIZE, MAX_TERRAIN_HEIGHT, SQUARE_SIZE, WORLD_HALF, MAP_NAME,
)

log = logging.getLogger(__name__)

TERRAIN_LAYER_NAME     = f"{MAP_NAME}_grass"
TERRAIN_LAYER_TEX      = f"levels/{MAP_NAME}/art/terrains/{MAP_NAME}_grass/diffuse.png"
SATELLITE_LAYER_NAME   = f"{MAP_NAME}_satellite"
SATELLITE_LAYER_TEX    = f"levels/{MAP_NAME}/art/terrain/satellite/satellite.png"


def _norm_to_uint16(elevation: np.ndarray, max_h: float) -> np.ndarray:
    n = np.clip(elevation / max_h, 0.0, 1.0)
    return (n * 65535.0).astype(np.uint16)


def write_ter(path: Path | str,
              elevation: np.ndarray,
              max_height: float = MAX_TERRAIN_HEIGHT,
              use_satellite: bool = False) -> None:
    """
    Write a BeamNG TerrainFile v9 .ter from an elevation array.

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

    size = BLOCK_SIZE  # v9 stores size² samples, not (size+1)²; world width
                       # = size * squareSize (squareSize on TerrainBlock).

    # Input row 0 = north (y = +WORLD_HALF); v9 row 0 = south (y = -WORLD_HALF).
    # Drop input row 0 (north edge) and the last column (east edge) so the
    # sample lattice lands exactly on world -WORLD_HALF + i*SQUARE_SIZE,
    # matching the TerrainBlock's squareSize and SW-corner position.
    grid = elevation[1:][::-1, :size]                     # rows 1024..1, cols 0..1023

    heights   = _norm_to_uint16(grid, max_height).astype("<u2")
    layermap  = np.zeros((size, size), dtype=np.uint8)    # material 0 everywhere

    material_name = SATELLITE_LAYER_NAME if use_satellite else TERRAIN_LAYER_NAME

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("B", 9))                      # version 9
        f.write(struct.pack("<I", size))                  # grid size (power of two)
        f.write(heights.tobytes())                        # heightMap, row 0 = south
        f.write(layermap.tobytes())                       # layerMap, row 0 = south
        f.write(struct.pack("<I", 1))                     # materialCount = 1
        name_bytes = material_name.encode("utf-8")
        f.write(struct.pack("B", len(name_bytes)))        # length-prefixed string
        f.write(name_bytes)

    size_mb = path.stat().st_size / 1_048_576
    log.info("Wrote %s (%.1f MB)", path, size_mb)


def write_terrain_json(path: Path | str,
                       datafile: str,
                       size: int,
                       materials: list[str],
                       heightmap_image: str | None = None) -> dict:
    """
    Write the companion `.terrain.json` metadata file.

    This file is descriptive only (the engine loads terrain data from the
    `.ter`) but the engine writes one on save and tools expect it, so we
    mirror the documented minimal layout.

    Parameters
    ----------
    path            : destination file path
    datafile        : virtual path of the `.ter` this describes
    size            : terrain grid size (samples per side)
    materials       : terrain material internal names, in layer-map order
    heightmap_image : optional virtual path of the exported heightmap PNG
    """
    n = size * size
    meta = {
        "version": 9,
        "datafile": datafile,
        "size": size,
        "binaryFormat": ("version(char), size(unsigned int), "
                         "heightMap(heightMapSize * heightMapItemSize), "
                         "layerMap(layerMapSize * layerMapItemSize), "
                         "materialNames"),
        "heightMapSize": n,
        "heightMapItemSize": 2,
        "layerMapSize": n,
        "layerMapItemSize": 1,
        "materials": list(materials),
    }
    if heightmap_image:
        meta["heightmapImage"] = heightmap_image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s", path)
    return meta


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
