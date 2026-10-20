"""
Terrain material definitions + textures for the classic v1 terrain path.

The `.ter` file stores only terrain material *names* — the layer map indexes
into that list. The matching `TerrainMaterial` objects live in
`levels/<map>/art/terrains/main.materials.json`; without that file BeamNG
resolves every layer to the "NO TEXTURE" warning material.

`TerrainBlock.materialTextureSet` is intentionally left empty, which selects
the classic v1 path: each material is a single `diffuseMap` (plus optional
normal/detail/macro maps). This module writes the definition file and
generates a simple tileable base texture per material so every map renders
with sensible ground colours out of the box.

The schema follows MapNG's classic-v1 export:

    { "class": "TerrainMaterial",
      "internalName": "<name in the .ter>",
      "diffuseMap": "/levels/<map>/art/terrains/<name>_diffuse.png",
      "diffuseSize": <metres spanned by one texture tile>,
      "groundmodelName": "GRASS" }

`internalName` must match the name stored in the `.ter` exactly, and
`diffuseSize` is a world distance in metres (not pixels).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# material-name keyword -> (base RGB, noise contrast, groundmodel, annotation)
_LOOKS = {
    "grass":   ((74, 112, 48), 34, "GRASS",   "GRASS"),
    "dirt":    ((98, 78, 54), 30, "DIRT",    "DIRT"),
    "asphalt": ((62, 62, 64), 16, "ASPHALT", "ASPHALT"),
    "sand":    ((176, 158, 108), 22, "SAND",  "SAND"),
    "rock":    ((104, 100, 96), 26, "ROCK",   "ROCK"),
}
_DEFAULT_LOOK = _LOOKS["grass"]


def _look_for(name: str):
    """Pick a ground look from the material name (case-insensitive)."""
    n = name.lower()
    for key, look in _LOOKS.items():
        if key in n:
            return look
    return _DEFAULT_LOOK


def _tileable_value_noise(size: int, cells: int, rng) -> np.ndarray:
    """Smooth value noise on a `cells`x`cells` lattice, wrapping at the edges."""
    lat = rng.random((cells, cells)).astype(np.float32)
    t = np.linspace(0.0, cells, size, endpoint=False, dtype=np.float32)
    i0 = np.floor(t).astype(np.int64) % cells
    i1 = (i0 + 1) % cells
    f = t - np.floor(t)
    f = f * f * (3.0 - 2.0 * f)            # smoothstep
    v00 = lat[np.ix_(i0, i0)]
    v01 = lat[np.ix_(i0, i1)]
    v10 = lat[np.ix_(i1, i0)]
    v11 = lat[np.ix_(i1, i1)]
    fr = f[:, None]
    fc = f[None, :]
    return (v00 * (1 - fr) * (1 - fc) + v01 * (1 - fr) * fc +
            v10 * fr * (1 - fc) + v11 * fr * fc)


def _noise_field(size: int, rng, octaves: int = 3, base_cells: int = 4) -> np.ndarray:
    """Multi-octave tileable noise in [0, 1]."""
    total = np.zeros((size, size), dtype=np.float32)
    amp, norm, cells = 1.0, 0.0, base_cells
    for _ in range(octaves):
        total += amp * _tileable_value_noise(size, cells, rng)
        norm += amp
        amp *= 0.5
        cells *= 2
    return total / norm


def _texture_for(name: str, size: int, rng) -> Image.Image:
    """A tileable RGB texture for one material."""
    rgb, contrast, _, _ = _look_for(name)
    delta = (_noise_field(size, rng) - 0.5) * 2.0 * contrast
    base = np.array(rgb, dtype=np.float32)[None, None, :]
    out = np.clip(base + delta[:, :, None], 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def write_terrain_materials(levels_dir: Path | str,
                            map_name: str,
                            materials: list[str],
                            tex_size: int = 512,
                            tile_meters: float = 64.0) -> dict:
    """
    Write `art/terrains/main.materials.json` plus one diffuse texture per
    material, and return the material definition dict that was written.

    Parameters
    ----------
    levels_dir  : the `levels/` directory containing `<map>/`
    map_name    : level folder name (used in the texture paths)
    materials   : terrain material internal names, in layer-map order
    tex_size    : generated texture resolution (power of two)
    tile_meters : world metres spanned by one texture tile
    """
    art = Path(levels_dir) / "art" / "terrains"
    art.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20240517)
    defs: dict[str, dict] = {}
    for name in materials:
        tex_name = f"{name}_diffuse.png"
        _texture_for(name, tex_size, rng).save(art / tex_name)
        _, _, ground, annotation = _look_for(name)
        defs[name] = {
            "class": "TerrainMaterial",
            "internalName": name,
            "diffuseMap": f"/levels/{map_name}/art/terrains/{tex_name}",
            "diffuseSize": float(tile_meters),
            "groundmodelName": ground,
            "annotation": annotation,
            "useSideProjection": False,
        }

    out = art / "main.materials.json"
    out.write_text(json.dumps(defs, indent=2) + "\n", encoding="utf-8")
    log.info("Terrain materials: %s (%d material[s])", out, len(defs))
    return defs
