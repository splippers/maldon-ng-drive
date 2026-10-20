"""
Tests for the TerrainBlock the level builder emits.

Map-agnostic: everything is driven by tools/constants.py (fed by
map_config.py), so the same assertions hold for every map in the network.
"""

from __future__ import annotations

from tools.constants import (
    MAP_NAME, BLOCK_SIZE, SQUARE_SIZE, WORLD_HALF,
)
from tools import level_builder as lb


def _terrain():
    return lb._terrain_block()


def test_is_a_terrain_block():
    t = _terrain()
    assert t["class"] == "TerrainBlock"


def test_terrain_file_points_at_map_ter():
    t = _terrain()
    assert t["terrainFile"] == f"levels/{MAP_NAME}/terrainGrid/{MAP_NAME}.ter"


def test_square_size_is_metres_per_sample():
    t = _terrain()
    assert t["squareSize"] == SQUARE_SIZE


def test_max_height_matches_terrain_encoder():
    """The engine scales stored heights by maxHeight/65536, so the
    TerrainBlock must declare the height the .ter was encoded against."""
    from tools.constants import MAX_TERRAIN_HEIGHT
    assert _terrain()["maxHeight"] == MAX_TERRAIN_HEIGHT


def test_position_is_south_west_corner():
    """position is the SW corner of the grid, and .ter row 0 = south edge,
    so sample (r, c) lands at world (-WORLD_HALF + c*sq, -WORLD_HALF + r*sq)."""
    t = _terrain()
    assert t["position"] == [-WORLD_HALF, -WORLD_HALF, 0]


def test_world_size_is_block_size_times_square_size():
    # world width = .ter size (BLOCK_SIZE) * squareSize
    assert BLOCK_SIZE * SQUARE_SIZE == 2 * WORLD_HALF
