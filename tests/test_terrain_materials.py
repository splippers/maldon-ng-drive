"""
Tests for terrain material definitions and the generated textures.

Map-agnostic: the material names are supplied by the caller (the generator
passes whatever names it wrote into the .ter), so the same assertions hold
for every map in the network.
"""

from __future__ import annotations

import json

import numpy as np

from tools.terrain_materials import (
    write_terrain_materials, _look_for, _noise_field, _texture_for,
)


def test_writes_main_materials_json(tmp_path):
    write_terrain_materials(tmp_path, "m", ["m_grass"])
    out = tmp_path / "art" / "terrains" / "main.materials.json"
    assert out.is_file()


def test_one_texture_per_material(tmp_path):
    write_terrain_materials(tmp_path, "m", ["m_grass", "Dirt", "Asphalt"])
    art = tmp_path / "art" / "terrains"
    assert (art / "m_grass_diffuse.png").is_file()
    assert (art / "Dirt_diffuse.png").is_file()
    assert (art / "Asphalt_diffuse.png").is_file()


def test_definition_keys_match_internal_names(tmp_path):
    names = ["m_grass", "Dirt", "Asphalt"]
    defs = write_terrain_materials(tmp_path, "m", names)
    assert set(defs) == set(names)
    for name in names:
        assert defs[name]["internalName"] == name


def test_all_defs_are_terrain_materials(tmp_path):
    defs = write_terrain_materials(tmp_path, "m", ["m_grass"])
    for d in defs.values():
        assert d["class"] == "TerrainMaterial"


def test_diffuse_map_points_at_generated_texture(tmp_path):
    defs = write_terrain_materials(tmp_path, "m", ["m_grass"])
    assert defs["m_grass"]["diffuseMap"] == \
        "/levels/m/art/terrains/m_grass_diffuse.png"


def test_diffuse_size_is_positive_metres(tmp_path):
    defs = write_terrain_materials(tmp_path, "m", ["m_grass"], tile_meters=64.0)
    assert defs["m_grass"]["diffuseSize"] == 64.0


def test_written_json_matches_returned_defs(tmp_path):
    defs = write_terrain_materials(tmp_path, "m", ["m_grass"])
    out = tmp_path / "art" / "terrains" / "main.materials.json"
    assert json.loads(out.read_text(encoding="utf-8")) == defs


def test_groundmodels_are_semantic(tmp_path):
    defs = write_terrain_materials(tmp_path, "m", ["Grass", "Dirt", "Asphalt"])
    assert defs["Grass"]["groundmodelName"] == "GRASS"
    assert defs["Dirt"]["groundmodelName"] == "DIRT"
    assert defs["Asphalt"]["groundmodelName"] == "ASPHALT"


def test_look_matching_is_case_insensitive():
    assert _look_for("m_grass")[2] == "GRASS"
    assert _look_for("GRASS")[2] == "GRASS"
    assert _look_for("Asphalt")[2] == "ASPHALT"


def test_unknown_material_gets_a_default_look():
    assert _look_for("mystery")[2] == "GRASS"


def test_texture_has_expected_size_and_mode(tmp_path):
    from PIL import Image
    img = _texture_for("m_grass", 64, np.random.default_rng(0))
    assert img.size == (64, 64) and img.mode == "RGB"


def test_noise_field_is_in_unit_range():
    n = _noise_field(64, np.random.default_rng(0))
    assert n.shape == (64, 64)
    assert 0.0 <= n.min() and n.max() <= 1.0


def test_noise_wraps_seamlessly():
    """The left and right edges are one lattice step apart, not a seam."""
    n = _noise_field(64, np.random.default_rng(0))
    seam = np.abs(n[:, 0] - n[:, -1]).mean()
    interior = np.abs(n[:, 16] - n[:, 17]).mean()
    assert seam <= interior * 3 + 0.05
