"""
Tests for the level-selector metadata writer (levels/<map>/info.json).

Map-agnostic: everything is driven by tools/constants.py (fed by
map_config.py), so the same assertions hold for every map in the network.
"""

from __future__ import annotations

import json

from tools.constants import (
    MAP_NAME, MAP_TITLE, MAP_DESC, MAP_AUTHOR, MAP_VERSION,
    BLOCK_SIZE, SQUARE_SIZE,
)
from tools import level_builder as lb


def _write(tmp_path):
    out = tmp_path / "levels" / MAP_NAME / "main.level.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    info_path = lb.write_info_json(out)
    return info_path, json.loads(info_path.read_text(encoding="utf-8"))


def test_writes_next_to_level(tmp_path):
    info_path, _ = _write(tmp_path)
    assert info_path == tmp_path / "levels" / MAP_NAME / "info.json"
    assert info_path.is_file()


def test_beamng_levelinfo_fields(tmp_path):
    _, info = _write(tmp_path)
    for key in ("title", "description", "size", "biome", "roads",
                "suitablefor", "features", "authors", "version",
                "defaultSpawnPointName", "spawnPoints", "previews"):
        assert key in info, f"missing {key}"
    assert isinstance(info["size"], list) and len(info["size"]) == 2


def test_uses_map_metadata(tmp_path):
    _, info = _write(tmp_path)
    assert info["title"] == MAP_TITLE
    assert info["description"] == MAP_DESC
    assert info["authors"] == MAP_AUTHOR
    assert info["version"] == MAP_VERSION
    assert info["size"] == [int(BLOCK_SIZE * SQUARE_SIZE)] * 2


def test_spawn_points_match_level_spawns(tmp_path):
    _, info = _write(tmp_path)
    assert [s["objectname"] for s in info["spawnPoints"]] == \
           [sp[0] for sp in lb._SPAWN_POINTS]
    assert info["defaultSpawnPointName"] == lb._SPAWN_POINTS[0][0]


def test_preview_is_a_filename_list(tmp_path):
    _, info = _write(tmp_path)
    assert info["previews"] == ["preview.png"]


def test_supports_flags(tmp_path):
    _, info = _write(tmp_path)
    assert info["supportsTraffic"] is True
    assert info["supportsTimeOfDay"] is True


def test_output_is_readable_utf8(tmp_path):
    info_path, _ = _write(tmp_path)
    raw = info_path.read_text(encoding="utf-8")
    assert "\\u" not in raw          # ensure_ascii=False
    assert raw.endswith("\n")
