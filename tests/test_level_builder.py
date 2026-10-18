"""
Tests for the level-selector metadata writer (levels/<map>/info.json).

Map-agnostic: everything is driven by map_config.py / tools.constants, so
the same assertions hold for every map in the ng-drive network.
"""

from __future__ import annotations

import json

import map_config

from tools.constants import MAP_NAME
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


def test_required_fields(tmp_path):
    _, info = _write(tmp_path)
    assert info["type"] == "map"
    assert info["name"]
    assert info["description"]
    assert info["author"]
    assert info["version"]


def test_uses_map_config_metadata(tmp_path):
    _, info = _write(tmp_path)
    assert info["name"] == map_config.MAP_TITLE
    assert info["description"] == map_config.MAP_DESC
    assert info["author"] == map_config.MAP_AUTHOR
    assert info["version"] == map_config.MAP_VERSION


def test_spawn_points_match_level_spawns(tmp_path):
    _, info = _write(tmp_path)
    assert [s["objectname"] for s in info["spawnPoints"]] == \
           [sp[0] for sp in lb._SPAWN_POINTS]
    assert all(s["name"] for s in info["spawnPoints"])


def test_spawn_labels_applied(tmp_path):
    _, info = _write(tmp_path)
    by_obj = {s["objectname"]: s["name"] for s in info["spawnPoints"]}
    for objname, label in getattr(map_config, "SPAWN_LABELS", {}).items():
        assert by_obj.get(objname) == label


def test_unlabelled_spawns_get_title_cased_names(tmp_path):
    labels = getattr(map_config, "SPAWN_LABELS", {})
    _, info = _write(tmp_path)
    by_obj = {s["objectname"]: s["name"] for s in info["spawnPoints"]}
    for objname, name in by_obj.items():
        if objname not in labels:
            expected = objname.removeprefix("spawn_").replace("_", " ").title()
            assert name == expected


def test_preview_points_at_map_preview(tmp_path):
    _, info = _write(tmp_path)
    assert info["previews"][0]["image"] == f"levels/{MAP_NAME}/preview.png"


def test_output_is_readable_utf8(tmp_path):
    info_path, _ = _write(tmp_path)
    raw = info_path.read_text(encoding="utf-8")
    assert "\\u" not in raw          # ensure_ascii=False
    assert raw.endswith("\n")
