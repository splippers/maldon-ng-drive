"""Tests for the historical photo billboard system."""

import json
import pytest
from pathlib import Path

from tools.photo_spots import (
    load_manifest,
    build_photo_objects,
    write_billboard_dae,
    _heading_to_quat,
)
from tools.constants import WORLD_HALF


_ELEV_FN = lambda wx, wy: 8.0


class TestManifest:
    def test_loads(self):
        photos = load_manifest()
        assert isinstance(photos, list)

    def test_each_entry_has_required_fields(self):
        for p in load_manifest():
            assert "id" in p
            assert "lat" in p and "lon" in p
            assert "caption" in p

    def test_latlons_plausible(self):
        for p in load_manifest():
            assert 51.0 < p["lat"] < 52.5, f"{p['id']} lat={p['lat']}"
            assert -1.0 < p["lon"] < 1.5,  f"{p['id']} lon={p['lon']}"

    def test_ids_unique(self):
        photos = load_manifest()
        ids = [p["id"] for p in photos]
        assert len(ids) == len(set(ids)), "Duplicate photo IDs"


class TestHeadingToQuat:
    def test_north_is_identity_rotation(self):
        """Heading 0 (north) → no rotation needed (facing +Y)."""
        q = _heading_to_quat(0.0)
        assert len(q) == 4
        # cos(0/2) = 1, sin(0/2) = 0
        assert abs(q[3]) == pytest.approx(1.0, abs=0.001)
        assert abs(q[2]) == pytest.approx(0.0, abs=0.001)

    def test_south_is_180_rotation(self):
        q = _heading_to_quat(180.0)
        assert abs(q[3]) == pytest.approx(0.0, abs=0.01)

    def test_returns_four_floats(self):
        q = _heading_to_quat(90.0)
        assert len(q) == 4
        for v in q:
            assert isinstance(v, float)

    def test_unit_quaternion(self):
        import math
        q = _heading_to_quat(45.0)
        magnitude = math.sqrt(sum(v**2 for v in q))
        assert magnitude == pytest.approx(1.0, abs=0.001)


class TestBuildPhotoObjects:
    def test_returns_list(self):
        result = build_photo_objects(_ELEV_FN, make_composites=False)
        assert isinstance(result, list)

    def test_in_world_photos_included(self):
        """Maldon photos should be within this map's world extent."""
        result = build_photo_objects(_ELEV_FN, make_composites=False)
        assert len(result) >= 2   # at least 2 Maldon-area photos in manifest

    def test_each_photo_has_billboard_and_waypoint(self):
        """Each manifest entry produces a TSStatic + a BeamNGWaypoint."""
        result = build_photo_objects(_ELEV_FN, make_composites=False)
        billboards = [o for o in result if o.get("class") == "TSStatic"]
        waypoints  = [o for o in result if o.get("class") == "BeamNGWaypoint"]
        # Should be equal count
        assert len(billboards) == len(waypoints)

    def test_billboard_has_scale(self):
        result = build_photo_objects(_ELEV_FN, make_composites=False)
        billboards = [o for o in result if o.get("class") == "TSStatic"]
        for b in billboards:
            assert "scale" in b
            assert len(b["scale"]) == 3
            assert b["scale"][0] > 0   # width
            assert b["scale"][2] > 0   # height

    def test_waypoint_has_radius(self):
        result = build_photo_objects(_ELEV_FN, make_composites=False)
        waypoints = [o for o in result if o.get("class") == "BeamNGWaypoint"]
        for wp in waypoints:
            assert wp["radius"] > 0

    def test_out_of_world_excluded(self, tmp_path, monkeypatch):
        """A photo far outside the map should be silently excluded."""
        import tools.photo_spots as ps
        fake_manifest = {
            "photos": [
                {"id": "far_away", "lat": 53.0, "lon": 2.0,
                 "heading": 0, "date": "", "caption": "Far", "file": ""}
            ]
        }
        fake_path = tmp_path / "photo_manifest.json"
        fake_path.write_text(json.dumps(fake_manifest))
        monkeypatch.setattr(ps, "_paths", lambda: {
            "manifest": fake_path, "photos": tmp_path, "composite": tmp_path / "c",
            "textures": tmp_path / "t", "shapes": tmp_path / "s", "map_name": "test"
        })
        result = ps.build_photo_objects(_ELEV_FN, make_composites=False)
        assert result == []

    def test_persistent_ids_unique(self):
        result = build_photo_objects(_ELEV_FN, make_composites=False)
        ids = [o["persistentId"] for o in result if "persistentId" in o]
        assert len(ids) == len(set(ids))


class TestBillboardDae:
    def test_creates_file(self, tmp_path):
        path = write_billboard_dae(shapes_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".dae"

    def test_valid_xml(self, tmp_path):
        import xml.etree.ElementTree as ET
        path = write_billboard_dae(shapes_dir=tmp_path)
        tree = ET.parse(path)
        root = tree.getroot()
        assert "COLLADA" in root.tag

    def test_has_texture_reference(self, tmp_path):
        path = write_billboard_dae(shapes_dir=tmp_path)
        content = path.read_text()
        assert "sampler" in content or "texture" in content.lower()

    def test_idempotent(self, tmp_path):
        """Calling twice should not overwrite the existing file."""
        path1 = write_billboard_dae(shapes_dir=tmp_path)
        mtime1 = path1.stat().st_mtime
        path2 = write_billboard_dae(shapes_dir=tmp_path)
        mtime2 = path2.stat().st_mtime
        assert mtime1 == mtime2
