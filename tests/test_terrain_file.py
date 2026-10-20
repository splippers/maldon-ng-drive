"""Tests for .ter binary format (BeamNG TerrainFile v9) and PNG heightmap."""

import struct
from pathlib import Path

import numpy as np
import pytest

from tools.terrain_file import (write_ter, write_heightmap_png, write_preview_png,
                                write_terrain_json)
from tools.constants import BLOCK_SIZE, GRID_SIZE, MAX_TERRAIN_HEIGHT, SQUARE_SIZE


@pytest.fixture
def flat_elev():
    return np.full((GRID_SIZE, GRID_SIZE), 76.0, dtype=np.float32)


@pytest.fixture
def ramp_elev():
    row = np.linspace(50.0, 90.0, GRID_SIZE, dtype=np.float32)
    return np.tile(row, (GRID_SIZE, 1))


@pytest.fixture
def varied_elev():
    rng = np.random.default_rng(0)
    return (70.0 + rng.standard_normal((GRID_SIZE, GRID_SIZE)) * 5).astype(np.float32)


def _read_ter(path: Path) -> dict:
    with path.open("rb") as f:
        version = struct.unpack("B", f.read(1))[0]
        size    = struct.unpack("<I", f.read(4))[0]
        n       = size * size
        heights = np.frombuffer(f.read(n * 2), dtype="<u2")
        layermap= np.frombuffer(f.read(n),     dtype="u1")
        matcnt  = struct.unpack("<I", f.read(4))[0]
        names   = []
        for _ in range(matcnt):
            ln = struct.unpack("B", f.read(1))[0]
            names.append(f.read(ln).decode("utf-8"))
    return dict(version=version, size=size, heights=heights,
                layermap=layermap, material_names=names)


class TestWriteTerHeader:
    def test_version(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        assert _read_ter(p)["version"] == 9

    def test_size_is_power_of_two(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        size = _read_ter(p)["size"]
        assert size == BLOCK_SIZE  # 1024
        assert (size & (size - 1)) == 0  # power of two

    def test_single_material(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        d = _read_ter(p)
        assert len(d["material_names"]) == 1


class TestWriteTerHeights:
    def test_height_count_is_size_squared(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        d = _read_ter(p)
        assert len(d["heights"]) == d["size"] ** 2  # 1024²

    def test_flat_terrain_constant(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        d = _read_ter(p)
        assert d["heights"].min() == d["heights"].max()

    def test_encoding_76m(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        expected = int(76.0 / MAX_TERRAIN_HEIGHT * 65535)
        actual   = int(_read_ter(p)["heights"][0])
        assert abs(actual - expected) <= 1

    def test_row0_is_south(self, flat_elev, tmp_path):
        """Row 0 = south: a north-low/south-high ramp (varying by row)
        should have its higher values in the first v9 rows."""
        north_low = np.linspace(50.0, 90.0, GRID_SIZE, dtype=np.float32)
        ramp_ns = np.tile(north_low[:, None], (1, GRID_SIZE))  # row 0 (north)=50, row -1 (south)=90
        p = tmp_path / "t.ter"
        write_ter(p, ramp_ns)
        d = _read_ter(p)
        size = d["size"]
        first_row = d["heights"][:size]   # v9 row 0 = south = high
        last_row  = d["heights"][-size:]  # v9 last row = north = low
        assert first_row.mean() > last_row.mean()

    def test_col0_is_west(self, flat_elev, tmp_path):
        """Col 0 = west: a west-low/east-high ramp (varying by column)
        should have its lower values in the first column."""
        west_low = np.linspace(50.0, 90.0, GRID_SIZE, dtype=np.float32)
        ramp_we = np.tile(west_low, (GRID_SIZE, 1))  # col 0 (west)=50, col -1 (east)=90
        p = tmp_path / "t.ter"
        write_ter(p, ramp_we)
        d = _read_ter(p)
        size = d["size"]
        col0 = d["heights"][::size]      # col 0 = west = low
        colN = d["heights"][size-1::size] # col -1 = east = high
        assert col0.mean() < colN.mean()

    def test_varied_range(self, varied_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, varied_elev)
        d = _read_ter(p)
        assert d["heights"].min() < d["heights"].max()

    def test_all_within_uint16(self, varied_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, varied_elev)
        d = _read_ter(p)
        assert d["heights"].max() <= 65535
        assert d["heights"].min() >= 0


class TestWriteTerLayermap:
    def test_layermap_all_zero(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        d = _read_ter(p)
        assert d["layermap"].max() == 0  # material 0 everywhere

    def test_layermap_count_matches_heights(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        d = _read_ter(p)
        assert len(d["layermap"]) == len(d["heights"])


class TestWriteTerrainJson:
    def _write(self, tmp_path, **kw):
        p = tmp_path / "sub" / "t.terrain.json"
        meta = write_terrain_json(p, "/levels/m/terrainGrid/m.ter",
                                  BLOCK_SIZE, ["m_grass"], **kw)
        return p, meta

    def test_creates_file_and_parents(self, tmp_path):
        p, _ = self._write(tmp_path)
        assert p.is_file()

    def test_version_and_size(self, tmp_path):
        _, meta = self._write(tmp_path)
        assert meta["version"] == 9
        assert meta["size"] == BLOCK_SIZE

    def test_sample_counts_match_size(self, tmp_path):
        _, meta = self._write(tmp_path)
        n = BLOCK_SIZE ** 2
        assert meta["heightMapSize"] == n and meta["heightMapItemSize"] == 2
        assert meta["layerMapSize"] == n and meta["layerMapItemSize"] == 1

    def test_materials_round_trip(self, tmp_path):
        _, meta = self._write(tmp_path)
        assert meta["materials"] == ["m_grass"]

    def test_datafile_recorded(self, tmp_path):
        _, meta = self._write(tmp_path)
        assert meta["datafile"].endswith("m.ter")

    def test_heightmap_image_optional(self, tmp_path):
        _, meta = self._write(tmp_path)
        assert "heightmapImage" not in meta
        _, meta2 = self._write(tmp_path, heightmap_image="/levels/m/h.png")
        assert meta2["heightmapImage"] == "/levels/m/h.png"

    def test_file_is_valid_json(self, tmp_path):
        import json
        p, _ = self._write(tmp_path)
        assert json.loads(p.read_text(encoding="utf-8"))["version"] == 9


class TestWriteTerMisc:
    def test_creates_parent_dirs(self, flat_elev, tmp_path):
        deep = tmp_path / "a" / "b" / "terrain.ter"
        write_ter(deep, flat_elev)
        assert deep.exists()

    def test_wrong_shape_raises(self, tmp_path):
        bad = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError):
            write_ter(tmp_path / "bad.ter", bad)

    def test_file_size_approx_3mb(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        size_mb = p.stat().st_size / 1_048_576
        # 1024² × (2 + 1) bytes + strings ≈ 3 MB
        assert 2.5 < size_mb < 3.5, f"Unexpected .ter size: {size_mb:.2f} MB"


class TestPng:
    def test_heightmap_creates_file(self, flat_elev, tmp_path):
        p = tmp_path / "h.png"
        write_heightmap_png(p, flat_elev)
        assert p.exists()

    def test_heightmap_is_valid_png(self, flat_elev, tmp_path):
        p = tmp_path / "h.png"
        write_heightmap_png(p, flat_elev)
        with p.open("rb") as f:
            sig = f.read(8)
        assert sig == b"\x89PNG\r\n\x1a\n"

    def test_heightmap_16bit_depth(self, flat_elev, tmp_path):
        """PNG IHDR should report 16-bit bit depth."""
        import struct, zlib
        p = tmp_path / "h.png"
        write_heightmap_png(p, flat_elev)
        data = p.read_bytes()
        # IHDR starts at offset 8+4+4 = 16 bytes (after sig, length, type)
        ihdr_data = data[16:16+13]
        bit_depth = ihdr_data[8]
        assert bit_depth == 16

    def test_preview_creates_rgb(self, flat_elev, tmp_path):
        from PIL import Image
        p = tmp_path / "preview.png"
        write_preview_png(p, flat_elev)
        img = Image.open(p)
        assert img.mode == "RGB"
        assert img.size == (512, 512)

    def test_preview_varied_colors(self, varied_elev, tmp_path):
        """Preview of varied terrain should have significant colour variation."""
        from PIL import Image
        import numpy as np
        p = tmp_path / "preview.png"
        write_preview_png(p, varied_elev)
        arr = np.array(Image.open(p))
        assert arr.std() > 5.0
