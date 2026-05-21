"""Tests for .ter binary format and PNG heightmap — v3.0 (1025×1025 grid)."""

import struct
from pathlib import Path

import numpy as np
import pytest

from tools.terrain_file import write_ter, write_heightmap_png, write_preview_png
from tools.constants import BLOCK_SIZE, GRID_SIZE, MAX_TERRAIN_HEIGHT


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
        magic   = f.read(4)
        version = struct.unpack("<I", f.read(4))[0]
        bsize   = struct.unpack("<I", f.read(4))[0]
        nlayers = struct.unpack("<I", f.read(4))[0]
        n_verts = (bsize + 1) ** 2
        heights = np.frombuffer(f.read(n_verts * 2), dtype="<u2")
        flags   = np.frombuffer(f.read(n_verts),     dtype="u1")
    return dict(magic=magic, version=version, bsize=bsize,
                nlayers=nlayers, heights=heights, flags=flags,
                n_verts=n_verts)


class TestWriteTerHeader:
    def test_magic(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        assert _read_ter(p)["magic"] == b"TERR"

    def test_version(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        assert _read_ter(p)["version"] == 2

    def test_block_size_is_1024(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        assert _read_ter(p)["bsize"] == BLOCK_SIZE

    def test_num_layers(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        assert _read_ter(p)["nlayers"] == 1


class TestWriteTerHeights:
    def test_height_count_is_1025_squared(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        d = _read_ter(p)
        assert len(d["heights"]) == GRID_SIZE ** 2  # 1025*1025 = 1 050 625

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

    def test_ramp_monotonic(self, ramp_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, ramp_elev)
        row = _read_ter(p)["heights"][:GRID_SIZE]
        assert all(row[i] <= row[i+1] for i in range(len(row)-1))

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


class TestWriteTerFlags:
    def test_flags_all_zero(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        assert _read_ter(p)["flags"].max() == 0

    def test_flags_count_matches_heights(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        d = _read_ter(p)
        assert len(d["flags"]) == len(d["heights"])


class TestWriteTerMisc:
    def test_creates_parent_dirs(self, flat_elev, tmp_path):
        deep = tmp_path / "a" / "b" / "terrain.ter"
        write_ter(deep, flat_elev)
        assert deep.exists()

    def test_wrong_shape_raises(self, tmp_path):
        bad = np.zeros((100, 100), dtype=np.float32)
        with pytest.raises(ValueError):
            write_ter(tmp_path / "bad.ter", bad)

    def test_file_size_approx_4mb(self, flat_elev, tmp_path):
        p = tmp_path / "t.ter"
        write_ter(p, flat_elev)
        size_mb = p.stat().st_size / 1_048_576
        # 1025² × (2 + 1 + 1) bytes + strings ≈ 4 MB
        assert 3.5 < size_mb < 5.0, f"Unexpected .ter size: {size_mb:.2f} MB"


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
