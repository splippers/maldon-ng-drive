"""Tests for Mapillary street-level imagery fetcher."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest
import requests

from tools import fetch_streetview as fs


def _img(i, lat, lon, angle=90.0, captured="2021-05-01T00:00:00Z", url="http://x/t.jpg"):
    return {
        "id": str(i),
        "thumb_1024_url": url,
        "computed_geometry": {"type": "Point", "coordinates": [lon, lat]},
        "captured_at": captured,
        "compass_angle": angle,
    }


def test_to_photo_fields():
    img = _img(42, 51.7, 0.66, angle=123.0, captured="2020-01-02T00:00:00Z")
    p = fs._to_photo(img, 51.7, 0.66)
    assert p["id"] == "mapillary_42"
    assert p["lat"] == 51.7
    assert p["lon"] == 0.66
    assert p["heading"] == 123.0
    assert p["date"] == "2020-01-02"
    assert p["file"] == "mapillary_42.jpg"
    assert "attribution" in p


def test_sample_dedups_by_spacing():
    imgs = [_img(i, 51.7 + i * 0.0001, 0.66) for i in range(10)]
    sampled = fs._sample(imgs, min_spacing_m=80, max_images=100)
    # not all 10 should survive an 80m spacing on a tiny cluster? Actually
    # 0.0001 deg lat ~ 11m, so 80m spacing keeps roughly 1 in 8.
    assert 1 <= len(sampled) <= 10
    # max_images cap
    sampled2 = fs._sample(imgs, min_spacing_m=1, max_images=3)
    assert len(sampled2) == 3


def test_sample_skips_missing_geometry():
    bad = {"id": "9", "thumb_1024_url": "u"}
    sampled = fs._sample([bad], min_spacing_m=1, max_images=5)
    assert sampled == []


def test_fetch_without_token_returns_empty():
    assert fs.fetch_streetview((51.7, 0.6, 51.8, 0.7), None) == []
    assert fs.fetch_streetview((51.7, 0.6, 51.8, 0.7), "") == []


def test_fetch_downloads_and_caches(tmp_path, monkeypatch):
    # redirect DATA_DIR/photos to tmp
    monkeypatch.setattr(fs, "DATA_DIR", tmp_path)
    # ensure photos dir resolution uses patched DATA_DIR
    monkeypatch.setattr("tools.constants.DATA_DIR", tmp_path)

    imgs = [_img(1, 51.72, 0.65), _img(2, 51.74, 0.67)]

    def fake_get(url, headers=None, params=None, timeout=None):
        resp = mock.MagicMock()
        if "graph.mapillary.com" in url:
            resp.json.return_value = {"data": imgs, "paging": {}}
            resp.raise_for_status.return_value = None
        else:
            resp.content = b"FAKEJPEG"
            resp.raise_for_status.return_value = None
        return resp

    monkeypatch.setattr(requests, "get", fake_get)

    photos = fs.fetch_streetview((51.7, 0.6, 51.8, 0.7), "TOK", max_images=10, min_spacing_m=1)
    assert len(photos) == 2
    photo_dir = tmp_path / "photos" / "mapillary"
    assert (photo_dir / "mapillary_1.jpg").exists()
    assert (photo_dir / "ATTRIBUTION.txt").exists()
    # cached: second run should not re-download (still works)
    photos2 = fs.fetch_streetview((51.7, 0.6, 51.8, 0.7), "TOK", max_images=10, min_spacing_m=1)
    assert len(photos2) == 2


def test_merge_manifest_dedups(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "DATA_DIR", tmp_path)
    monkeypatch.setattr("tools.constants.DATA_DIR", tmp_path)
    # initial manifest
    mpath = tmp_path / "photos" / "photo_manifest.json"
    mpath.parent.mkdir(parents=True)
    mpath.write_text(json.dumps({"photos": [{"id": "existing", "lat": 1, "lon": 2}]}))

    new = [{"id": "mapillary_1", "lat": 3, "lon": 4, "file": "m.jpg", "_url": "x"}]
    fs.merge_manifest(new)
    data = json.loads(mpath.read_text())
    ids = [p["id"] for p in data["photos"]]
    assert "existing" in ids
    assert "mapillary_1" in ids
    # no _url leaked
    assert all("_url" not in p for p in data["photos"])
    # dedup on re-merge
    fs.merge_manifest(new)
    data2 = json.loads(mpath.read_text())
    assert sum(1 for p in data2["photos"] if p["id"] == "mapillary_1") == 1


def test_query_failure_returns_empty(monkeypatch):
    def boom(url, headers=None, params=None, timeout=None):
        raise requests.ConnectionError("no network")
    monkeypatch.setattr(requests, "get", boom)
    photos = fs.fetch_streetview((51.7, 0.6, 51.8, 0.7), "TOK", max_images=5, min_spacing_m=1)
    assert photos == []