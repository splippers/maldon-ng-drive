# maldon-ng-drive

BeamNG:Drive map of **Maldon, Essex** — historic market town on the
Blackwater Estuary and the eastern terminus of the GER Maldon branch line
(1848–1964). 2 km × 2 km world at 2 m terrain resolution.

This repo is the **network template**: `tools/` is shared across every map in
the ng-drive network (witham, cressing, braintree, rayne, felsted …) and is
re-targeted by editing `map_config.py`.

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

python3 tools/fetch_osm.py            # download OSM for the map BBOX
python3 tools/generate.py --lidar     # LiDAR terrain + full map
python3 tools/generate.py --lidar --satellite --zip   # + aerial imagery, pack mod
```

Output: `levels/maldon/main.level.json` plus terrain files under
`levels/maldon/terrainGrid/` and 3D building meshes under
`levels/maldon/art/shapes/buildings/`.

## Pipeline

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `tools/heightmap.py` | Elevation: EA LiDAR DTM 2 m → SRTM cache → live API → synthetic |
| 2 | `tools/satellite.py` | ESRI WorldImagery ground texture (optional) |
| 3 | `tools/terrain_file.py` | Binary `.ter` + heightmap PNG + preview |
| 4 | `tools/osm_parse.py` | Parse OSM into roads / buildings / water / landuse / leisure |
| 5 | `tools/osm_roads.py`, `tools/buildings.py`, … | Scene geometry |
| 5b | `tools/game_elements.py` | Derive game elements from OSM road tags |
| 6 | `tools/buildings3d.py` | Extruded building COLLADA meshes |
| 7 | `tools/railway_network.py` | Shared GER trackbed filtered to this map |
| 8 | `tools/photo_spots.py` | Historical photo billboards |
| 9 | `tools/level_builder.py` | Assemble `main.level.json` |

## Terrain — Environment Agency LiDAR

`tools/lidar.py` replaces the SRTM 30 m base with the **EA LiDAR Composite
DTM (2 m, ±15 cm RMSE)** via the Defra Data Services Platform WCS, using a
pure-Python WGS84 → OSGB36 → EPSG:27700 transform (no GDAL/PROJ runtime
dependency). Tiles are cached in `data/lidar/`, resampled grids in
`data/maldon_lidar_1025.npy`.

LiDAR is used automatically when available; `--lidar` forces a download.
The synthetic road cuts / flattening are skipped for LiDAR bases so the real
survey is preserved (only ±0.3 m micro-detail noise is added).

## Game elements from OSM tags

`tools/game_elements.py` reads the road *semantics* OSM already carries and
derives BeamNG scene elements automatically, so road furniture is data-driven
instead of hand-placed:

| Element | OSM source | BeamNG |
|---------|-----------|--------|
| **Central reservation** | `dual_carriageway=yes`, or paired opposing `oneway=yes` ways | `median_concrete` / `median_grass` DecalRoad |
| **Guard rail** | `barrier=guard_rail / crash_barrier / jersey_barrier` | `guard_rail` DecalRoad on both sides |
| **Sidewalk** | `sidewalk=both / left / right` | `sidewalk` DecalRoad |
| **Cycleway** | `cycleway=lane / track / shared` | `cycleway` DecalRoad |
| **Speed limit** | `maxspeed` (mph → km/h) | `speed_limit` marker every 400 m |
| **Street lighting** | `lit=yes` | `street_lamp` marker every 35 m |
| **Bridge / tunnel** | `bridge=*`, `tunnel=*` | bridge deck levelled; deck/portal marker |
| **Lane width** | `lanes`, `width` | carriageway width |

Dual carriageways are usually mapped as two one-way ways a few metres apart;
they are paired by proximity + opposing direction and the median is laid along
the mid-line. Point elements are emitted as `TSStatic` placeholders (blank
`shapeName`) in the `GameElements` group, ready to be given meshes in the
World Editor.

## Street-level imagery (optional)

Maldon can pull CC-BY-SA licensed street-level photos from
[Mapillary](https://www.mapillary.com/) and place them as in-world
billboards (same pipeline as the historical photo spots):

```bash
export MAPILLARY_ACCESS_TOKEN=MLY|...
python3 tools/fetch_streetview.py --max 40      # fetch + cache + update manifest
# or, integrated into a full build:
python3 tools/generate.py --streetview --zip
```

Without a token the step is skipped. Downloaded images live under
`data/photos/mapillary/` (gitignored) and attribution is written to
`data/photos/mapillary/ATTRIBUTION.txt`.

## Tests

```bash
python3 -m pytest tests -q
```

## Data sources & licences

- OpenStreetMap — © OpenStreetMap contributors, ODbL
- Environment Agency LiDAR Composite DTM — Open Government Licence
- ESRI WorldImagery — for development/preview use
