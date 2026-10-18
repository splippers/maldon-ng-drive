"""
Assemble the BeamNG main.level.json scene graph — v3.0.

New in v3.0
───────────
• Real OSM road geometry (all highways in the bounding box)
• Building markers + footprint outlines
• 3D extruded building meshes (.dae) from buildings3d
• River Chelmer + Stebbing Brook water features + OSM stream network
• Historical railway trackbed (Witham–Dunmow branch, closed 1953)
• Sports pitch outlines
• Forest / woodland groups
• 8 vehicle spawn points
• UK-latitude sun settings (51.86°N)
• ScatterSky with English overcast parameters
• Increased terrain LOD settings for 2 m resolution
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from tools.constants import MAP_NAME, SQUARE_SIZE

log = logging.getLogger(__name__)

_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _uid(name: str) -> str:
    return str(uuid.uuid5(_NS, f"{MAP_NAME}.{name}"))


def _obj(cls: str, name: str, **props) -> dict[str, Any]:
    return {"class": cls, "name": name, "persistentId": _uid(name), **props}


def _group(name: str, *children) -> dict[str, Any]:
    o = _obj("SimGroup", name)
    o["children"] = list(children)
    return o


# ── Scene primitives ──────────────────────────────────────────────────────────

def _level_info() -> dict:
    return _obj(
        "LevelInfo", "LevelInfo",
        visibleDistance      = 2000,
        fogColor             = [0.65, 0.68, 0.72, 1.0],
        fogDensity           = 0.0,
        fogDensityOffset     = 900.0,
        canvasClearColor     = [0, 0, 0, 255],
        ambientLightColor    = [0.18, 0.19, 0.22, 1.0],
        advancedLightingModel= True,
    )


def _terrain_block() -> dict:
    return _obj(
        "TerrainBlock", "Terrain",
        position            = [0, 0, 0],
        rotation            = [1, 0, 0, 0],
        scale               = [1, 1, 1],
        squareSize          = SQUARE_SIZE,
        terrainFile         = f"levels/{MAP_NAME}/terrainGrid/{MAP_NAME}.ter",
        baseTexSize         = 512,           # higher for 2 m terrain
        overrideGroundModel = False,
        physicsProxyType    = "Trimesh",
        castDynamicShadows  = False,
        maxDetailDistance   = 300,
        screenError         = 8,             # finer LOD splits for 2 m terrain
    )


def _sun() -> dict:
    # UK summer afternoon: sun bearing ~214° (SSW), elevation ~45°
    return _obj(
        "Sun", "TheSun",
        azimuth          = 214,
        elevation        = 42,
        color            = [0.93, 0.91, 0.86, 1.0],
        ambient          = [0.20, 0.21, 0.25, 1.0],
        shadowDistance   = 600,
        shadowSoftness   = 0.20,
        numSplits        = 4,
        logWeight        = 0.91,
        attenuationRatio = [0.0, 1.0, 1.0],
    )


def _scatter_sky() -> dict:
    return _obj(
        "ScatterSky", "ScatterSky",
        skyBrightness     = 22,
        sunSize           = 1.0,
        colorizeAmount    = 0.15,
        colorize          = [0.65, 0.80, 1.0],
        rayleighScattering= 0.0040,
        mieScattering     = 0.0055,     # slight haze typical of Essex
        exposure          = 1.0,
        nightColor        = [0.02, 0.02, 0.08, 1.0],
        windSpeed         = 1.5,
    )


def _spawn(name: str, x: float, y: float, z: float,
           yaw_deg: float = 0.0) -> dict:
    import math
    half = math.radians(yaw_deg) / 2
    return _obj(
        "SpawnSphere", name,
        position     = [round(x,2), round(y,2), round(z+0.5,2)],
        rotation     = [0.0, 0.0, round(math.sin(half),4), round(math.cos(half),4)],
        radius       = 3.0,
        sphereWeight = 100,
        indoorWeight = 0,
        outdoorWeight= 100,
    )


def _decal_road(road: dict) -> dict:
    nodes = [
        [round(n[0],2), round(n[1],2), round(n[2],2), road["width"]]
        for n in road["nodes"]
    ]
    return _obj(
        "DecalRoad", road["name"],
        material        = road["material"],
        renderPriority  = 10,
        textureLength   = 5,
        drivability     = 1,
        improvedSpline  = True,
        breakAngle      = 3,
        depthBias       = -0.001,
        nodes           = nodes,
    )


# ── Spawn points — loaded from map_config.py if defined, else defaults ────────
def _load_spawn_points() -> list[tuple]:
    try:
        import map_config
        if hasattr(map_config, "SPAWN_POINTS"):
            return map_config.SPAWN_POINTS
    except ModuleNotFoundError:
        pass
    # Default spawn points (Felsted layout — override in map_config.py)
    return [
        ("spawn_main_entrance",    -248, -200, 72.5,  90),
        ("spawn_car_park",         -296, -395, 72.5, 180),
        ("spawn_campus_centre",       0,    0, 76.5,   0),
        ("spawn_sports_fields",     450,  490, 73.5, 270),
        ("spawn_chapel_approach",  -140, -180, 74.0,  90),
        ("spawn_north_road",       -238,  800, 81.0, 180),
        ("spawn_braintree_east",    900,  -45, 74.5, 270),
        ("spawn_sports_centre",     200,  110, 76.0, 180),
    ]

_SPAWN_POINTS = _load_spawn_points()


# ── Level metadata (info.json) — BeamNG LevelInfo schema ──────────────────────
def _info_meta() -> dict:
    """Level-selector metadata from tools/constants.py (fed by map_config.py)."""
    from tools import constants as C
    return {
        "title":       C.MAP_TITLE,
        "desc":        C.MAP_DESC,
        "author":      C.MAP_AUTHOR,
        "version":     C.MAP_VERSION,
        "biome":       getattr(C, "MAP_BIOME", "English Countryside"),
        "roads":       getattr(C, "MAP_ROADS", "OSM road network"),
        "suitablefor": getattr(C, "MAP_SUITABLEFOR", "Free Roam"),
        "features":    getattr(C, "MAP_FEATURES", "OSM roads, LiDAR terrain"),
        "labels":      getattr(C, "SPAWN_LABELS", {}),
    }


def write_info_json(out_path: Path | str,
                    level_size: int | None = None) -> Path:
    """
    Write levels/<map>/info.json in BeamNG's LevelInfo schema.

    BeamNG reads this to populate the level selector (title, size, biome,
    spawn points, preview).  Everything comes from tools/constants.py, so
    every map in the ng-drive network gets a valid, consistent file.
    """
    out_path  = Path(out_path)
    level_dir = out_path.parent
    meta      = _info_meta()

    if level_size is None:
        from tools.constants import BLOCK_SIZE, SQUARE_SIZE
        level_size = int(BLOCK_SIZE * SQUARE_SIZE)

    spawn_points = []
    for sp in _SPAWN_POINTS:
        entry = {"objectname": sp[0]}
        label = meta["labels"].get(sp[0])
        if label:
            entry["name"] = label
        spawn_points.append(entry)

    info = {
        "title":       meta["title"],
        "description": meta["desc"],
        "size":        [level_size, level_size],
        "biome":       meta["biome"],
        "roads":       meta["roads"],
        "suitablefor": meta["suitablefor"],
        "features":    meta["features"],
        "authors":     meta["author"],
        "version":     meta["version"],
        "defaultSpawnPointName": spawn_points[0]["objectname"] if spawn_points else "",
        "spawnPoints":       spawn_points,
        "supportsTraffic":   True,
        "supportsTimeOfDay": True,
        "previews":          ["preview.png"],
    }

    info_path = level_dir / "info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")
    log.info("Level info written: %s (%d spawns)", info_path, len(spawn_points))
    return info_path


# ── Public API ─────────────────────────────────────────────────────────────────

def build_level(roads:            list[dict],
                buildings:        tuple[list[dict], list[dict]] | None = None,
                water:            list[dict] | None = None,
                vegetation:       list[dict] | None = None,
                building_meshes:  list[dict] | None = None,
                railway:          list[dict] | None = None,
                photos:           list[dict] | None = None,
                game_elements:    dict | None = None,
                out_path:         Path | str = None) -> None:
    """
    Write the main.level.json scene graph.

    Parameters
    ----------
    roads           : road dicts from osm_roads.build_roads()
    buildings       : (markers, footprints) from buildings.build_building_objects()
    water           : objects from water.build_water_objects()
    vegetation      : objects from vegetation.build_vegetation_objects()
    building_meshes : 3D TSStatic objects from buildings3d.build_building_meshes()
    railway         : historical railway objects from railway.build_railway_objects()
    game_elements   : {"linear": [...], "objects": [...]} from
                      game_elements.build_game_elements() — medians/barriers
                      join the Roads group, points go in a GameElements group
    out_path        : destination file
    """
    if out_path is None:
        from tools.constants import LEVELS_DIR
        out_path = LEVELS_DIR / "main.level.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    elements     = game_elements or {}
    elem_linear  = list(elements.get("linear", []))
    elem_objects = list(elements.get("objects", []))

    # Roads (real carriageways + derived linear elements like medians)
    roads_group = _group("Roads", *[_decal_road(r) for r in [*roads, *elem_linear]])

    # Buildings — markers/footprints from osm buildings + 3D meshes from buildings3d
    bld_markers:    list[dict] = []
    bld_footprints: list[dict] = []
    if buildings:
        bld_markers, bld_footprints = buildings
    bld_3d = list(building_meshes or [])
    bld_group = _group("Buildings",
                        *bld_markers,
                        *bld_3d,
                        _group("Footprints", *bld_footprints))

    # Water
    water_group = _group("Water", *(water or []))

    # Vegetation
    veg_group = _group("Vegetation", *(vegetation or []))

    # Historical railway
    rail_group = _group("Railway", *(railway or []))

    # Historical photo billboards
    photo_group = _group("Photos", *(photos or []))

    # Game elements (speed signs, lamps, bridge/tunnel markers)
    elem_group = _group("GameElements", *elem_objects)

    # Spawn — reload each call so live changes to map_config.py take effect
    spawn_pts = _load_spawn_points()
    spawn_group = _group("Spawn", *[_spawn(*s) for s in spawn_pts])

    mission_group = _group(
        "MissionGroup",
        _level_info(),
        _scatter_sky(),
        _sun(),
        _terrain_block(),
        spawn_group,
        roads_group,
        bld_group,
        water_group,
        veg_group,
        rail_group,
        photo_group,
        elem_group,
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mission_group, f, indent=2)

    # Level-selector metadata (info.json) — always emitted alongside the level
    write_info_json(out_path)

    road_count  = len(roads) + len(elem_linear)
    bld_count   = len(bld_markers) + len(bld_3d)
    water_count = len(water or [])
    veg_count   = len(vegetation or [])
    rail_count  = len(railway or [])
    log.info(
        "Level written: %d roads (+%d linear elements), %d buildings, "
        "%d water, %d vegetation, %d railway, %d game objects, %d spawns → %s",
        road_count, len(elem_linear), bld_count, water_count, veg_count,
        rail_count, len(elem_objects), len(_SPAWN_POINTS), out_path,
    )
