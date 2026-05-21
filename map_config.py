"""
Map-specific configuration for maldon-ng-drive.

All other tools/ modules import from this file (via tools/constants.py)
so changing this one file re-targets the whole pipeline to a different map.
"""

# ── Identity ──────────────────────────────────────────────────────────────────
MAP_NAME    = "maldon"
MAP_TITLE   = "Maldon East"
MAP_VERSION = "1.0.0"
MAP_DESC    = (
    "Maldon, Essex — historic market town on the Blackwater Estuary. "
    "Home to the Battle of Maldon (991 AD) and the eastern terminus of "
    "the GER Maldon branch line (1848–1964). "
    "2 km × 2 km world at 2 m terrain resolution. "
    "Features: Maldon East station site, Hythe Quay, River Blackwater, "
    "and the trackbed of the former branch line."
)
MAP_AUTHOR = "ng-drive-network"

# ── GPS anchor ────────────────────────────────────────────────────────────────
# Centre on Maldon East station site (demolished; approx. location).
# GPS ref: 51.7314°N, 0.6633°E  (TL 856 072)
CENTER_LAT = 51.7314
CENTER_LON =  0.6633

# OSM/elevation query bounding box (S, W, N, E) — slightly wider than the world.
BBOX = (51.712, 0.635, 51.752, 0.695)

# ── Key elevation anchors (metres ASL, from SRTM + local survey) ──────────────
CAMPUS_ELEV  = 10.0   # town centre / station level (Maldon is low-lying)
VALLEY_ELEV  =  2.0   # Blackwater Estuary / saltmarsh
HILL_N_ELEV  = 40.0   # Danbury/Langford ridge to the north-west

# ── Terrain ───────────────────────────────────────────────────────────────────
BLOCK_SIZE  = 1024
SQUARE_SIZE = 2.0

# ── Spawn points ──────────────────────────────────────────────────────────────
# (name, world_X, world_Y, elev_m, yaw_deg_cw_from_north)
# World coords: X east, Y north from CENTER_LAT/LON
SPAWN_POINTS = [
    ("spawn_station_platform",    0,    0,   8.5,  180),  # Maldon East platform
    ("spawn_station_approach",  -80,  -60,   9.0,    0),  # Station Road approach
    ("spawn_hythe_quay",        320, -420,   2.5,   90),  # historic wharf, →east
    ("spawn_market_hill",      -200,  280,  18.0,  180),  # town centre, →south
    ("spawn_promenade",         500, -500,   3.0,  270),  # riverside path, →west
    ("spawn_trackbed_east",     900, -120,  10.0,  270),  # railway exit →west
    ("spawn_trackbed_west",    -950, -130,  12.0,   90),  # railway entry ←east (toward Witham)
    ("spawn_langford_road",    -400,  200,  15.0,  180),  # B1018 north, →south
]
