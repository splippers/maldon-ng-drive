"""
Historical photo billboard system — ng-drive network.

Reads data/photos/photo_manifest.json and places in-world photo panels:
  - A vertical TSStatic billboard (double-sided plane) at each GPS location
  - A BeamNGWaypoint trigger so the player can press a key to see the photo
  - An optional "then/now" composite PNG (historical | current satellite crop)

photo_manifest.json format:
  {
    "photos": [
      {
        "id":       "maldon_east_1910",
        "lat":      51.7314,
        "lon":      0.6633,
        "heading":  180,          ← compass bearing the photo faces (degrees, N=0)
        "date":     "c. 1910",
        "caption":  "Maldon East station looking south from the platform",
        "file":     "maldon_east_1910.jpg",   ← relative to data/photos/
        "width_m":  5.0,          ← billboard physical width in game
        "height_m": 3.5           ← billboard physical height in game
      }
    ]
  }

Output:
  data/photos/composite/<id>.png         ← then/now side-by-side image
  levels/<map>/art/textures/photo/<id>.png  ← in-game billboard texture
  (scene objects returned by build_photo_objects())
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


def _paths():
    from tools.constants import DATA_DIR, LEVELS_DIR, MAP_NAME
    return {
        "manifest":  DATA_DIR / "photos" / "photo_manifest.json",
        "photos":    DATA_DIR / "photos",
        "composite": DATA_DIR / "photos" / "composite",
        "textures":  LEVELS_DIR / "art" / "textures" / "photo",
        "shapes":    LEVELS_DIR / "art" / "shapes",
        "map_name":  MAP_NAME,
    }


def load_manifest() -> list[dict]:
    p = _paths()["manifest"]
    if not p.exists():
        log.info("No photo_manifest.json at %s — skipping photo spots", p)
        return []
    data = json.loads(p.read_text())
    return data.get("photos", [])


def _heading_to_quat(heading_deg: float) -> list[float]:
    """Convert compass heading (N=0, E=90) to BeamNG quaternion [x,y,z,w]."""
    # BeamNG: Y = north, so azimuth 0° faces +Y.
    # Rotation around Z axis by -heading (because heading is CW from north).
    angle = math.radians(-heading_deg)
    return [0.0, 0.0, round(math.sin(angle / 2), 6), round(math.cos(angle / 2), 6)]


def _make_composite(photo_path: Path, out_path: Path, caption: str, date: str) -> bool:
    """
    Create a 1024×384 "then/now" composite PNG.
    Left half: historical photo (if available); right half: caption card.
    Returns True on success.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        out_path.parent.mkdir(parents=True, exist_ok=True)
        W, H = 1024, 384

        canvas = Image.new("RGB", (W, H), (30, 30, 30))

        # Left: historical photo
        if photo_path.exists():
            try:
                photo = Image.open(photo_path).convert("RGB")
                # Fit into left half (512×384)
                photo.thumbnail((512, 384), Image.LANCZOS)
                pw, ph = photo.size
                canvas.paste(photo, ((512 - pw) // 2, (384 - ph) // 2))
            except Exception as e:
                log.warning("Could not load photo %s: %s", photo_path, e)
        else:
            # Placeholder
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 170), "[Historical photo]", fill=(160, 160, 160))

        # Divider line
        draw = ImageDraw.Draw(canvas)
        draw.line([(512, 0), (512, H)], fill=(200, 200, 200), width=2)

        # Right: caption card
        draw.rectangle([514, 0, W, H], fill=(20, 25, 35))
        draw.text((530, 30),  date,    fill=(200, 180, 100))
        # Word-wrap caption
        words = caption.split()
        lines, line = [], []
        for word in words:
            line.append(word)
            if len(" ".join(line)) > 30:
                lines.append(" ".join(line[:-1]))
                line = [word]
        if line:
            lines.append(" ".join(line))
        for i, l in enumerate(lines[:8]):
            draw.text((530, 80 + i * 26), l, fill=(230, 230, 230))

        canvas.save(str(out_path))
        return True

    except ImportError:
        log.warning("PIL not available — skipping composite image generation")
        return False
    except Exception as e:
        log.warning("Composite failed for %s: %s", out_path.name, e)
        return False


def build_photo_objects(
    elevation_fn: Callable[[float, float], float],
    make_composites: bool = True,
) -> list[dict]:
    """
    Generate BeamNG scene objects for all photo spots in the manifest.
    Returns a list of dicts (TSStatic billboard + BeamNGWaypoint per photo).
    """
    from tools.constants import WORLD_HALF, gps_to_world

    photos = load_manifest()
    if not photos:
        return []

    p = _paths()
    p["composite"].mkdir(parents=True, exist_ok=True)
    p["textures"].mkdir(parents=True, exist_ok=True)
    p["shapes"].mkdir(parents=True, exist_ok=True)

    objects: list[dict] = []
    placed = 0

    for photo in photos:
        pid     = photo["id"]
        lat     = photo["lat"]
        lon     = photo["lon"]
        heading = photo.get("heading", 0.0)
        date    = photo.get("date", "")
        caption = photo.get("caption", "")
        fname   = photo.get("file", "")
        w_m     = photo.get("width_m", 4.0)
        h_m     = photo.get("height_m", 3.0)

        wx, wy = gps_to_world(lat, lon)
        lim = WORLD_HALF + 50
        if abs(wx) > lim or abs(wy) > lim:
            continue   # outside this map

        z = elevation_fn(wx, wy) + 0.1   # just above ground

        # Composite image
        photo_path = p["photos"] / fname if fname else Path("/nonexistent")
        composite_path = p["composite"] / f"{pid}.png"
        tex_path = p["textures"] / f"{pid}.png"

        if make_composites:
            ok = _make_composite(photo_path, composite_path, caption, date)
            if ok and not tex_path.exists():
                import shutil
                shutil.copy2(composite_path, tex_path)

        # Billboard TSStatic
        tex_rel = f"levels/{p['map_name']}/art/textures/photo/{pid}.png"
        objects.append({
            "class":     "TSStatic",
            "name":      f"photo_{pid}",
            "shapeName": f"levels/{p['map_name']}/art/shapes/billboard_plane.dae",
            "position":  [round(wx, 2), round(wy, 2), round(z, 2)],
            "scale":     [round(w_m, 2), 0.05, round(h_m, 2)],
            "rotation":  _heading_to_quat(heading),
            "persistentId": str(uuid.uuid4()),
            "_photo": {
                "id":      pid,
                "date":    date,
                "caption": caption,
                "texture": tex_rel,
            },
        })

        # Waypoint trigger
        objects.append({
            "class":      "BeamNGWaypoint",
            "name":       f"wp_photo_{pid}",
            "position":   [round(wx, 2), round(wy, 2), round(z, 2)],
            "radius":     8.0,
            "persistentId": str(uuid.uuid4()),
            "_photo_id":  pid,
        })

        placed += 1

    log.info("Photo spots: %d/%d within map extent", placed, len(photos))
    return objects


def write_billboard_dae(shapes_dir: Path | None = None) -> Path:
    """
    Write the shared double-sided billboard plane .dae file.
    1×1 unit in XZ plane (X: -0.5–0.5, Z: 0–1, Y=0), faces +Y.
    Scale via TSStatic to [width_m, 0.05, height_m].
    """
    from tools.constants import LEVELS_DIR, MAP_NAME

    if shapes_dir is None:
        shapes_dir = LEVELS_DIR / "art" / "shapes"
    shapes_dir.mkdir(parents=True, exist_ok=True)

    dae_path = shapes_dir / "billboard_plane.dae"
    if dae_path.exists():
        return dae_path

    # 4 corners of a unit plane in XZ
    verts = [(-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.0, 1.0), (-0.5, 0.0, 1.0)]
    # Front (+Y): tris (0,2,1), (0,3,2)
    # Back  (-Y): tris (0,1,2), (0,2,3)
    FRONT = [(0, 2, 1), (0, 3, 2)]
    BACK  = [(0, 1, 2), (0, 2, 3)]
    UV = {0: (0,0), 1: (1,0), 2: (1,1), 3: (0,1)}
    UV_B = {0: (1,0), 1: (0,0), 2: (0,1), 3: (1,1)}

    pos_f, nrm_f, uv_f = [], [], []
    for tri in FRONT:
        for vi in tri:
            pos_f.extend(verts[vi]); nrm_f.extend((0,1,0)); uv_f.extend(UV[vi])
    for tri in BACK:
        for vi in tri:
            pos_f.extend(verts[vi]); nrm_f.extend((0,-1,0)); uv_f.extend(UV_B[vi])

    n = len(FRONT) + len(BACK)
    p_idx = " ".join(f"{i} {i} {i}" for i in range(n * 3))

    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_effects>
    <effect id="MatEffect">
      <profile_COMMON>
        <newparam sid="surface0"><surface type="2D"><init_from>TexImg</init_from></surface></newparam>
        <newparam sid="sampler0"><sampler2D><source>surface0</source></sampler2D></newparam>
        <technique sid="common">
          <phong>
            <diffuse><texture texture="sampler0" texcoord="UVSET0"/></diffuse>
          </phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_images>
    <image id="TexImg" name="TexImg"><init_from>./placeholder.png</init_from></image>
  </library_images>
  <library_materials>
    <material id="Mat" name="billboard"><instance_effect url="#MatEffect"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="BillboardMesh">
      <mesh>
        <source id="pos"><float_array count="{len(pos_f)}">{' '.join(f'{v:.6f}' for v in pos_f)}</float_array>
          <technique_common><accessor source="#pos-array" count="{n*3}" stride="3">
            <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
          </accessor></technique_common></source>
        <source id="nrm"><float_array count="{len(nrm_f)}">{' '.join(f'{v:.6f}' for v in nrm_f)}</float_array>
          <technique_common><accessor source="#nrm-array" count="{n*3}" stride="3">
            <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
          </accessor></technique_common></source>
        <source id="uv"><float_array count="{len(uv_f)}">{' '.join(f'{v:.6f}' for v in uv_f)}</float_array>
          <technique_common><accessor source="#uv-array" count="{n*3}" stride="2">
            <param name="S" type="float"/><param name="T" type="float"/>
          </accessor></technique_common></source>
        <vertices id="verts"><input semantic="POSITION" source="#pos"/></vertices>
        <triangles count="{n}" material="Mat">
          <input semantic="VERTEX"   source="#verts" offset="0"/>
          <input semantic="NORMAL"   source="#nrm"   offset="1"/>
          <input semantic="TEXCOORD" source="#uv"    offset="2" set="0"/>
          <p>{p_idx}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene">
      <node id="BillboardNode" type="NODE">
        <instance_geometry url="#BillboardMesh">
          <bind_material><technique_common>
            <instance_material symbol="Mat" target="#Mat">
              <bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD" input_set="0"/>
            </instance_material>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
"""
    dae_path.write_text(xml)
    return dae_path
