"""Auto-fix for scene/robot jsonc configs when schema validation fails."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

def _autofix_scene_and_robot_configs(scene_path: Path, sim_config_path: str) -> tuple[tempfile.TemporaryDirectory, str, str]:
    """
    Делает временную "совместимую со схемой" копию scene/robot config'ов.

    Проблема: некоторые конфиги из example_user_scripts используют:
    - lidar-type: "generic-cylindrical" (через дефис) вместо "generic_cylindrical"
    - дополнительные поля report-point-cloud / report-azimuth-elevation-range, которых нет в schema/robot_config_schema.jsonc

    ВАЖНО: исходные файлы не трогаем — создаём временную папку и пишем туда JSON (без комментариев).

    Returns:
        (tmp_dir_obj, fixed_scene_filename, fixed_sim_config_path)
    """
    import commentjson  # зависимость projectairsim

    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="projectairsim_autofix_")
    tmp_dir = Path(tmp_dir_obj.name)

    # load original scene (jsonc)
    with scene_path.open("r", encoding="utf-8") as f:
        scene = commentjson.load(f)

    # mapping for lidar-type normalization
    lidar_type_map = {
        "generic-cylindrical": "generic_cylindrical",
        "gpu-cylindrical": "gpu_cylindrical",
        "generic-rosette": "generic_rosette",
    }

    # Fix robot-configs referenced by scene
    actors = scene.get("actors", [])
    for actor in actors:
        if actor.get("type") != "robot":
            continue
        robot_cfg_rel = actor.get("robot-config")
        if not robot_cfg_rel or not isinstance(robot_cfg_rel, str):
            continue

        robot_cfg_src = scene_path.parent / robot_cfg_rel
        if not robot_cfg_src.exists():
            # leave as-is; World will error later with path issue
            continue

        with robot_cfg_src.open("r", encoding="utf-8") as rf:
            robot_cfg = commentjson.load(rf)

        sensors = robot_cfg.get("sensors", [])
        if isinstance(sensors, list):
            for s in sensors:
                if not isinstance(s, dict):
                    continue
                if s.get("type") != "lidar":
                    continue

                lt = s.get("lidar-type")
                if isinstance(lt, str) and lt in lidar_type_map:
                    s["lidar-type"] = lidar_type_map[lt]

                # remove fields that are not present in current robot schema
                s.pop("report-point-cloud", None)
                s.pop("report-azimuth-elevation-range", None)

        # write fixed robot config to temp dir
        fixed_robot_name = f"autofix_{Path(robot_cfg_rel).name}"
        fixed_robot_path = tmp_dir / fixed_robot_name
        fixed_robot_path.write_text(json.dumps(robot_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        # point scene actor to fixed robot config (relative to new sim_config_path)
        actor["robot-config"] = fixed_robot_name

    # write fixed scene config
    fixed_scene_name = f"autofix_{scene_path.name}"
    fixed_scene_path = tmp_dir / fixed_scene_name
    fixed_scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")

    # Return temp dir as sim_config_path and the fixed scene name within it
    return tmp_dir_obj, fixed_scene_name, str(tmp_dir)


