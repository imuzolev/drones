"""
Программа для управления дроном в ProjectAirSim.
Выполняет облет по периметру (квадрат) и сканирование местности в каждом углу с учетом препятствий.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import tempfile
import time
import math
import numpy as np
from pathlib import Path
from typing import Optional

# Импорты из пакета lidar_mapping
from lidar_mapping.bootstrap import ProjectAirSimClient, Drone, World, REPO_ROOT
from lidar_mapping.config_autofix import _autofix_scene_and_robot_configs
from lidar_mapping.lio import SimpleLIO
from lidar_mapping.mapping import PointCloudAccumulator, save_point_cloud_to_ply, _transform_points_body_to_world, _point_cloud_to_2d_map, _save_map_to_png
from lidar_mapping.path_tracker import PathTracker
from lidar_mapping.state import LidarLatest, PoseLatest, ImuLatest
from lidar_mapping.visualizer import RealtimePointCloudVisualizer

def _get_repulsive_velocity(
    lidar_points: np.ndarray, 
    influence_dist: float = 3.0, 
    max_repulse: float = 2.0
) -> tuple[float, float]:
    """Рассчитывает вектор отталкивания от препятствий в плоскости XY (body frame)"""
    if lidar_points is None or lidar_points.size == 0:
        return 0.0, 0.0
        
    # Фильтруем точки по высоте (на уровне дрона +/- 0.5м)
    # Z в body frame: + вниз. 
    mask_z = np.abs(lidar_points[:, 2]) < 0.5
    pts = lidar_points[mask_z]
    
    if pts.size == 0:
        return 0.0, 0.0

    # Вычисляем расстояния в плоскости XY
    dists_sq = pts[:, 0]**2 + pts[:, 1]**2
    dists = np.sqrt(dists_sq)
    
    # Берем только точки внутри радиуса влияния
    mask_dist = (dists < influence_dist) & (dists > 0.1)
    pts_near = pts[mask_dist]
    dists_near = dists[mask_dist]
    
    if pts_near.size == 0:
        return 0.0, 0.0
        
    # Сила отталкивания ~ 1/dist
    # Вектор от препятствия к дрону = -pts_near
    # Нормируем вектор: -pts_near / dists_near
    # Вес: (1/dists_near - 1/influence_dist)
    
    weights = (1.0 / dists_near - 1.0 / influence_dist)
    # Нормализуем векторы направления (от препятствия к дрону)
    # pts_near это координаты препятствия. Нам нужно отталкивание, т.е. вектор ОТ препятствия (0,0) - (x,y) = (-x, -y)
    fx = -pts_near[:, 0] / dists_near * weights
    fy = -pts_near[:, 1] / dists_near * weights
    
    total_fx = np.sum(fx)
    total_fy = np.sum(fy)
    
    # Ограничиваем максимальную силу
    force_mag = math.hypot(total_fx, total_fy)
    if force_mag > max_repulse:
        scale = max_repulse / force_mag
        total_fx *= scale
        total_fy *= scale
        
    return total_fx, total_fy

async def fly_to_point_with_avoidance(
    drone: Drone,
    target_n: float,
    target_e: float,
    height: float,
    pose_latest: PoseLatest,
    lidar_latest: LidarLatest,
    cruise_speed: float = 2.0,
    stop_dist: float = 1.0,
    avoid_dist: float = 1.5,
    influence_dist: float = 3.0
):
    """
    Летит к заданной точке с избеганием препятствий.
    Если препятствие слишком близко, останавливается или обходит.
    """
    print(f"[NAV] Навигация к ({target_n:.1f}, {target_e:.1f}) с обходом препятствий...")
    
    dt = 0.1
    timeout = 120.0 # 2 минуты макс на сегмент
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # 1. Получаем текущую позицию
        pose_msg, _ = pose_latest.snapshot()
        if not pose_msg or not isinstance(pose_msg, dict):
            await asyncio.sleep(dt)
            continue
            
        pos = pose_msg.get("position", {})
        cur_n = float(pos.get("x", 0.0))
        cur_e = float(pos.get("y", 0.0))
        cur_z = float(pos.get("z", 0.0))
        
        # Получаем yaw (рыскание)
        ori = pose_msg.get("orientation", {})
        # Упрощенный перевод кватерниона в yaw (только если нужно для body frame)
        # q0=w, q1=x, q2=y, q3=z
        q0 = float(ori.get("w", 1.0))
        q1 = float(ori.get("x", 0.0))
        q2 = float(ori.get("y", 0.0))
        q3 = float(ori.get("z", 0.0))
        yaw = math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))

        # 2. Вектор к цели (в world frame NED)
        dn = target_n - cur_n
        de = target_e - cur_e
        dist_to_target = math.hypot(dn, de)
        
        if dist_to_target < stop_dist:
            print(f"[NAV] Цель достигнута (dist={dist_to_target:.2f}m)")
            break
            
        # Нормализуем желаемый вектор скорости
        speed_cmd = min(cruise_speed, dist_to_target) # Замедляемся у цели
        vn_des = (dn / dist_to_target) * speed_cmd
        ve_des = (de / dist_to_target) * speed_cmd
        
        # 3. Переводим желаемую скорость в Body Frame (так как лидар в body frame)
        # V_body = R_trans * V_world
        # v_fwd = vn * cos(yaw) + ve * sin(yaw)
        # v_right = -vn * sin(yaw) + ve * cos(yaw)
        
        v_fwd_des = vn_des * math.cos(yaw) + ve_des * math.sin(yaw)
        v_right_des = -vn_des * math.sin(yaw) + ve_des * math.cos(yaw)
        
        # 4. Получаем данные лидара и рассчитываем отталкивание
        pts, _ = lidar_latest.snapshot()
        rep_fwd, rep_right = 0.0, 0.0
        
        min_dist_detected = 999.0
        
        if pts is not None and pts.size > 0:
            # Рассчитываем вектор отталкивания
            rep_fwd, rep_right = _get_repulsive_velocity(pts, influence_dist=influence_dist, max_repulse=2.0)
            
            # Проверяем минимальную дистанцию спереди для экстренной остановки
            # Конус спереди +/- 30 градусов
            dists = np.linalg.norm(pts[:, :2], axis=1)
            angles = np.arctan2(pts[:, 1], pts[:, 0])
            mask_front = (np.abs(angles) < math.radians(30)) & (dists > 0.1)
            if np.any(mask_front):
                min_dist_detected = np.min(dists[mask_front])
        
        # 5. Комбинируем скорости
        # Если препятствие ОЧЕНЬ близко (< avoid_dist), repulsion будет сильным
        # Если < 1.0м, просто останавливаем движение вперед
        
        v_fwd_final = v_fwd_des + rep_fwd
        v_right_final = v_right_des + rep_right
        
        # Экстренное торможение при опасной близости
        if min_dist_detected < 1.0: # Порог безопасности (был 12, просили минимальный)
             print(f"[NAV] ОПАСНОСТЬ ({min_dist_detected:.2f}м)! Стоп/Отход.")
             v_fwd_final = -0.5 # Немного назад
        elif min_dist_detected < avoid_dist:
             # Замедляемся пропорционально
             scale = (min_dist_detected - 1.0) / (avoid_dist - 1.0)
             scale = max(0.0, min(1.0, scale))
             v_fwd_final *= scale
             print(f"[NAV] Препятствие {min_dist_detected:.1f}м, замедление...")

        # Ограничение скорости
        total_vel = math.hypot(v_fwd_final, v_right_final)
        if total_vel > cruise_speed:
            scale = cruise_speed / total_vel
            v_fwd_final *= scale
            v_right_final *= scale
            
        # Управление высотой (простой P-контроллер)
        dz = height - cur_z
        vz_cmd = np.clip(dz * 1.0, -1.0, 1.0)
        
        # Поворот (yaw): смотрим в сторону движения или цели?
        # Лучше смотреть на цель, чтобы лидар видел препятствия по пути
        # Целевой yaw
        yaw_target = math.atan2(de, dn)
        yaw_err = yaw_target - yaw
        # Нормализация угла -pi..pi
        while yaw_err > math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        
        yaw_rate_cmd = np.clip(yaw_err * 1.0, -1.0, 1.0)
        
        # 6. Отправка команды
        await drone.move_by_velocity_body_frame_async(
            v_forward=float(v_fwd_final),
            v_right=float(v_right_final),
            v_down=float(vz_cmd), # В body frame v_down совпадает с world v_down при малых кренах
            duration=float(dt),
            yaw_is_rate=True,
            yaw=float(yaw_rate_cmd)
        )
        
        await asyncio.sleep(dt)

async def hover_and_scan_at_corner(
    drone: Drone,
    pose_latest: PoseLatest,
    acc: PointCloudAccumulator,
    height: float,
    duration_sec: float = 5.0
):
    """Зависает на текущей позиции и сканирует (накапливает точки)"""
    print(f"[SCAN] Сканирование в точке (длительность: {duration_sec}с)...")
    
    # Получаем текущую позицию для удержания
    start_n, start_e = 0.0, 0.0
    pose_msg, _ = pose_latest.snapshot()
    if pose_msg and isinstance(pose_msg, dict):
        pos = pose_msg.get("position", {})
        start_n = float(pos.get("x", 0.0))
        start_e = float(pos.get("y", 0.0))
    
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        # Удерживаем позицию
        await drone.move_to_position_async(north=start_n, east=start_e, down=height, velocity=1.0)
        
        # Вывод прогресса
        if int(time.time() - start_time) % 2 == 0:
             print(f"[SCAN] Накоплено точек: {acc._total_points}")
        
        await asyncio.sleep(0.5)
    
    print("[SCAN] Сканирование завершено.")

async def fly_perimeter_and_scan(
    drone: Drone,
    pose_latest: PoseLatest,
    lidar_latest: LidarLatest,
    acc: PointCloudAccumulator,
    side_length: float = 20.0,
    height: float = -5.0,
    velocity: float = 2.0,
    scan_duration: float = 5.0
):
    """
    Облетает квадрат по периметру, делая сканирование в каждом углу.
    Использует навигацию с избеганием препятствий.
    """
    print(f"[MISSION] Начинаю облет периметра (сторона: {side_length}м, высота: {-height}м)")

    # 1. Получаем начальную позицию
    start_n, start_e = 0.0, 0.0
    got_pose = False
    
    for _ in range(20): 
        pose_msg, _ = pose_latest.snapshot()
        if pose_msg and isinstance(pose_msg, dict):
            pos = pose_msg.get("position", {})
            if pos:
                start_n = float(pos.get("x", 0.0))
                start_e = float(pos.get("y", 0.0))
                got_pose = True
                break
        await asyncio.sleep(0.1)
        
    if not got_pose:
        print("[WARN] Не удалось получить позицию через подписку, пробуем прямой запрос...")
        try:
            start_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
            start_n = float(start_pos["x"])
            start_e = float(start_pos["y"])
        except Exception as e:
            print(f"[ERROR] Не удалось получить начальную позицию дрона: {e}")
            return
    
    corners = [
        (start_n + side_length, start_e),              # Угол 1 (Вперед)
        (start_n + side_length, start_e + side_length), # Угол 2 (Вперед-Вправо)
        (start_n, start_e + side_length),              # Угол 3 (Вправо)
        (start_n, start_e)                             # Угол 4 (Старт - возврат)
    ]
    
    print("[MISSION] Сканирование на стартовой точке...")
    await hover_and_scan_at_corner(drone, pose_latest, acc, height, scan_duration)

    for i, (target_n, target_e) in enumerate(corners, 1):
        print(f"[MISSION] Перелет к углу {i}/4: North={target_n:.1f}, East={target_e:.1f}")
        
        # Летим к углу с избеганием препятствий
        await fly_to_point_with_avoidance(
            drone=drone,
            target_n=target_n,
            target_e=target_e,
            height=height,
            pose_latest=pose_latest,
            lidar_latest=lidar_latest,
            cruise_speed=velocity,
            stop_dist=1.5,      # Считаем прибывшим если ближе 1.5м
            avoid_dist=2.0,     # Начинаем тормозить за 2м
            influence_dist=3.5  # Начинаем отворачивать за 3.5м
        )
        
        # Сканируем
        print(f"[MISSION] Прибыли в район угла {i}. Выполняем сканирование...")
        await hover_and_scan_at_corner(drone, pose_latest, acc, height, scan_duration)

    print("[MISSION] Облет периметра завершен.")

async def main():
    parser = argparse.ArgumentParser(description="ProjectAirSim: Облет периметра и сканирование.")
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--lidar-name", default="lidar1")
    parser.add_argument("--imu-name", default="IMU1")
    parser.add_argument("--side-length", type=float, default=20.0, help="Длина стороны квадрата (м).")
    parser.add_argument("--height", type=float, default=-5.0, help="Высота (отрицательная = вверх).")
    parser.add_argument("--velocity", type=float, default=2.0, help="Скорость полета (м/с).")
    parser.add_argument("--scan-duration", type=float, default=5.0, help="Время сканирования в каждом углу (сек).")
    parser.add_argument("--scene", default="scene_blocks_lidar_mapping.jsonc")
    parser.add_argument("--sim-config-path", default="sim_config")
    
    args = parser.parse_args()

    # --- КОПИЯ SETUP КОДА ИЗ lidar_mapping/cli.py ---
    client = ProjectAirSimClient()
    acc = PointCloudAccumulator(max_points=500_000)
    tmp_cfg_dir: Optional[tempfile.TemporaryDirectory] = None
    lidar_latest = LidarLatest()
    pose_latest = PoseLatest()
    imu_latest = ImuLatest()
    visualizer: Optional[RealtimePointCloudVisualizer] = None
    saved_map_path = None

    try:
        client.connect()
        print("Connected to ProjectAirSim")

        # Load Scene
        example_dir = REPO_ROOT / "client" / "python" / "example_user_scripts"
        original_cwd = os.getcwd()
        os.chdir(str(example_dir))
        try:
            scene_to_load = args.scene
            sim_config_path = args.sim_config_path
            scene_candidate = (Path(sim_config_path) / scene_to_load)
            
            try:
                world = World(client, scene_to_load, delay_after_load_sec=2, sim_config_path=sim_config_path)
            except Exception as e:
                if scene_candidate.exists():
                     print("[WARN] Ошибка загрузки сцены, пробую авто-фикс...")
                     tmp_cfg_dir, fixed_scene, fixed_cfg = _autofix_scene_and_robot_configs(
                         scene_candidate.resolve(), sim_config_path
                     )
                     world = World(client, fixed_scene, delay_after_load_sec=2, sim_config_path=fixed_cfg)
                else:
                    raise e
        finally:
            os.chdir(original_cwd)

        drone = Drone(client, world, args.drone_name)
        path_tracker = PathTracker(world, min_distance=0.1)

        # Запуск визуализатора
        try:
            visualizer = RealtimePointCloudVisualizer(acc=acc, update_interval=0.1)
            visualizer.start()
            print("[VISUALIZER] Визуализация запущена.")
        except Exception as e:
            print(f"[WARN] Без визуализации: {e}")

        # Callbacks
        # Pose
        def _pose_callback(_, pose_msg):
            pose_latest.update(pose_msg)
            if pose_msg and isinstance(pose_msg, dict):
                pos = pose_msg.get("position", {})
                if pos:
                    path_tracker.update_position((float(pos.get("x",0)), float(pos.get("y",0)), float(pos.get("z",0))))
        
        try:
            client.subscribe(drone.robot_info["actual_pose"], _pose_callback)
        except Exception: 
            pass 

        # Lidar
        def _lidar_callback(lidar_data):
             if lidar_data is None: return
             try:
                 import numpy as np
                 pc = lidar_data.get("point_cloud", None)
                 if pc is None: return
                 pts = np.asarray(pc, dtype=np.float32)
                 if pts.size < 3: return
                 pts_body = np.reshape(pts, (int(pts.shape[0] / 3), 3))
                 
                 # Transform
                 pose_msg, _ = pose_latest.snapshot()
                 if pose_msg and isinstance(pose_msg, dict):
                     pos = pose_msg.get("position", {})
                     ori = pose_msg.get("orientation", {})
                     if pos and ori:
                         pts_world = _transform_points_body_to_world(pts_body, pos, ori)
                         acc.add_points(pts_world)
                         lidar_latest.update(pts_body, stamp=time.time())
                     else:
                         acc.add_points(pts_body)
                         lidar_latest.update(pts_body, stamp=time.time())
                 else:
                     acc.add_points(pts_body)
                     lidar_latest.update(pts_body, stamp=time.time())
             except Exception: pass

        try:
            client.subscribe(drone.sensors[args.lidar_name]["lidar"], lambda _, msg: _lidar_callback(msg))
        except Exception as e:
            print(f"Lidar subscribe failed: {e}")

        # --- ЗАПУСК ПОЛЕТА ---
        drone.enable_api_control()
        drone.arm()
        
        print("Взлет...")
        await drone.takeoff_async()
        
        # Подъем на рабочую высоту
        print(f"Набор высоты {-args.height}м...")
        await drone.move_to_position_async(north=0, east=0, down=args.height, velocity=2.0)
        await asyncio.sleep(2.0)

        # Выполнение миссии
        await fly_perimeter_and_scan(
            drone, pose_latest, lidar_latest, acc, 
            side_length=args.side_length, 
            height=args.height, 
            velocity=args.velocity,
            scan_duration=args.scan_duration
        )

        # --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---
        print("[SAVE] Сохранение результатов...")
        final_points = acc.snapshot()
        if final_points is not None and final_points.shape[0] > 0:
            save_point_cloud_to_ply(final_points, filename="result.ply")
            
            # Генерация 2D карты
            drone_pos = None
            pose_msg, _ = pose_latest.snapshot()
            if pose_msg:
                p = pose_msg.get("position", {})
                drone_pos = (float(p.get("x",0)), float(p.get("y",0)), float(p.get("z",-1)))
            
            map_2d, block_mask, flight_mask = _point_cloud_to_2d_map(
                final_points, resolution=0.1, map_size_m=100.0, drone_position=drone_pos
            )
            
            if map_2d is not None:
                os.makedirs("output", exist_ok=True)
                map_path = os.path.abspath("output/slam_map_2d.png")
                _save_map_to_png(map_2d, map_path, blocking_obstacles_mask=block_mask, flight_obstacles_mask=flight_mask)
                print(f"[SUCCESS] Карта сохранена: {map_path}")
        
        print("Посадка...")
        await drone.land_async()
        drone.disarm()
        drone.disable_api_control()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if visualizer:
            print("Визуализация активна. Закройте окно вручную.")
        client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
