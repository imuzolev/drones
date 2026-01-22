"""CLI entry for lidar mapping. This is the original main() moved into a module."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from .bootstrap import ProjectAirSimClient, Drone, World, REPO_ROOT
from .config_autofix import _autofix_scene_and_robot_configs
from .lio import SimpleLIO
from .mapping import PointCloudAccumulator, save_point_cloud_to_ply, _transform_points_body_to_world, _point_cloud_to_2d_map, _save_map_to_png
from .missions import (
    explore_area_reactive,
    explore_forward_only,
    explore_waypoints_sequential,
    explore_map_systematic,
    full_scan_mapping,
    scan_and_navigate_shelves,
    fly_square_by_position,
    fly_square_by_velocity,
    circular_mapping_flight,
    hover_and_collect_slam,
)
from .path_tracker import PathTracker
from .state import LidarLatest, PoseLatest, ImuLatest
from .visualizer import RealtimePointCloudVisualizer

async def main():
    parser = argparse.ArgumentParser(description="ProjectAirSim: SLAM картографирование местности с помощью лидара.")
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--lidar-name", default="lidar1")
    parser.add_argument("--imu-name", default="IMU1", help="Имя IMU сенсора.")
    parser.add_argument("--side-length", type=float, default=10.0)
    parser.add_argument("--height", type=float, default=-5.0, help="Высота в NED (отрицательное значение = вверх).")
    parser.add_argument("--velocity", type=float, default=1.5)
    parser.add_argument("--mission", default="slam", choices=["explore", "square", "shelves", "slam", "systematic", "forward", "waypoints"])
    parser.add_argument("--scene", default="scene_blocks_lidar_mapping.jsonc")
    parser.add_argument("--sim-config-path", default="sim_config")
    parser.add_argument("--acc-max-points", type=int, default=500_000, help="Макс. накопленных точек в памяти.")
    parser.add_argument("--slam-radius", type=float, default=30.0, help="Радиус кругового облета для SLAM (м).")
    parser.add_argument("--slam-circles", type=int, default=2, help="Количество кругов облета для SLAM.")

    # Explore-mode params (simple coverage + reactive lidar avoidance)
    parser.add_argument("--explore-extent-n", type=float, default=80.0, help="Размер области по North (м) от старта.")
    parser.add_argument("--explore-extent-e", type=float, default=80.0, help="Размер области по East (м) от старта.")
    parser.add_argument("--explore-lane-step", type=float, default=10.0, help="Шаг между проходами по East (м).")
    parser.add_argument("--explore-timeout", type=float, default=600.0, help="Лимит времени исследования (сек).")
    parser.add_argument("--ctrl-dt", type=float, default=0.25, help="Шаг управления (сек), меньше = более реактивно.")
    parser.add_argument("--avoid-dist", type=float, default=12.0, help="Дистанция срабатывания уклонения (м). Увеличено для предотвращения касаний ножек.")
    parser.add_argument("--influence-dist", type=float, default=18.0, help="Радиус влияния для отталкивания (м). Увеличено для предотвращения касаний ножек.")
    parser.add_argument("--max-repulse", type=float, default=2.5, help="Максимальная отталкивающая скорость (м/с).")
    parser.add_argument("--max-yaw-rate", type=float, default=1.2, help="Максимальная скорость рыскания (рад/с).")
    parser.add_argument("--arrive-tol", type=float, default=2.0, help="Допуск достижения точки (м).")
    parser.add_argument("--grid-resolution", type=float, default=2.5, help="Разрешение сетки для систематического исследования (м).")
    
    # Shelf detection params
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Размер вокселя для очистки облака точек (м).")
    parser.add_argument("--nb-neighbors", type=int, default=20, help="Количество соседей для удаления выбросов.")
    parser.add_argument("--std-ratio", type=float, default=2.0, help="Стандартное отклонение для удаления выбросов.")
    parser.add_argument("--min-shelf-height", type=float, default=1.0, help="Минимальная высота стеллажа (м).")
    parser.add_argument("--cluster-distance", type=float, default=3.0, help="Расстояние для кластеризации стеллажей (м).")
    parser.add_argument("--start-height", type=float, default=-2.0, help="Начальная высота сканирования (м, NED, отрицательное = вверх).")
    parser.add_argument("--end-height", type=float, default=-8.0, help="Конечная высота сканирования (м, NED, отрицательное = вверх).")
    parser.add_argument("--height-step", type=float, default=1.5, help="Шаг изменения высоты между слоями (м).")
    
    args = parser.parse_args()

    # Create a Project AirSim client
    client = ProjectAirSimClient()

    # Подготовка накопления точек для SLAM
    acc = PointCloudAccumulator(max_points=args.acc_max_points)
    tmp_cfg_dir: Optional[tempfile.TemporaryDirectory] = None
    lidar_latest = LidarLatest()
    pose_latest = PoseLatest()
    imu_latest = ImuLatest()
    lio_slam = SimpleLIO()
    visualizer: Optional[RealtimePointCloudVisualizer] = None
    
    try:
        # Connect to simulation environment
        client.connect()
        print("Connected")
        
        # Change to the example_user_scripts directory where sim_config is located
        example_dir = REPO_ROOT / "client" / "python" / "example_user_scripts"
        original_cwd = os.getcwd()
        os.chdir(str(example_dir))
        
        try:
            scene_to_load = args.scene
            sim_config_path = args.sim_config_path

            # если пользователь оставил дефолт, но файла нет — откатываемся на базовую сцену
            scene_candidate = (Path(sim_config_path) / scene_to_load)
            if not scene_candidate.exists() and scene_to_load == "scene_blocks_lidar_mapping.jsonc":
                scene_to_load = "scene_basic_drone.jsonc"
                scene_candidate = (Path(sim_config_path) / scene_to_load)

            # Create a World object to interact with the sim world and load a scene
            try:
                world = World(
                    client,
                    scene_to_load,
                    delay_after_load_sec=2,
                    sim_config_path=sim_config_path,
                )
            except Exception as e:
                # Попытка авто-фикса конфигов при ошибке валидации схемы
                try:
                    import jsonschema

                    is_validation = isinstance(e, jsonschema.exceptions.ValidationError)
                except Exception:
                    is_validation = False

                if is_validation and scene_candidate.exists():
                    print("[WARN] Конфиг не прошел валидацию схемы. Пробую авто-фикс (временная копия config).")
                    tmp_cfg_dir, fixed_scene_name, fixed_sim_config_path = _autofix_scene_and_robot_configs(
                        scene_path=scene_candidate.resolve(),
                        sim_config_path=sim_config_path,
                    )
                    world = World(
                        client,
                        fixed_scene_name,
                        delay_after_load_sec=2,
                        sim_config_path=fixed_sim_config_path,
                    )
                    # важно: теперь все относительные robot-config пути берутся из temp dir,
                    # а сам drone/sensors корректно создадутся из исправленных config'ов
                else:
                    raise
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
        
        # Create a Drone object to interact with a drone in the loaded sim world
        drone = Drone(client, world, args.drone_name)

        # Создаем трекер пути дрона для отрисовки красной линии траектории
        path_tracker = PathTracker(world, min_distance=0.1)
        
        # Переменная для хранения пути к сохраненной карте
        saved_map_path = None

        # Создаем и запускаем визуализатор в реальном времени
        try:
            visualizer = RealtimePointCloudVisualizer(
                acc=acc,
                update_interval=0.1,  # Обновление каждые 100 мс
                max_display_points=100_000  # Максимум точек для отображения
            )
            visualizer.start()
            print("[VISUALIZER] Визуализация запущена. Окно откроется сразу.")
        except Exception as e:
            print(f"[WARN] Не удалось запустить визуализацию в реальном времени: {e}")
            print("[WARN] Продолжаем работу без визуализации в реальном времени.")

        # Подписка на actual_pose (позиция/ориентация) — для explore без pynng Timeout
        try:
            def _pose_callback(_, pose_msg):
                pose_latest.update(pose_msg)
                # Обновляем путь дрона для отрисовки красной линии траектории
                if pose_msg is not None and isinstance(pose_msg, dict):
                    pos = pose_msg.get("position", {})
                    if pos:
                        n = float(pos.get("x", 0.0))
                        e = float(pos.get("y", 0.0))
                        d = float(pos.get("z", 0.0))
                        path_tracker.update_position((n, e, d))
            
            client.subscribe(
                drone.robot_info["actual_pose"],
                _pose_callback,
            )
            print("[pose] subscribed: actual_pose")
        except Exception as e:
            print(f"[WARN] pose subscribe failed: {e}")

        # Подписка на LIDAR
        _lidar_callback_count = [0]  # Используем список для изменяемого счетчика в замыкании
        
        def _lidar_callback(lidar_data):
            if lidar_data is None:
                return
            try:
                import numpy as np
            except Exception:
                return
            pc = lidar_data.get("point_cloud", None)
            if pc is None:
                return
            pts = np.asarray(pc, dtype=np.float32)
            if pts.size < 3:
                return
            pts_body = np.reshape(pts, (int(pts.shape[0] / 3), 3))
            
            # Трансформируем точки из body frame в world frame (NED)
            # Используем последнюю известную позу дрона
            pose_msg, _pose_ts = pose_latest.snapshot()
            if pose_msg is not None and isinstance(pose_msg, dict):
                pos = pose_msg.get("position", {})
                ori = pose_msg.get("orientation", {})
                if pos and ori:
                    # Трансформируем точки в world frame
                    pts_world = _transform_points_body_to_world(pts_body, pos, ori)
                    acc.add_points(pts_world)
                    # ВАЖНО: Для навигации используем локальные координаты (body frame), 
                    # чтобы правильно определять препятствия относительно дрона
                    lidar_latest.update(pts_body, stamp=time.time())
                else:
                    # Если поза недоступна, накапливаем как есть (будет трансформировано позже)
                    acc.add_points(pts_body)
                    lidar_latest.update(pts_body, stamp=time.time())
            else:
                # Если поза недоступна, накапливаем как есть
                acc.add_points(pts_body)
                lidar_latest.update(pts_body, stamp=time.time())
            
            # Отладочная информация (каждые 10 кадров)
            _lidar_callback_count[0] += 1
            if _lidar_callback_count[0] % 10 == 0:
                total_points = acc._total_points
                snapshot = acc.snapshot(max_points=100)
                snapshot_count = snapshot.shape[0] if snapshot is not None else 0
                print(f"[LIDAR] Получено {pts_body.shape[0]} точек, всего накоплено: {total_points}, снимок: {snapshot_count}")

        try:
            client.subscribe(
                drone.sensors[args.lidar_name]["lidar"],
                lambda _, lidar_msg: _lidar_callback(lidar_msg),
            )
            print(f"[lidar] subscribed: {args.lidar_name}")
        except Exception as e:
            print(f"[WARN] LIDAR subscribe failed '{args.lidar_name}': {e}")

        # Подписка на IMU для LIO-SLAM
        def _imu_callback(imu_data):
            if imu_data is None:
                return
            try:
                # IMU данные могут приходить в разных форматах
                # Пытаемся извлечь данные из сообщения
                imu_dict = None
                
                if isinstance(imu_data, dict):
                    # Если данные уже в виде словаря
                    imu_dict = imu_data
                elif hasattr(imu_data, 'getData'):
                    # Если это объект с методом getData (ImuMessage)
                    try:
                        imu_dict = imu_data.getData()
                    except Exception:
                        pass
                elif isinstance(imu_data, str):
                    # Если данные в виде JSON строки
                    try:
                        import json
                        imu_dict = json.loads(imu_data)
                    except Exception:
                        pass
                
                # Нормализуем структуру данных
                if imu_dict and isinstance(imu_dict, dict):
                    # Проверяем и нормализуем структуру
                    normalized = {}
                    if "orientation" in imu_dict:
                        normalized["orientation"] = imu_dict["orientation"]
                    if "angular_velocity" in imu_dict:
                        normalized["angular_velocity"] = imu_dict["angular_velocity"]
                    if "linear_acceleration" in imu_dict:
                        normalized["linear_acceleration"] = imu_dict["linear_acceleration"]
                    if "time_stamp" in imu_dict:
                        normalized["time_stamp"] = imu_dict["time_stamp"]
                    
                    if normalized:
                        imu_latest.update(normalized)
            except Exception as e:
                # Игнорируем ошибки парсинга, чтобы не прерывать работу
                pass

        try:
            if args.imu_name in drone.sensors:
                client.subscribe(
                    drone.sensors[args.imu_name]["imu_kinematics"],
                    lambda _, imu_msg: _imu_callback(imu_msg),
                )
                print(f"[IMU] subscribed: {args.imu_name}")
                print("[LIO-SLAM] LiDAR + IMU + LIO-SLAM enabled for precise navigation")
            else:
                print(f"[WARN] IMU sensor '{args.imu_name}' not found. Available sensors: {list(drone.sensors.keys())}")
                print("[WARN] LIO-SLAM will work with LiDAR only (IMU unavailable)")
        except Exception as e:
            print(f"[WARN] IMU subscribe failed '{args.imu_name}': {e}")
            print("[WARN] LIO-SLAM will work with LiDAR only (IMU unavailable)")
        
        # Set the drone to be ready to fly
        drone.enable_api_control()
        drone.arm()
        
        print("Taking off...")
        takeoff_task = await drone.takeoff_async()
        await takeoff_task
        
        # Поднимаемся на высоту 5 метров
        target_height = -5.0  # NED: отрицательное значение = вверх
        print(f"[SLAM] Подъем на высоту 5 метров...")
        await drone.move_to_position_async(north=0.0, east=0.0, down=target_height, velocity=2.0)
        await asyncio.sleep(2.0)  # Ждем стабилизации
        
        # Выполняем SLAM картографирование на месте (зависание)
        print("[SLAM] Начинаем SLAM картографирование на месте...")
        slam_duration = 30.0  # Длительность сбора данных в секундах
        await hover_and_collect_slam(
            drone=drone,
            lidar_latest=lidar_latest,
            pose_latest=pose_latest,
            acc=acc,
            height=target_height,
            duration_sec=slam_duration,
        )
        
        # Даем время на финальное накопление данных
        print("[SLAM] Завершаем накопление данных...")
        await asyncio.sleep(2.0)
        
        # Опускаемся
        print("[SLAM] Опускание дрона...")
        await drone.move_to_position_async(north=0.0, east=0.0, down=-1.0, velocity=2.0)
        await asyncio.sleep(2.0)
        
        # Получаем накопленное облако точек
        points_xyz = acc.snapshot()
        if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
            print("[SLAM] ERROR: Не накоплено точек для создания карты!")
        else:
            print(f"[SLAM] Накоплено {points_xyz.shape[0]} точек")
            
            # Получаем текущую позицию дрона для определения критических препятствий
            drone_position = None
            try:
                pose_msg, _ts = pose_latest.snapshot()
                if pose_msg is not None and isinstance(pose_msg, dict):
                    pos = pose_msg.get("position", {})
                    drone_n = float(pos.get("x", 0.0))
                    drone_e = float(pos.get("y", 0.0))
                    drone_z = float(pos.get("z", -1.0))
                    drone_position = (drone_n, drone_e, drone_z)
                    print(f"[SLAM] Позиция дрона для анализа препятствий: ({drone_n:.2f}, {drone_e:.2f}, {drone_z:.2f})")
                else:
                    # Попытка получить позицию напрямую от дрона
                    kin = drone.get_ground_truth_kinematics()
                    pos = kin["pose"]["position"]
                    drone_n = float(pos["x"])
                    drone_e = float(pos["y"])
                    drone_z = float(pos["z"])
                    drone_position = (drone_n, drone_e, drone_z)
                    print(f"[SLAM] Позиция дрона получена напрямую: ({drone_n:.2f}, {drone_e:.2f}, {drone_z:.2f})")
            except Exception as e:
                print(f"[SLAM] WARN: Не удалось получить позицию дрона для анализа препятствий: {e}")
            
            # Получаем avoid_dist из аргументов командной строки
            avoid_dist = getattr(args, 'avoid_dist', 12.0)
            
            # Создаем и сохраняем 2D карту
            print("[SLAM] Создание 2D карты...")
            map_2d, blocking_mask, flight_obstacles_mask = _point_cloud_to_2d_map(
                points_xyz, 
                resolution=0.1, 
                map_size_m=100.0,
                drone_position=drone_position,
                avoid_dist=avoid_dist
            )
            if map_2d is not None:
                # Создаем директорию для сохранения карты, если её нет
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                map_path = os.path.join(output_dir, "slam_map_2d.png")
                map_path_absolute = os.path.abspath(map_path)
                success = _save_map_to_png(
                    map_2d,
                    map_path,
                    colormap="viridis",
                    blocking_obstacles_mask=blocking_mask,
                    flight_obstacles_mask=flight_obstacles_mask,
                )
                if success:
                    saved_map_path = map_path_absolute  # Сохраняем путь для вывода в конце
                    print(f"[SLAM] 2D карта успешно сохранена в {map_path}")
                    print(f"[SLAM] Абсолютный путь к карте: {map_path_absolute}")
                else:
                    print("[SLAM] Ошибка при сохранении 2D карты")
            else:
                print("[SLAM] Не удалось создать 2D карту из облака точек")
            
            # Визуализация в реальном времени уже работает, не нужно вызывать display_point_cloud
            print("[SLAM] Облако точек отображается в окне визуализации в реальном времени.")
            
            # Сохраняем облако точек в корень проекта
            result_path = "result.ply"
            save_point_cloud_to_ply(points_xyz, filename=result_path)
        
        # Продолжаем выполнение только если mission не "slam"
        if args.mission != "slam":
            if args.mission == "square":
                await fly_square_by_position(drone, side_length=args.side_length, height=args.height, velocity=args.velocity)
            elif args.mission == "shelves":
                print("[shelves] starting full scan mapping and shelf navigation")
                await scan_and_navigate_shelves(
                    drone=drone,
                    lidar_latest=lidar_latest,
                    pose_latest=pose_latest,
                    acc=acc,
                    extent_n=args.explore_extent_n,
                    extent_e=args.explore_extent_e,
                    start_height=args.start_height,
                    end_height=args.end_height,
                    height_step=args.height_step,
                    cruise_speed=args.velocity,
                    voxel_size=args.voxel_size,
                    nb_neighbors=args.nb_neighbors,
                    std_ratio=args.std_ratio,
                    min_shelf_height=args.min_shelf_height,
                    cluster_distance=args.cluster_distance,
                    dt=args.ctrl_dt,
                    avoid_dist=args.avoid_dist,
                    influence_dist=args.influence_dist,
                    max_repulse=args.max_repulse,
                    max_yaw_rate=args.max_yaw_rate,
                )
            elif args.mission == "systematic":
                print("[systematic] starting systematic map exploration with obstacle avoidance")
                # Инициализация LIO-SLAM с начальной позицией
                pose_msg, _ts = pose_latest.snapshot()
                if pose_msg is not None and isinstance(pose_msg, dict):
                    pos = pose_msg.get("position", {})
                    ori = pose_msg.get("orientation", {})
                    if pos and ori:
                        lio_slam.position = [
                            float(pos.get("x", 0.0)),
                            float(pos.get("y", 0.0)),
                            float(pos.get("z", 0.0))
                        ]
                        lio_slam.orientation = {
                            "w": float(ori.get("w", 1.0)),
                            "x": float(ori.get("x", 0.0)),
                            "y": float(ori.get("y", 0.0)),
                            "z": float(ori.get("z", 0.0))
                        }
                
                await explore_map_systematic(
                    drone=drone,
                    lidar_latest=lidar_latest,
                    pose_latest=pose_latest,
                    imu_latest=imu_latest,
                    lio_slam=lio_slam,
                    path_tracker=path_tracker,
                    extent_n=args.explore_extent_n,
                    extent_e=args.explore_extent_e,
                    z=args.height,
                    cruise_speed=args.velocity,
                    dt=args.ctrl_dt,
                    arrive_tol=args.arrive_tol,
                    avoid_dist=args.avoid_dist,
                    influence_dist=args.influence_dist,
                    max_repulse=args.max_repulse,
                    max_yaw_rate=args.max_yaw_rate,
                    grid_resolution=args.grid_resolution,
                    total_timeout_sec=args.explore_timeout,
                )
            elif args.mission == "forward":
                print("[forward] starting forward-only exploration with SLAM mapping and obstacle avoidance")
                # Инициализация LIO-SLAM с начальной позицией
                pose_msg, _ts = pose_latest.snapshot()
                if pose_msg is not None and isinstance(pose_msg, dict):
                    pos = pose_msg.get("position", {})
                    ori = pose_msg.get("orientation", {})
                    if pos and ori:
                        lio_slam.position = [
                            float(pos.get("x", 0.0)),
                            float(pos.get("y", 0.0)),
                            float(pos.get("z", 0.0))
                        ]
                        lio_slam.orientation = {
                            "w": float(ori.get("w", 1.0)),
                            "x": float(ori.get("x", 0.0)),
                            "y": float(ori.get("y", 0.0)),
                            "z": float(ori.get("z", 0.0))
                        }
                
                await explore_forward_only(
                    drone=drone,
                    lidar_latest=lidar_latest,
                    pose_latest=pose_latest,
                    imu_latest=imu_latest,
                    lio_slam=lio_slam,
                    extent_n=args.explore_extent_n,
                    extent_e=args.explore_extent_e,
                    z=args.height,
                    cruise_speed=args.velocity,
                    dt=args.ctrl_dt,
                    avoid_dist=args.avoid_dist,
                    max_yaw_rate=args.max_yaw_rate,
                    total_timeout_sec=args.explore_timeout,
                )
            elif args.mission == "waypoints":
                print("[waypoints] starting sequential waypoint navigation A → B → C → D → E → A")
                # Инициализация LIO-SLAM с начальной позицией
                pose_msg, _ts = pose_latest.snapshot()
                if pose_msg is not None and isinstance(pose_msg, dict):
                    pos = pose_msg.get("position", {})
                    ori = pose_msg.get("orientation", {})
                    if pos and ori:
                        lio_slam.position = [
                            float(pos.get("x", 0.0)),
                            float(pos.get("y", 0.0)),
                            float(pos.get("z", 0.0))
                        ]
                        lio_slam.orientation = {
                            "w": float(ori.get("w", 1.0)),
                            "x": float(ori.get("x", 0.0)),
                            "y": float(ori.get("y", 0.0)),
                            "z": float(ori.get("z", 0.0))
                        }
                
                await explore_waypoints_sequential(
                    drone=drone,
                    lidar_latest=lidar_latest,
                    pose_latest=pose_latest,
                    imu_latest=imu_latest,
                    lio_slam=lio_slam,
                    extent_n=args.explore_extent_n,
                    extent_e=args.explore_extent_e,
                    z=args.height,
                    cruise_speed=args.velocity,
                    dt=args.ctrl_dt,
                    avoid_dist=args.avoid_dist,
                    max_yaw_rate=args.max_yaw_rate,
                    total_timeout_sec=args.explore_timeout,
                )
            else:
                print("[explore] starting reactive exploration + obstacle avoidance with LIO-SLAM")
                # Инициализация LIO-SLAM с начальной позицией
                pose_msg, _ts = pose_latest.snapshot()
                if pose_msg is not None and isinstance(pose_msg, dict):
                    pos = pose_msg.get("position", {})
                    ori = pose_msg.get("orientation", {})
                    if pos and ori:
                        lio_slam.position = [
                            float(pos.get("x", 0.0)),
                            float(pos.get("y", 0.0)),
                            float(pos.get("z", 0.0))
                        ]
                        lio_slam.orientation = {
                            "w": float(ori.get("w", 1.0)),
                            "x": float(ori.get("x", 0.0)),
                            "y": float(ori.get("y", 0.0)),
                            "z": float(ori.get("z", 0.0))
                        }
                
                await explore_area_reactive(
                    drone=drone,
                    lidar_latest=lidar_latest,
                    pose_latest=pose_latest,
                    imu_latest=imu_latest,
                    lio_slam=lio_slam,
                    extent_n=args.explore_extent_n,
                    extent_e=args.explore_extent_e,
                    z=args.height,
                    cruise_speed=args.velocity,
                    dt=args.ctrl_dt,
                    arrive_tol=args.arrive_tol,
                    avoid_dist=args.avoid_dist,
                    influence_dist=args.influence_dist,
                    max_repulse=args.max_repulse,
                    max_yaw_rate=args.max_yaw_rate,
                    total_timeout_sec=args.explore_timeout,
                )
        
        # Сохраняем финальное облако точек перед посадкой
        print("[SAVE] Сохранение финального облака точек...")
        final_points = acc.snapshot()
        if final_points is not None and getattr(final_points, "size", 0) > 0:
            result_path = "result.ply"
            save_point_cloud_to_ply(final_points, filename=result_path)
            
            # Если карта еще не создана, создаем её из финального облака точек
            if saved_map_path is None:
                print("[SAVE] Создание финальной 2D карты...")
                try:
                    # Получаем позицию дрона
                    drone_position = None
                    try:
                        pose_msg, _ts = pose_latest.snapshot()
                        if pose_msg is not None and isinstance(pose_msg, dict):
                            pos = pose_msg.get("position", {})
                            if pos:
                                drone_n = float(pos.get("x", 0.0))
                                drone_e = float(pos.get("y", 0.0))
                                drone_z = float(pos.get("z", -1.0))
                                drone_position = (drone_n, drone_e, drone_z)
                    except Exception:
                        pass
                    
                    avoid_dist = getattr(args, 'avoid_dist', 12.0)
                    map_2d, blocking_mask, flight_obstacles_mask = _point_cloud_to_2d_map(
                        final_points,
                        resolution=0.1,
                        map_size_m=100.0,
                        drone_position=drone_position,
                        avoid_dist=avoid_dist
                    )
                    if map_2d is not None:
                        output_dir = "output"
                        os.makedirs(output_dir, exist_ok=True)
                        map_path = os.path.join(output_dir, "slam_map_2d.png")
                        map_path_absolute = os.path.abspath(map_path)
                        success = _save_map_to_png(
                            map_2d,
                            map_path,
                            colormap="viridis",
                            blocking_obstacles_mask=blocking_mask,
                            flight_obstacles_mask=flight_obstacles_mask,
                        )
                        if success:
                            saved_map_path = map_path_absolute
                            print(f"[SAVE] Финальная 2D карта сохранена в {map_path}")
                except Exception as e:
                    print(f"[WARN] Не удалось создать финальную карту: {e}")
        else:
            print("[WARN] Финальное облако точек пустое, сохранение пропущено")
        
        print("Landing...")
        land_task = await drone.land_async()
        await land_task
        
        # Shut down the drone
        drone.disarm()
        drone.disable_api_control()
        
        print("[INFO] Миссия завершена. Окно визуализации остается открытым.")
        print("[INFO] Закройте окно вручную, когда закончите просмотр.")
        
        # Выводим путь к созданной карте в конце
        if saved_map_path is not None:
            print("=" * 80)
            print(f"[FINAL] Путь к созданной карте: {saved_map_path}")
            print("=" * 80)
        else:
            print("[FINAL] Карта не была создана.")
        
    except Exception as err:
        print(f"Exception occurred: {err}")
        import traceback
        traceback.print_exc()
    finally:
        # Визуализатор НЕ останавливаем - окно остается открытым для просмотра
        # Пользователь может закрыть окно вручную когда захочет
        if visualizer is not None:
            print("[INFO] Визуализатор продолжает работать. Закройте окно вручную для завершения.")
        
        if tmp_cfg_dir is not None:
            with contextlib.suppress(Exception):
                tmp_cfg_dir.cleanup()
        # Always disconnect from the simulation environment
        client.disconnect()

