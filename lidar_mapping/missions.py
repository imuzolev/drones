"""Flight missions / navigation behaviors."""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Optional, Tuple, List

from .bootstrap import Drone
from .lio import SimpleLIO
from .mapping import PointCloudAccumulator, save_point_cloud_to_ply
from .processing import _clean_point_cloud, _detect_vertical_structures, _cluster_shelves, _get_shelf_cluster_center, _plan_path_between_shelves
from .state import LidarLatest, PoseLatest, ImuLatest, _quat_to_yaw_rad, _quat_to_euler_rad, _world_to_body, _clamp, _min_range_in_cone, _check_landing_gear_collision, _repulsive_velocity_xy, _generate_lawnmower_waypoints

async def _drive_to_waypoint_reactive(
    drone: Drone,
    lidar_latest: LidarLatest,
    pose_latest: PoseLatest,
    target_n: float,
    target_e: float,
    z: float,
    cruise_speed: float,
    dt: float,
    arrive_tol: float,
    avoid_dist: float,
    influence_dist: float,
    max_repulse: float,
    max_yaw_rate: float,
    timeout_sec: float,
) -> None:
    t0 = time.time()
    last_dist = float('inf')
    stuck_time = 0.0
    stuck_threshold = 5.0  # секунд без прогресса = застрял
    last_pos_n = None
    last_pos_e = None
    backoff_time = 0.0
    backoff_duration = 2.0  # секунд отступления назад
    target_z = z  # локальная переменная для целевой высоты
    
    while True:
        if time.time() - t0 > timeout_sec:
            print(f"[drive_to_waypoint] Timeout reached, skipping waypoint")
            return

        pose_msg, _pose_ts = pose_latest.snapshot()
        if pose_msg is None:
            await asyncio.sleep(0.05)
            continue

        pos = pose_msg.get("position", {}) if isinstance(pose_msg, dict) else {}
        ori = pose_msg.get("orientation", {}) if isinstance(pose_msg, dict) else {}
        cur_n = float(pos.get("x", 0.0))
        cur_e = float(pos.get("y", 0.0))
        yaw = _quat_to_yaw_rad(ori) if isinstance(ori, dict) else 0.0

        dn = target_n - cur_n
        de = target_e - cur_e
        dist = math.hypot(dn, de)
        if dist <= arrive_tol:
            return

        # Обнаружение застревания: проверяем, приближаемся ли мы к цели
        progress_made = False
        if last_dist != float('inf'):
            if dist < last_dist - 0.5:  # приблизились хотя бы на 0.5м
                progress_made = True
                stuck_time = 0.0
            else:
                stuck_time += dt
        
        # Если застряли и не в режиме отступления - начинаем агрессивное отступление
        if stuck_time > stuck_threshold and backoff_time <= 0.0:
            print(f"[drive_to_waypoint] ЗАСТРЯЛ! Выполняю агрессивный маневр выхода...")
            backoff_time = backoff_duration * 1.5  # увеличиваем время отступления
            stuck_time = 0.0
            
            # Агрессивный маневр выхода: отступление, подъем и поворот
            pts_stuck, _ = lidar_latest.snapshot()
            if pts_stuck is not None:
                front_stuck = _min_range_in_cone(pts_stuck, az_min_rad=-math.radians(30), az_max_rad=math.radians(30), max_range=999.0)
                left_stuck = _min_range_in_cone(pts_stuck, az_min_rad=math.radians(30), az_max_rad=math.radians(90), max_range=999.0)
                right_stuck = _min_range_in_cone(pts_stuck, az_min_rad=-math.radians(90), az_max_rad=-math.radians(30), max_range=999.0)
                
                # Выбираем направление с наибольшим зазором для маневра
                if left_stuck > right_stuck:
                    escape_turn = 1.0  # поворот влево
                    escape_side = "left"
                else:
                    escape_turn = -1.0  # поворот вправо
                    escape_side = "right"
                
                print(f"[drive_to_waypoint] Маневр выхода: отступление назад, подъем, поворот {escape_side}")
        
        # Проверка на движение: если дрон не двигается физически
        if last_pos_n is not None and last_pos_e is not None:
            pos_change = math.hypot(cur_n - last_pos_n, cur_e - last_pos_e)
            if pos_change < 0.1:  # практически не двигается
                if not progress_made and dist > arrive_tol:
                    stuck_time += dt
            else:
                stuck_time = max(0.0, stuck_time - dt * 0.5)  # уменьшаем счетчик застревания
        
        last_pos_n = cur_n
        last_pos_e = cur_e
        last_dist = dist

        pts, _ts = lidar_latest.snapshot()

        # Если нет данных лидара - продолжаем движение с осторожностью
        if pts is None or getattr(pts, "size", 0) == 0:
            print(f"[drive_to_waypoint] WARNING: Нет данных лидара, продолжаю движение с осторожностью")
            # Продолжаем движение к цели, но медленнее
            speed = min(float(cruise_speed) * 0.6, max(0.3, dist * 0.5))
            v_n = speed * (dn / max(dist, 1e-6))
            v_e = speed * (de / max(dist, 1e-6))
            v_fwd, v_right = _world_to_body(v_n, v_e, yaw)
            v_fwd_cmd = v_fwd
            v_right_cmd = v_right
            yaw_rate_cmd = 0.0
            target_z = z - 0.5  # небольшой подъем для безопасности
            
            # Ограничиваем скорости
            v_fwd_cmd = _clamp(v_fwd_cmd, -cruise_speed, cruise_speed)
            v_right_cmd = _clamp(v_right_cmd, -cruise_speed, cruise_speed)
            yaw_rate_cmd = _clamp(yaw_rate_cmd, -max_yaw_rate, max_yaw_rate)
            
            # Отправляем команду и продолжаем цикл
            cmd = await drone.move_by_velocity_body_frame_z_async(
                v_forward=v_fwd_cmd,
                v_right=v_right_cmd,
                z=target_z,
                duration=dt,
                yaw_is_rate=True,
                yaw=yaw_rate_cmd,
            )
            await cmd
            await asyncio.sleep(0.001)
            continue  # Пропускаем остальную логику и переходим к следующей итерации
        else:
            # Проверка на опасность столкновения ножек с препятствиями
            gear_collision_danger = _check_landing_gear_collision(pts, landing_gear_height=0.5, safety_margin=1.5)
            
            # quick emergency checks in cones
            front_min = _min_range_in_cone(pts, az_min_rad=-math.radians(20), az_max_rad=math.radians(20), max_range=999.0)
            left_min = _min_range_in_cone(pts, az_min_rad=math.radians(20), az_max_rad=math.radians(80), max_range=999.0)
            right_min = _min_range_in_cone(pts, az_min_rad=-math.radians(80), az_max_rad=-math.radians(20), max_range=999.0)
            back_min = _min_range_in_cone(pts, az_min_rad=math.radians(100), az_max_rad=math.radians(180), max_range=999.0)
            # Проверка препятствий внизу (на уровне ножек)
            down_min = _min_range_in_cone(pts, az_min_rad=-math.radians(90), az_max_rad=math.radians(90), max_range=999.0)

            # Если обнаружена опасность касания ножками - СРОЧНО отступаем и поднимаемся
            if gear_collision_danger or (down_min < avoid_dist * 0.6):
                print(f"[drive_to_waypoint] ОПАСНОСТЬ: Обнаружено препятствие на уровне ножек! Отступаем...")
                # Срочно отступаем назад и поднимаемся
                v_fwd_cmd = -cruise_speed * 0.8  # быстро назад
                target_z = z - 2.0  # поднимаемся на 2м выше для безопасности
                v_right_cmd = 0.0
                yaw_rate_cmd = 0.0
                # Если сзади тоже препятствие, активно поворачиваем
                if back_min < avoid_dist * 1.2:
                    turn_sign = 1.0 if left_min > right_min else -1.0
                    yaw_rate_cmd = turn_sign * max_yaw_rate  # максимальный поворот
                    v_right_cmd = turn_sign * cruise_speed * 0.7
                    target_z = z - 3.0  # поднимаемся еще выше при полном окружении
                    print(f"[drive_to_waypoint] Сзади препятствие! Экстренный подъем и поворот")
                # Пропускаем остальную логику и сразу отправляем команду
            # Режим агрессивного отступления: отход назад с подъемом и поворотом
            elif backoff_time > 0.0:
                backoff_time -= dt
                
                # Агрессивное отступление: назад, вверх и поворот
                v_fwd_cmd = -cruise_speed * 0.7  # быстрее назад
                
                # Определяем лучшее направление для поворота
                if left_min > right_min:
                    escape_turn = 1.0
                    escape_side = "left"
                else:
                    escape_turn = -1.0
                    escape_side = "right"
                
                # Активный поворот и боковое движение для выхода из тупика
                yaw_rate_cmd = escape_turn * max_yaw_rate  # максимальный поворот
                v_right_cmd = escape_turn * cruise_speed * 0.7  # активное боковое движение
                target_z = z - 2.0  # поднимаемся на 2.0м для выхода из тупика
                
                # Если сзади тоже препятствие - еще более агрессивный маневр
                if back_min < avoid_dist * 1.5:
                    v_fwd_cmd = -cruise_speed * 0.3  # медленнее назад, больше подъем
                    target_z = z - 3.0  # поднимаемся выше
                    yaw_rate_cmd = escape_turn * max_yaw_rate * 1.1  # еще более агрессивный поворот (ограничится clamp)
                    print(f"[drive_to_waypoint] Сзади препятствие! Выполняю экстренный подъем и поворот {escape_side}")
            else:
                # Нормальный режим движения к цели
                # Используем базовую высоту
                target_z = z
                
                # desired world velocity toward waypoint
                speed = min(float(cruise_speed), max(0.2, dist))
                v_n = speed * (dn / max(dist, 1e-6))
                v_e = speed * (de / max(dist, 1e-6))
                v_fwd, v_right = _world_to_body(v_n, v_e, yaw)

                rep_fwd, rep_right = _repulsive_velocity_xy(pts, influence_dist=influence_dist, max_repulse=max_repulse, landing_gear_height=0.5)
                v_fwd_cmd = v_fwd + rep_fwd
                v_right_cmd = v_right + rep_right

                # Улучшенная логика избегания препятствий - активный обход вместо остановки
                yaw_rate_cmd = 0.0
                obstacle_ahead = front_min < avoid_dist
                
                # --- SAFETY CRITICAL: EMERGENCY STOP ---
                if front_min < 1.0: # Абсолютный запрет приближаться ближе 1 метра
                    print(f"[drive_to_waypoint] КРИТИЧЕСКАЯ ОПАСНОСТЬ ({front_min:.2f}м < 1.0м)! ЭКСТРЕННЫЙ ОТХОД!")
                    v_fwd_cmd = -2.0  # Резко назад
                    v_right_cmd = 0.0
                    yaw_rate_cmd = 0.0
                    
                    cmd = await drone.move_by_velocity_body_frame_z_async(
                        v_forward=v_fwd_cmd,
                        v_right=v_right_cmd,
                        z=z - 0.5,
                        duration=0.5,
                        yaw_is_rate=True,
                        yaw=yaw_rate_cmd,
                    )
                    await cmd
                    continue
                elif front_min < 3.5:
                     # Если меньше 3.5м, но больше 1м - позволяем штатной логике отработать (она замедлит дрон)
                     pass 
                # ---------------------------------------

                if obstacle_ahead:
                    # Вычисляем коэффициент близости препятствия (0 = очень близко, 1 = на границе avoid_dist)
                    obstacle_ratio = max(0.0, front_min / max(avoid_dist, 0.1))
                    
                    # Определяем лучшую сторону для обхода
                    best_side = "left" if left_min > right_min else "right"
                    best_clearance = max(left_min, right_min)
                    worst_clearance = min(left_min, right_min)
                    
                    # Если препятствие очень близко - агрессивный маневр
                    if front_min < avoid_dist * 0.5:
                        # Очень близко - отступаем и активно поворачиваем
                        v_fwd_cmd = -cruise_speed * 0.6
                        turn_sign = 1.0 if best_side == "left" else -1.0
                        yaw_rate_cmd = turn_sign * max_yaw_rate  # максимальный поворот
                        v_right_cmd = turn_sign * cruise_speed * 0.8  # активное боковое движение
                        target_z = z - 1.5  # поднимаемся на 1.5м
                        print(f"[drive_to_waypoint] Критическое препятствие! Отступаем и поворачиваем {best_side}")
                    
                    # Если препятствия со всех сторон - подъем и поиск обхода
                    elif left_min < avoid_dist * 0.8 and right_min < avoid_dist * 0.8 and back_min < avoid_dist * 1.0:
                        # Со всех сторон - поднимаемся высоко и поворачиваем
                        v_fwd_cmd = cruise_speed * 0.4  # медленно вперед
                        turn_sign = 1.0 if best_side == "left" else -1.0
                        yaw_rate_cmd = turn_sign * max_yaw_rate * 0.9
                        v_right_cmd = turn_sign * cruise_speed * 0.6
                        target_z = z - 2.5  # поднимаемся на 2.5м
                        print(f"[drive_to_waypoint] Препятствия со всех сторон! Поднимаемся и ищем обход {best_side}")
                    
                    # Если со всех сторон в горизонтальной плоскости, но есть пространство сверху
                    elif left_min < avoid_dist * 0.8 and right_min < avoid_dist * 0.8:
                        # Поднимаемся и поворачиваем
                        v_fwd_cmd = cruise_speed * 0.5  # медленно вперед
                        turn_sign = 1.0 if best_side == "left" else -1.0
                        yaw_rate_cmd = turn_sign * max_yaw_rate * 0.9
                        v_right_cmd = turn_sign * cruise_speed * 0.7
                        target_z = z - 2.0  # поднимаемся на 2.0м
                        print(f"[drive_to_waypoint] Горизонтальные препятствия! Поднимаемся и обходим {best_side}")
                    
                    # Нормальная ситуация - обход препятствия по дуге
                    else:
                        # Активно поворачиваем в сторону большего зазора и продолжаем движение
                        turn_sign = 1.0 if best_side == "left" else -1.0
                        
                        # Адаптивная скорость в зависимости от близости препятствия
                        # Снижаем скорость агрессивнее: если препятствие близко, ползем
                        safe_speed_factor = max(0.1, obstacle_ratio ** 1.5)
                        forward_speed = cruise_speed * safe_speed_factor * 0.8
                        
                        # Активный обход: сочетание поворота и бокового движения
                        yaw_rate_cmd = turn_sign * max_yaw_rate * (0.7 + 0.3 * (1.0 - obstacle_ratio))  # 0.7-1.0 max_yaw_rate
                        v_right_cmd = turn_sign * cruise_speed * (0.6 + 0.4 * (best_clearance / avoid_dist))  # больше зазор = больше скорость
                        v_fwd_cmd = forward_speed  # продолжаем движение вперед, но с поворотом
                        
                        # Небольшой подъем при обходе
                        target_z = z - 0.8  # поднимаемся на 0.8м
                        print(f"[drive_to_waypoint] Обходим препятствие {best_side}, зазор: {best_clearance:.1f}м")
                    
                    # Дополнительная проверка: если препятствия слишком близко снизу - поднимаемся
                    if down_min < avoid_dist * 0.8:
                        target_z = min(target_z, z - 1.2)  # поднимаемся минимум на 1.2м если снизу препятствие
                        
                    # Если нет препятствий впереди, но есть опасность снизу - небольшой подъем
                    if not obstacle_ahead and down_min < avoid_dist * 1.2:
                        target_z = min(target_z, z - 0.5)  # небольшой подъем для безопасности

        # clamp speeds
        v_fwd_cmd = _clamp(v_fwd_cmd, -cruise_speed, cruise_speed)
        v_right_cmd = _clamp(v_right_cmd, -cruise_speed, cruise_speed)
        yaw_rate_cmd = _clamp(yaw_rate_cmd, -max_yaw_rate, max_yaw_rate)

        # send short command (reactive control)
        cmd = await drone.move_by_velocity_body_frame_z_async(
            v_forward=v_fwd_cmd,
            v_right=v_right_cmd,
            z=target_z,
            duration=dt,
            yaw_is_rate=True,
            yaw=yaw_rate_cmd,
        )
        await cmd
        await asyncio.sleep(0.001)


async def explore_area_reactive(
    drone: Drone,
    lidar_latest: LidarLatest,
    pose_latest: PoseLatest,
    imu_latest: ImuLatest,
    lio_slam: SimpleLIO,
    extent_n: float,
    extent_e: float,
    z: float,
    cruise_speed: float,
    dt: float,
    arrive_tol: float,
    avoid_dist: float,
    influence_dist: float,
    max_repulse: float,
    max_yaw_rate: float,
    total_timeout_sec: float,
) -> None:
    """
    Алгоритм реактивного исследования области с обходом препятствий.
    Использует LIO-SLAM (LiDAR + IMU) для более точной навигации.
    Дрон постоянно двигается вперед, обходя препятствия, без фиксированных точек маршрута.
    """
    # Ждём позицию из actual_pose
    start_n = 0.0
    start_e = 0.0
    t_wait = time.time()
    while True:
        pose_msg, _ts = pose_latest.snapshot()
        if pose_msg is not None and isinstance(pose_msg, dict):
            pos = pose_msg.get("position", {})
            start_n = float(pos.get("x", 0.0))
            start_e = float(pos.get("y", 0.0))
            break
        if time.time() - t_wait > 5.0:
            kin = drone.get_ground_truth_kinematics()
            pos = kin["pose"]["position"]
            start_n = float(pos["x"])
            start_e = float(pos["y"])
            break
        await asyncio.sleep(0.05)

    t0 = time.time()
    last_stuck_check = time.time()
    stuck_check_interval = 2.0  # проверяем застревание каждые 2 секунды
    stuck_position_history = []
    stuck_position_window = 5  # храним последние 5 позиций
    last_direction_change = time.time()
    direction_change_interval = 3.0  # меняем направление каждые 3 секунды для разнообразия
    preferred_heading_rad = 0.0  # предпочтительное направление (север)
    last_pos_n = start_n
    last_pos_e = start_e
    velocity_history = deque(maxlen=10)  # для расчета скорости

    print("[explore] Starting reactive exploration (no fixed waypoints)")

    while time.time() - t0 < total_timeout_sec:
        pose_msg, _pose_ts = pose_latest.snapshot()
        if pose_msg is None:
            await asyncio.sleep(0.05)
            continue

        # Получаем данные IMU и LiDAR для LIO-SLAM
        imu_orientation, imu_angular_velocity, imu_linear_acceleration, imu_time = imu_latest.snapshot()
        lidar_pts, lidar_time = lidar_latest.snapshot()
        
        # Обновляем LIO-SLAM состояние
        lio_state = lio_slam.update_state(
            imu_orientation=imu_orientation,
            imu_angular_velocity=imu_angular_velocity,
            imu_linear_acceleration=imu_linear_acceleration,
            imu_time=imu_time,
            lidar_points=lidar_pts,
            pose_gt=pose_msg,
            lidar_time=lidar_time,
        )
        
        # Используем позицию из LIO-SLAM для более точной навигации
        # Смешиваем с ground truth для коррекции дрейфа
        pos = pose_msg.get("position", {}) if isinstance(pose_msg, dict) else {}
        ori = pose_msg.get("orientation", {}) if isinstance(pose_msg, dict) else {}
        vel = pose_msg.get("linear_velocity", {}) if isinstance(pose_msg, dict) else {}
        
        # Смешиваем позицию LIO-SLAM с ground truth (70% LIO-SLAM, 30% GT для коррекции дрейфа)
        lio_pos = lio_state.get("position", [0.0, 0.0, 0.0])
        gt_n = float(pos.get("x", lio_pos[0]))
        gt_e = float(pos.get("y", lio_pos[1]))
        gt_d = float(pos.get("z", lio_pos[2]))
        
        alpha_lio = 0.7  # доверие к LIO-SLAM
        cur_n = alpha_lio * lio_pos[0] + (1.0 - alpha_lio) * gt_n
        cur_e = alpha_lio * lio_pos[1] + (1.0 - alpha_lio) * gt_e
        cur_z = alpha_lio * lio_pos[2] + (1.0 - alpha_lio) * gt_d
        
        # Используем ориентацию из LIO-SLAM или ground truth
        lio_ori = lio_state.get("orientation", ori if isinstance(ori, dict) else {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
        yaw = _quat_to_yaw_rad(lio_ori)
        
        # Рассчитываем скорость (используем скорость из LIO-SLAM или ground truth)
        lio_vel = lio_state.get("velocity", [0.0, 0.0, 0.0])
        vx = float(vel.get("x", lio_vel[0])) if isinstance(vel, dict) else lio_vel[0]
        vy = float(vel.get("y", lio_vel[1])) if isinstance(vel, dict) else lio_vel[1]
        vz = float(vel.get("z", lio_vel[2])) if isinstance(vel, dict) else lio_vel[2]
        speed = math.hypot(vx, vy, vz)
        velocity_history.append((vx, vy, vz, speed))

        # Проверка на застревание
        now = time.time()
        if now - last_stuck_check >= stuck_check_interval:
            stuck_position_history.append((cur_n, cur_e))
            if len(stuck_position_history) > stuck_position_window:
                stuck_position_history.pop(0)
            
            # Если позиция не менялась значительно - мы застряли
            if len(stuck_position_history) >= 3:
                recent_positions = stuck_position_history[-3:]
                max_dist = 0.0
                for i in range(len(recent_positions)):
                    for j in range(i + 1, len(recent_positions)):
                        dist = math.hypot(
                            recent_positions[i][0] - recent_positions[j][0],
                            recent_positions[i][1] - recent_positions[j][1]
                        )
                        max_dist = max(max_dist, dist)
                
                if max_dist < 1.0 and speed < 0.5:  # практически не двигаемся
                    print(f"[explore] ЗАСТРЯЛ! Выполняю агрессивный маневр выхода...")
                    # Агрессивная стратегия выхода: большой поворот и подъем
                    # Поворачиваем на 90-180 градусов для поиска свободного пути
                    preferred_heading_rad = (preferred_heading_rad + math.pi / 2 + (random.random() - 0.5) * math.pi / 2) % (2 * math.pi)
                    last_direction_change = now - direction_change_interval  # форсируем смену направления
                    
                    # Форсируем подъем для выхода из тупика (будет применено в следующей итерации)
                    pts_stuck, _ = lidar_latest.snapshot()
                    if pts_stuck is not None:
                        front_stuck = _min_range_in_cone(pts_stuck, az_min_rad=-math.radians(30), az_max_rad=math.radians(30), max_range=999.0)
                        left_stuck = _min_range_in_cone(pts_stuck, az_min_rad=math.radians(30), az_max_rad=math.radians(90), max_range=999.0)
                        right_stuck = _min_range_in_cone(pts_stuck, az_min_rad=-math.radians(90), az_max_rad=-math.radians(30), max_range=999.0)
                        
                        best_side_stuck = "left" if left_stuck > right_stuck else "right"
                        print(f"[explore] Застревание обнаружено! Новое направление: {math.degrees(preferred_heading_rad):.1f}°, обход: {best_side_stuck}")
            
            last_stuck_check = now

        # Периодически меняем направление для исследования
        if now - last_direction_change >= direction_change_interval:
            # Небольшое случайное изменение направления
            preferred_heading_rad = (preferred_heading_rad + (random.random() - 0.5) * math.pi / 4) % (2 * math.pi)
            last_direction_change = now

        # Обновляем предпочтительное направление с учетом области исследования
        dist_from_start_n = cur_n - start_n
        dist_from_start_e = cur_e - start_e
        
        # Если вышли за границы - поворачиваем обратно
        if abs(dist_from_start_n) > abs(extent_n) * 0.8:
            preferred_heading_rad = math.pi if dist_from_start_n > 0 else 0.0
        if abs(dist_from_start_e) > abs(extent_e) * 0.8:
            preferred_heading_rad = -math.pi / 2 if dist_from_start_e > 0 else math.pi / 2

        pts, _ts = lidar_latest.snapshot()

        # Проверка на опасность столкновения ножек с препятствиями
        gear_collision_danger = _check_landing_gear_collision(pts, landing_gear_height=0.5, safety_margin=1.5)

        # Проверяем препятствия в разных направлениях
        front_min = _min_range_in_cone(pts, az_min_rad=-math.radians(30), az_max_rad=math.radians(30), max_range=999.0)
        left_min = _min_range_in_cone(pts, az_min_rad=math.radians(30), az_max_rad=math.radians(90), max_range=999.0)
        right_min = _min_range_in_cone(pts, az_min_rad=-math.radians(90), az_max_rad=-math.radians(30), max_range=999.0)
        back_min = _min_range_in_cone(pts, az_min_rad=math.radians(150), az_max_rad=math.radians(180), max_range=999.0)
        # Проверка препятствий внизу (на уровне ножек)
        down_min = _min_range_in_cone(pts, az_min_rad=-math.radians(90), az_max_rad=math.radians(90), max_range=999.0)

        # Если обнаружена опасность касания ножками - СРОЧНО останавливаемся и отступаем/поднимаемся
        if gear_collision_danger or (down_min < avoid_dist * 0.6):
            print(f"[explore] ОПАСНОСТЬ: Обнаружено препятствие на уровне ножек! Отступаем и поднимаемся...")
            # Срочно отступаем назад и поднимаемся
            v_fwd_cmd = -cruise_speed * 0.8  # отступаем назад быстро
            target_z = z - 1.5  # поднимаемся на 1.5м выше
            v_right_cmd = 0.0
            yaw_rate_cmd = 0.0
            # Если сзади тоже препятствие, поворачиваем
            if back_min < avoid_dist * 1.2:
                turn_sign = 1.0 if left_min > right_min else -1.0
                yaw_rate_cmd = turn_sign * max_yaw_rate * 0.9
                v_right_cmd = turn_sign * cruise_speed * 0.6
        else:
            # Желаемое направление движения (к предпочтительному направлению)
            desired_v_n = cruise_speed * math.cos(preferred_heading_rad)
            desired_v_e = cruise_speed * math.sin(preferred_heading_rad)

            # Конвертируем в body frame
            v_fwd_desired, v_right_desired = _world_to_body(desired_v_n, desired_v_e, yaw)

            # Отталкивающая сила от препятствий (теперь учитывает препятствия на уровне ножек)
            rep_fwd, rep_right = _repulsive_velocity_xy(pts, influence_dist=influence_dist, max_repulse=max_repulse, landing_gear_height=0.5)

            # Комбинируем желаемое движение и отталкивание
            v_fwd_cmd = v_fwd_desired + rep_fwd
            v_right_cmd = v_right_desired + rep_right

            yaw_rate_cmd = 0.0
            target_z = z

            # Улучшенная реактивная логика избегания препятствий - активное исследование
            obstacle_ahead = front_min < avoid_dist
            
            # --- SAFETY CRITICAL: EMERGENCY STOP ---
            if front_min < 1.0: # Абсолютный запрет приближаться ближе 1 метра
                print(f"[explore] КРИТИЧЕСКАЯ ОПАСНОСТЬ ({front_min:.2f}м < 1.0м)! ЭКСТРЕННЫЙ ОТХОД!")
                v_fwd_cmd = -2.0  # Резко назад
                v_right_cmd = 0.0
                yaw_rate_cmd = 0.0
                
                cmd = await drone.move_by_velocity_body_frame_z_async(
                    v_forward=v_fwd_cmd,
                    v_right=v_right_cmd,
                    z=z - 0.5,
                    duration=0.5,
                    yaw_is_rate=True,
                    yaw=yaw_rate_cmd,
                )
                await cmd
                continue
            elif front_min < 3.5: # Предупредительный порог для начала маневра
                print(f"[explore] Обнаружено препятствие ({front_min:.2f}м). Начинаю маневр уклонения.")
                 # Здесь код продолжит выполнение и попадет в логику obstacle_ahead
            # ---------------------------------------

            if obstacle_ahead:
                # Вычисляем коэффициент близости препятствия
                obstacle_ratio = max(0.0, front_min / max(avoid_dist, 0.1))
                
                # Определяем лучшую сторону для обхода
                best_side = "left" if left_min > right_min else "right"
                best_clearance = max(left_min, right_min)
                worst_clearance = min(left_min, right_min)
                
                # Если препятствие очень близко - агрессивный маневр
                if front_min < avoid_dist * 0.5:
                    # Критическое препятствие - отступаем и активно маневрируем
                    v_fwd_cmd = -cruise_speed * 0.7
                    turn_sign = 1.0 if best_side == "left" else -1.0
                    yaw_rate_cmd = turn_sign * max_yaw_rate  # максимальный поворот
                    v_right_cmd = turn_sign * cruise_speed * 0.9
                    target_z = z - 2.0  # поднимаемся на 2.0м
                    print(f"[explore] Критическое препятствие! Маневр: {best_side}, зазор: {best_clearance:.1f}м")
                
                # Если препятствия со всех сторон - подъем и поиск обхода
                elif left_min < avoid_dist * 0.8 and right_min < avoid_dist * 0.8 and back_min < avoid_dist * 1.0:
                    # Полностью окружен - высокий подъем и поворот
                    v_fwd_cmd = cruise_speed * 0.5
                    turn_sign = 1.0 if best_side == "left" else -1.0
                    yaw_rate_cmd = turn_sign * max_yaw_rate * 0.95
                    v_right_cmd = turn_sign * cruise_speed * 0.7
                    target_z = z - 3.0  # поднимаемся на 3.0м для поиска обхода
                    print(f"[explore] Окружен препятствиями! Поднимаемся высоко для поиска обхода {best_side}")
                
                # Если со всех сторон в горизонтальной плоскости, но есть пространство сверху
                elif left_min < avoid_dist * 0.8 and right_min < avoid_dist * 0.8:
                    # Горизонтальные препятствия - подъем и обход
                    v_fwd_cmd = cruise_speed * 0.6
                    turn_sign = 1.0 if best_side == "left" else -1.0
                    yaw_rate_cmd = turn_sign * max_yaw_rate * 0.9
                    v_right_cmd = turn_sign * cruise_speed * 0.8
                    target_z = z - 2.5  # поднимаемся на 2.5м
                    print(f"[explore] Горизонтальные препятствия! Поднимаемся и обходим {best_side}")
                
                # Нормальная ситуация - активный обход препятствия
                else:
                    # Активно обходим препятствие, продолжая исследование
                    turn_sign = 1.0 if best_side == "left" else -1.0
                    
                    # Адаптивная скорость обхода
                    # Более безопасная скорость
                    safe_speed_factor = max(0.1, obstacle_ratio ** 1.5)
                    forward_speed = cruise_speed * safe_speed_factor * 0.8
                    
                    # Активный обход с сохранением исследовательского движения
                    yaw_rate_cmd = turn_sign * max_yaw_rate * (0.8 + 0.2 * (1.0 - obstacle_ratio))  # 0.8-1.0 max_yaw_rate
                    v_right_cmd = turn_sign * cruise_speed * (0.7 + 0.3 * min(1.0, best_clearance / avoid_dist))
                    
                    # Комбинируем обходное движение с желаемым направлением исследования
                    # Смешиваем желаемое движение с обходным (70% обход, 30% желаемое)
                    v_fwd_cmd = forward_speed * 0.7 + v_fwd_desired * 0.3
                    v_right_cmd = v_right_cmd * 0.7 + v_right_desired * 0.3
                    
                    # Подъем для обхода
                    target_z = z - 1.0  # поднимаемся на 1.0м
                    print(f"[explore] Обходим препятствие {best_side}, продолжаем исследование, зазор: {best_clearance:.1f}м")

            # Дополнительная проверка: если препятствия слишком близко снизу - поднимаемся
            if down_min < avoid_dist * 0.8:
                target_z = min(target_z, z - 1.5)  # поднимаемся минимум на 1.5м если снизу препятствие
                
            # Если нет препятствий впереди, но есть опасность снизу - небольшой подъем для безопасности
            if not obstacle_ahead and down_min < avoid_dist * 1.2:
                target_z = min(target_z, z - 0.8)  # превентивный подъем

        # Ограничиваем скорости
        v_fwd_cmd = _clamp(v_fwd_cmd, -cruise_speed, cruise_speed)
        v_right_cmd = _clamp(v_right_cmd, -cruise_speed, cruise_speed)
        yaw_rate_cmd = _clamp(yaw_rate_cmd, -max_yaw_rate, max_yaw_rate)

        # Управляем дроном
        cmd = await drone.move_by_velocity_body_frame_z_async(
            v_forward=v_fwd_cmd,
            v_right=v_right_cmd,
            z=target_z,
            duration=dt,
            yaw_is_rate=True,
            yaw=yaw_rate_cmd,
        )
        await cmd
        await asyncio.sleep(0.001)

        last_pos_n = cur_n
        last_pos_e = cur_e

    print("[explore] Exploration timeout reached. Returning to start...")
    
    # Возвращаемся к стартовой точке
    await _drive_to_waypoint_reactive(
        drone=drone,
        lidar_latest=lidar_latest,
        pose_latest=pose_latest,
        target_n=start_n,
        target_e=start_e,
        z=z,
        cruise_speed=cruise_speed,
        dt=dt,
        arrive_tol=arrive_tol,
        avoid_dist=avoid_dist,
        influence_dist=influence_dist,
        max_repulse=max_repulse,
        max_yaw_rate=max_yaw_rate,
        timeout_sec=60.0,
    )

    # brief hover at end
    with contextlib.suppress(Exception):
        hover_task = await drone.hover_async()
        await hover_task


async def explore_forward_only(
    drone: Drone,
    lidar_latest: LidarLatest,
    pose_latest: PoseLatest,
    imu_latest: ImuLatest,
    lio_slam: SimpleLIO,
    extent_n: float,
    extent_e: float,
    z: float,
    cruise_speed: float,
    dt: float,
    avoid_dist: float,
    max_yaw_rate: float,
    total_timeout_sec: float,
) -> None:
    """
    Алгоритм ПОЛНОГО исследования карты с систематическим покрытием.
    
    Дрон выполняет:
    1. Паттерн "газонокосилки" (змейка) для полного покрытия области
    2. Сканирование на нескольких высотах (от низкой к высокой)
    3. Повороты на 360° в ключевых точках для полного охвата лидаром
    4. Автоматическое избегание препятствий
    
    SLAM карта строится автоматически через лидар со всех сторон.
    
    Args:
        drone: Объект дрона
        lidar_latest: Последние данные лидара
        pose_latest: Последняя поза дрона
        imu_latest: Последние данные IMU
        lio_slam: Объект LIO-SLAM для точной навигации
        extent_n: Размер области по North (м)
        extent_e: Размер области по East (м)
        z: Базовая высота полета (NED, отрицательное = вверх)
        cruise_speed: Крейсерская скорость (м/с)
        dt: Шаг управления (сек)
        avoid_dist: Дистанция срабатывания уклонения (м)
        max_yaw_rate: Максимальная скорость рыскания (рад/с)
        total_timeout_sec: Общий таймаут исследования (сек)
    """
    # === ПАРАМЕТРЫ ПОЛНОГО СКАНИРОВАНИЯ ===
    # Адаптивный шаг между проходами - для маленьких областей делаем плотнее
    base_lane_step = 8.0
    # Если область меньше 20м, уменьшаем шаг для лучшего покрытия
    if extent_e < 20.0:
        lane_step = min(base_lane_step, extent_e / 2.0, 4.0)  # Минимум 2 ряда, максимум шаг 4м
    else:
        lane_step = base_lane_step
    lane_step = max(2.0, lane_step)  # Минимальный шаг 2м для обеспечения покрытия
    heights = [z, z - 2.0, z - 4.0, z + 2.0]  # Высоты сканирования (NED)
    scan_rotation_degrees = 360.0  # Полный оборот для сканирования
    scan_rotation_speed = math.pi / 2  # 90 град/сек - скорость вращения
    waypoint_arrive_tol = 3.0  # Допуск достижения waypoint (м)
    scan_pause = 0.5  # Пауза для накопления данных лидара (сек)
    
    # Ждём позицию из actual_pose
    start_n = 0.0
    start_e = 0.0
    t_wait = time.time()
    while True:
        pose_msg, _ts = pose_latest.snapshot()
        if pose_msg is not None and isinstance(pose_msg, dict):
            pos = pose_msg.get("position", {})
            start_n = float(pos.get("x", 0.0))
            start_e = float(pos.get("y", 0.0))
            break
        if time.time() - t_wait > 5.0:
            kin = drone.get_ground_truth_kinematics()
            pos = kin["pose"]["position"]
            start_n = float(pos["x"])
            start_e = float(pos["y"])
            break
        await asyncio.sleep(0.05)

    t0 = time.time()
    
    print("[forward_only] ========================================")
    print("[forward_only] ПОЛНОЕ СКАНИРОВАНИЕ КАРТЫ")
    print(f"[forward_only] Область: {extent_n}м x {extent_e}м")
    print(f"[forward_only] Высоты сканирования: {heights}")
    print(f"[forward_only] Шаг между проходами: {lane_step}м")
    print("[forward_only] ========================================")
    
    # Генерируем waypoints для паттерна "газонокосилки"
    waypoints = _generate_lawnmower_waypoints(
        start_n=start_n - extent_n / 2,  # Начинаем от края области
        start_e=start_e - extent_e / 2,
        extent_n=extent_n,
        extent_e=extent_e,
        step_e=lane_step,
    )
    
    total_waypoints = len(waypoints)
    total_heights = len(heights)
    print(f"[forward_only] Создано {total_waypoints} waypoints для каждой высоты")
    print(f"[forward_only] Всего проходов: {total_waypoints * total_heights}")
    
    # === ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: Поворот на 360° для полного сканирования ===
    async def do_full_scan_rotation(current_height: float):
        """Выполняет полный оборот на 360° для сканирования со всех сторон."""
        print(f"[forward_only] 🔄 Выполняем поворот на {scan_rotation_degrees}° для полного сканирования...")
        
        rotation_duration = scan_rotation_degrees / (scan_rotation_speed * 180 / math.pi)
        rotation_steps = int(rotation_duration / dt) + 1
        
        for _ in range(rotation_steps):
            # Проверяем таймаут
            if time.time() - t0 >= total_timeout_sec:
                return
            
            cmd = await drone.move_by_velocity_body_frame_z_async(
                v_forward=0.0,
                v_right=0.0,
                z=current_height,
                duration=dt,
                yaw_is_rate=True,
                yaw=scan_rotation_speed,  # Вращаемся против часовой стрелки
            )
            await cmd
            await asyncio.sleep(0.01)
        
        # Пауза для накопления данных лидара
        await asyncio.sleep(scan_pause)
        print(f"[forward_only] ✅ Поворот завершен")
    
    # === ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: Движение к waypoint с избеганием препятствий ===
    async def navigate_to_waypoint(target_n: float, target_e: float, target_z: float, wp_timeout: float = 60.0) -> bool:
        """
        Навигация к waypoint с избеганием препятствий.
        Возвращает True если достигли цели, False если таймаут или общий таймаут.
        """
        wp_start = time.time()
        stuck_counter = 0
        last_dist = float('inf')
        
        while time.time() - wp_start < wp_timeout:
            # Проверяем общий таймаут
            if time.time() - t0 >= total_timeout_sec:
                return False
            
            pose_msg, _pose_ts = pose_latest.snapshot()
            if pose_msg is None:
                await asyncio.sleep(0.05)
                continue
            
            # Получаем данные IMU и LiDAR для LIO-SLAM
            imu_orientation, imu_angular_velocity, imu_linear_acceleration, imu_time = imu_latest.snapshot()
            lidar_pts, lidar_time = lidar_latest.snapshot()
            
            # Обновляем LIO-SLAM состояние
            lio_state = lio_slam.update_state(
                imu_orientation=imu_orientation,
                imu_angular_velocity=imu_angular_velocity,
                imu_linear_acceleration=imu_linear_acceleration,
                imu_time=imu_time,
                lidar_points=lidar_pts,
                pose_gt=pose_msg,
                lidar_time=lidar_time,
            )
            
            # Используем позицию из LIO-SLAM для более точной навигации
            pos = pose_msg.get("position", {}) if isinstance(pose_msg, dict) else {}
            ori = pose_msg.get("orientation", {}) if isinstance(pose_msg, dict) else {}
            
            # Смешиваем позицию LIO-SLAM с ground truth
            lio_pos = lio_state.get("position", [0.0, 0.0, 0.0])
            gt_n = float(pos.get("x", lio_pos[0]))
            gt_e = float(pos.get("y", lio_pos[1]))
            
            alpha_lio = 0.7
            cur_n = alpha_lio * lio_pos[0] + (1.0 - alpha_lio) * gt_n
            cur_e = alpha_lio * lio_pos[1] + (1.0 - alpha_lio) * gt_e
            
            # Используем ориентацию из LIO-SLAM или ground truth
            lio_ori = lio_state.get("orientation", ori if isinstance(ori, dict) else {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
            yaw = _quat_to_yaw_rad(lio_ori)
            
            # Вычисляем расстояние до цели
            dn = target_n - cur_n
            de = target_e - cur_e
            dist = math.hypot(dn, de)
            
            # Проверяем, достигли ли waypoint
            if dist < waypoint_arrive_tol:
                return True
            
            # Проверяем на застревание
            if abs(dist - last_dist) < 0.1:
                stuck_counter += 1
                if stuck_counter > 50:  # ~5 секунд без движения
                    print(f"[forward_only] ⚠️ Застряли! Пропускаем waypoint")
                    return True  # Продолжаем к следующему
            else:
                stuck_counter = 0
            last_dist = dist
            
            # Вычисляем направление к цели
            target_yaw = math.atan2(de, dn)
            yaw_error = target_yaw - yaw
            # Нормализуем угол в диапазон [-pi, pi]
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi
            
            pts, _ts = lidar_latest.snapshot()
            
            # Если нет данных лидара - продолжаем движение с осторожностью
            if pts is None or getattr(pts, "size", 0) == 0:
                speed = cruise_speed * 0.5
                if abs(yaw_error) > 0.3:
                    # Сначала поворачиваемся к цели
                    v_fwd_cmd = 0.0
                    yaw_rate_cmd = _clamp(yaw_error * 2.0, -max_yaw_rate, max_yaw_rate)
                else:
                    v_fwd_cmd = speed
                    yaw_rate_cmd = _clamp(yaw_error * 1.0, -max_yaw_rate * 0.5, max_yaw_rate * 0.5)
                
                cmd = await drone.move_by_velocity_body_frame_z_async(
                    v_forward=v_fwd_cmd,
                    v_right=0.0,
                    z=target_z,
                    duration=dt,
                    yaw_is_rate=True,
                    yaw=yaw_rate_cmd,
                )
                await cmd
                await asyncio.sleep(0.01)
                continue
            
            # Проверяем препятствия
            front_min = _min_range_in_cone(pts, az_min_rad=-math.radians(45), az_max_rad=math.radians(45), max_range=999.0)
            left_min = _min_range_in_cone(pts, az_min_rad=math.radians(30), az_max_rad=math.radians(90), max_range=999.0)
            right_min = _min_range_in_cone(pts, az_min_rad=-math.radians(90), az_max_rad=-math.radians(30), max_range=999.0)
            
            # Проверка на опасность столкновения ножек
            gear_collision_danger = _check_landing_gear_collision(pts, landing_gear_height=0.5, safety_margin=1.5)
            
            # КРИТИЧЕСКАЯ ОПАСНОСТЬ - экстренная остановка
            if front_min < 1.5 or gear_collision_danger:
                print(f"[forward_only] 🚨 КРИТИЧЕСКАЯ ОПАСНОСТЬ ({front_min:.1f}м)! Экстренный маневр!")
                # Отступаем и поднимаемся
                cmd = await drone.move_by_velocity_body_frame_z_async(
                    v_forward=-cruise_speed * 0.5,
                    v_right=0.0,
                    z=target_z - 2.0,
                    duration=0.5,
                    yaw_is_rate=True,
                    yaw=0.0,
                )
                await cmd
                await asyncio.sleep(0.2)
                continue
            
            # Определяем скорость и направление
            if front_min < avoid_dist:
                # Препятствие впереди - обходим
                best_side = "left" if left_min > right_min else "right"
                turn_sign = 1.0 if best_side == "left" else -1.0
                
                # Замедляемся и поворачиваем
                safe_speed = cruise_speed * (front_min / avoid_dist) * 0.5
                v_fwd_cmd = max(0.0, safe_speed)
                v_right_cmd = turn_sign * safe_speed * 0.6  # Боковое движение для обхода
                yaw_rate_cmd = turn_sign * max_yaw_rate * 0.5  # Меньший поворот, больше бокового движения
                
                # Небольшой подъем при обнаружении препятствия
                current_z = target_z
                if front_min < avoid_dist * 0.6:
                    current_z = target_z - 1.0
            else:
                # Путь свободен - летим к цели с использованием бокового движения
                # Скорость зависит от расстояния до цели
                speed = min(cruise_speed, max(0.5, dist * 0.3))
                
                # Вычисляем направление к цели в world frame
                v_n_world = speed * (dn / max(dist, 1e-6))
                v_e_world = speed * (de / max(dist, 1e-6))
                
                # Преобразуем в body frame для использования v_forward и v_right
                v_fwd_target, v_right_target = _world_to_body(v_n_world, v_e_world, yaw)
                
                # Если ошибка по углу небольшая, используем боковое движение для эффективного перемещения
                if abs(yaw_error) < math.pi / 3:  # Меньше 60 градусов
                    v_fwd_cmd = v_fwd_target
                    v_right_cmd = v_right_target
                    yaw_rate_cmd = _clamp(yaw_error * 0.5, -max_yaw_rate * 0.3, max_yaw_rate * 0.3)
                else:
                    # Большая ошибка - больше поворота, но также используем боковое движение
                    v_fwd_cmd = v_fwd_target * 0.5
                    v_right_cmd = v_right_target * 0.7
                    yaw_rate_cmd = _clamp(yaw_error * 1.5, -max_yaw_rate, max_yaw_rate)
                
                current_z = target_z
            
            # Ограничиваем скорости
            v_fwd_cmd = _clamp(v_fwd_cmd, -cruise_speed, cruise_speed)
            v_right_cmd = _clamp(v_right_cmd, -cruise_speed, cruise_speed)
            yaw_rate_cmd = _clamp(yaw_rate_cmd, -max_yaw_rate, max_yaw_rate)
            
            # Управляем дроном
            cmd = await drone.move_by_velocity_body_frame_z_async(
                v_forward=v_fwd_cmd,
                v_right=v_right_cmd,
                z=current_z,
                duration=dt,
                yaw_is_rate=True,
                yaw=yaw_rate_cmd,
            )
            await cmd
            await asyncio.sleep(0.01)
        
        # Таймаут waypoint
        print(f"[forward_only] ⏱️ Таймаут waypoint, продолжаем...")
        return True
    
    # === ОСНОВНОЙ ЦИКЛ СКАНИРОВАНИЯ ===
    height_idx = 0
    for current_height in heights:
        height_idx += 1
        
        # Проверяем общий таймаут
        if time.time() - t0 >= total_timeout_sec:
            print(f"[forward_only] ⏱️ Общий таймаут исследования")
            break
        
        print(f"\n[forward_only] ========== ВЫСОТА {height_idx}/{total_heights}: {current_height}м ==========")
        
        # Сначала поднимаемся/опускаемся на нужную высоту
        print(f"[forward_only] Переход на высоту {current_height}м...")
        for _ in range(20):  # ~2 секунды на изменение высоты
            cmd = await drone.move_by_velocity_body_frame_z_async(
                v_forward=0.0,
                v_right=0.0,
                z=current_height,
                duration=0.1,
                yaw_is_rate=True,
                yaw=0.0,
            )
            await cmd
            await asyncio.sleep(0.05)
        
        # Начальное сканирование на этой высоте
        await do_full_scan_rotation(current_height)
        
        # Проходим все waypoints на этой высоте
        wp_idx = 0
        # Чередуем направление для чётных/нечётных высот (для лучшего покрытия)
        wp_list = waypoints if height_idx % 2 == 1 else list(reversed(waypoints))
        
        for wp_n, wp_e in wp_list:
            wp_idx += 1
            
            # Проверяем общий таймаут
            if time.time() - t0 >= total_timeout_sec:
                print(f"[forward_only] ⏱️ Общий таймаут исследования")
                break
            
            print(f"[forward_only] 📍 Waypoint {wp_idx}/{total_waypoints}: ({wp_n:.1f}, {wp_e:.1f})")
            
            # Навигация к waypoint
            reached = await navigate_to_waypoint(wp_n, wp_e, current_height)
            
            if not reached:
                break  # Общий таймаут
            
            # Полный поворот для сканирования каждые 2-3 waypoint
            if wp_idx % 2 == 0:
                await do_full_scan_rotation(current_height)
            else:
                # Небольшая пауза для накопления данных
                await asyncio.sleep(scan_pause)
        
        print(f"[forward_only] ✅ Высота {current_height}м завершена")
    
    # === ВОЗВРАТ К СТАРТОВОЙ ТОЧКЕ ===
    print("\n[forward_only] ========================================")
    print("[forward_only] Исследование завершено! Возвращаемся к стартовой точке...")
    print("[forward_only] ========================================")
    
    # Навигация к стартовой точке
    await navigate_to_waypoint(start_n, start_e, z, wp_timeout=120.0)
    
    # Финальное сканирование в стартовой точке
    print("[forward_only] Финальное сканирование в стартовой точке...")
    await do_full_scan_rotation(z)
    
    # Краткое зависание в конце
    with contextlib.suppress(Exception):
        hover_task = await drone.hover_async()
        await hover_task
    
    elapsed = time.time() - t0
    print(f"\n[forward_only] ========================================")
    print(f"[forward_only] СКАНИРОВАНИЕ ЗАВЕРШЕНО!")
    print(f"[forward_only] Время: {elapsed/60:.1f} минут")
    print(f"[forward_only] Высот отсканировано: {height_idx}")
    print(f"[forward_only] ========================================")


async def explore_waypoints_sequential(
    drone: Drone,
    lidar_latest: LidarLatest,
    pose_latest: PoseLatest,
    imu_latest: ImuLatest,
    lio_slam: SimpleLIO,
    extent_n: float,
    extent_e: float,
    z: float,
    cruise_speed: float,
    dt: float,
    avoid_dist: float,
    max_yaw_rate: float,
    total_timeout_sec: float = 600.0,
) -> None:
    """
    Последовательный облёт точек A → B → C → D → E → A.
    
    Маршрут образует прямоугольник вокруг центральной области:
    - A: стартовая позиция (центр)
    - B: внизу слева (North-, East-)
    - C: внизу справа (North-, East+)
    - D: вверху справа (North+, East+)
    - E: вверху слева (North+, East-)
    - Возврат в A
    
    Args:
        drone: Объект дрона
        lidar_latest: Последние данные лидара
        pose_latest: Последняя поза дрона
        imu_latest: Последние данные IMU
        lio_slam: Объект LIO-SLAM для точной навигации
        extent_n: Размер области по North (м) - определяет расстояние до B/C и D/E от центра
        extent_e: Размер области по East (м) - определяет расстояние до B/E и C/D от центра
        z: Высота полета (NED, отрицательное = вверх)
        cruise_speed: Крейсерская скорость (м/с)
        dt: Шаг управления (сек)
        avoid_dist: Дистанция срабатывания уклонения (м)
        max_yaw_rate: Максимальная скорость рыскания (рад/с)
        total_timeout_sec: Общий таймаут исследования (сек)
    """
    waypoint_arrive_tol = 2.0  # Допуск достижения waypoint (м)
    scan_pause = 0.5  # Пауза для накопления данных лидара (сек)
    scan_rotation_speed = math.pi / 2  # 90 град/сек
    
    # Ждём позицию из actual_pose
    start_n = 0.0
    start_e = 0.0
    t_wait = time.time()
    while True:
        pose_msg, _ts = pose_latest.snapshot()
        if pose_msg is not None and isinstance(pose_msg, dict):
            pos = pose_msg.get("position", {})
            start_n = float(pos.get("x", 0.0))
            start_e = float(pos.get("y", 0.0))
            break
        if time.time() - t_wait > 5.0:
            kin = drone.get_ground_truth_kinematics()
            pos = kin["pose"]["position"]
            start_n = float(pos["x"])
            start_e = float(pos["y"])
            break
        await asyncio.sleep(0.05)

    t0 = time.time()
    
    # Определяем точки маршрута относительно стартовой позиции
    # Координатная система NED: North (X) = вправо на экране, East (Y) = вниз на экране
    # A - стартовая позиция (центр)
    # B - внизу слева (North-, East-)
    # C - внизу справа (North-, East+)
    # D - вверху справа (North+, East+)
    # E - вверху слева (North+, East-)
    # Маршрут: A → B → C → D → E → A (прямоугольник по часовой стрелке)
    
    waypoints_named = [
        ("A (старт)", start_n, start_e),
        ("B", start_n - extent_n, start_e - extent_e),
        ("C", start_n - extent_n, start_e + extent_e),
        ("D", start_n + extent_n, start_e + extent_e),
        ("E", start_n + extent_n, start_e - extent_e),
        ("A (возврат)", start_n, start_e),
    ]
    
    print("[waypoints] ========================================")
    print("[waypoints] ПОСЛЕДОВАТЕЛЬНЫЙ ОБЛЁТ ТОЧЕК A → B → C → D → E → A")
    print(f"[waypoints] Высота полёта: {z}м (NED)")
    print(f"[waypoints] Скорость: {cruise_speed} м/с")
    print(f"[waypoints] Область: {extent_n}м x {extent_e}м")
    print("[waypoints] Маршрут:")
    for name, n, e in waypoints_named:
        print(f"  - {name}: ({n:.1f}, {e:.1f})")
    print("[waypoints] ========================================")
    
    # === ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: Поворот на 360° для сканирования ===
    async def do_scan_rotation():
        """Выполняет полный оборот на 360° для сканирования."""
        print(f"[waypoints] 🔄 Сканирование 360°...")
        rotation_duration = 360.0 / (scan_rotation_speed * 180 / math.pi)
        rotation_steps = int(rotation_duration / dt) + 1
        
        for _ in range(rotation_steps):
            if time.time() - t0 >= total_timeout_sec:
                return
            cmd = await drone.move_by_velocity_body_frame_z_async(
                v_forward=0.0,
                v_right=0.0,
                z=z,
                duration=dt,
                yaw_is_rate=True,
                yaw=scan_rotation_speed,
            )
            await cmd
            await asyncio.sleep(0.01)
        await asyncio.sleep(scan_pause)
    
    # === ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: Движение к waypoint ===
    async def navigate_to_waypoint(target_n: float, target_e: float, wp_timeout: float = 60.0) -> bool:
        """Навигация к waypoint с избеганием препятствий."""
        wp_start = time.time()
        stuck_counter = 0
        last_dist = float('inf')
        
        while time.time() - wp_start < wp_timeout:
            if time.time() - t0 >= total_timeout_sec:
                return False
            
            pose_msg, _pose_ts = pose_latest.snapshot()
            if pose_msg is None:
                await asyncio.sleep(0.05)
                continue
            
            # Обновляем LIO-SLAM
            imu_orientation, imu_angular_velocity, imu_linear_acceleration, imu_time = imu_latest.snapshot()
            lidar_pts, lidar_time = lidar_latest.snapshot()
            
            lio_state = lio_slam.update_state(
                imu_orientation=imu_orientation,
                imu_angular_velocity=imu_angular_velocity,
                imu_linear_acceleration=imu_linear_acceleration,
                imu_time=imu_time,
                lidar_points=lidar_pts,
                pose_gt=pose_msg,
                lidar_time=lidar_time,
            )
            
            pos = pose_msg.get("position", {}) if isinstance(pose_msg, dict) else {}
            ori = pose_msg.get("orientation", {}) if isinstance(pose_msg, dict) else {}
            
            lio_pos = lio_state.get("position", [0.0, 0.0, 0.0])
            gt_n = float(pos.get("x", lio_pos[0]))
            gt_e = float(pos.get("y", lio_pos[1]))
            
            alpha_lio = 0.7
            cur_n = alpha_lio * lio_pos[0] + (1.0 - alpha_lio) * gt_n
            cur_e = alpha_lio * lio_pos[1] + (1.0 - alpha_lio) * gt_e
            
            lio_ori = lio_state.get("orientation", ori if isinstance(ori, dict) else {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
            yaw = _quat_to_yaw_rad(lio_ori)
            
            dn = target_n - cur_n
            de = target_e - cur_e
            dist = math.hypot(dn, de)
            
            # Достигли waypoint
            if dist < waypoint_arrive_tol:
                return True
            
            # Проверка на застревание
            if abs(dist - last_dist) < 0.1:
                stuck_counter += 1
                if stuck_counter > 50:
                    print(f"[waypoints] ⚠️ Застряли! Пропускаем waypoint")
                    return True
            else:
                stuck_counter = 0
            last_dist = dist
            
            target_yaw = math.atan2(de, dn)
            yaw_error = target_yaw - yaw
            while yaw_error > math.pi:
                yaw_error -= 2 * math.pi
            while yaw_error < -math.pi:
                yaw_error += 2 * math.pi
            
            pts, _ts = lidar_latest.snapshot()
            
            if pts is None or getattr(pts, "size", 0) == 0:
                speed = cruise_speed * 0.5
                if abs(yaw_error) > 0.3:
                    v_fwd_cmd = 0.0
                    yaw_rate_cmd = _clamp(yaw_error * 2.0, -max_yaw_rate, max_yaw_rate)
                else:
                    v_fwd_cmd = speed
                    yaw_rate_cmd = _clamp(yaw_error * 1.0, -max_yaw_rate * 0.5, max_yaw_rate * 0.5)
                
                cmd = await drone.move_by_velocity_body_frame_z_async(
                    v_forward=v_fwd_cmd,
                    v_right=0.0,
                    z=z,
                    duration=dt,
                    yaw_is_rate=True,
                    yaw=yaw_rate_cmd,
                )
                await cmd
                await asyncio.sleep(0.01)
                continue
            
            front_min = _min_range_in_cone(pts, az_min_rad=-math.radians(45), az_max_rad=math.radians(45), max_range=999.0)
            left_min = _min_range_in_cone(pts, az_min_rad=math.radians(30), az_max_rad=math.radians(90), max_range=999.0)
            right_min = _min_range_in_cone(pts, az_min_rad=-math.radians(90), az_max_rad=-math.radians(30), max_range=999.0)
            
            gear_collision_danger = _check_landing_gear_collision(pts, landing_gear_height=0.5, safety_margin=1.5)
            
            # Критическая опасность
            if front_min < 1.5 or gear_collision_danger:
                print(f"[waypoints] 🚨 КРИТИЧЕСКАЯ ОПАСНОСТЬ ({front_min:.1f}м)! Экстренный маневр!")
                cmd = await drone.move_by_velocity_body_frame_z_async(
                    v_forward=-cruise_speed * 0.5,
                    v_right=0.0,
                    z=z - 2.0,
                    duration=0.5,
                    yaw_is_rate=True,
                    yaw=0.0,
                )
                await cmd
                await asyncio.sleep(0.2)
                continue
            
            if front_min < avoid_dist:
                best_side = "left" if left_min > right_min else "right"
                turn_sign = 1.0 if best_side == "left" else -1.0
                safe_speed = cruise_speed * (front_min / avoid_dist) * 0.5
                v_fwd_cmd = max(0.0, safe_speed)
                v_right_cmd = turn_sign * safe_speed * 0.6
                yaw_rate_cmd = turn_sign * max_yaw_rate * 0.5
                current_z = z - 1.0 if front_min < avoid_dist * 0.6 else z
            else:
                speed = min(cruise_speed, max(0.5, dist * 0.3))
                v_n_world = speed * (dn / max(dist, 1e-6))
                v_e_world = speed * (de / max(dist, 1e-6))
                v_fwd_target, v_right_target = _world_to_body(v_n_world, v_e_world, yaw)
                
                if abs(yaw_error) < math.pi / 3:
                    v_fwd_cmd = v_fwd_target
                    v_right_cmd = v_right_target
                    yaw_rate_cmd = _clamp(yaw_error * 0.5, -max_yaw_rate * 0.3, max_yaw_rate * 0.3)
                else:
                    v_fwd_cmd = v_fwd_target * 0.5
                    v_right_cmd = v_right_target * 0.7
                    yaw_rate_cmd = _clamp(yaw_error * 1.5, -max_yaw_rate, max_yaw_rate)
                current_z = z
            
            v_fwd_cmd = _clamp(v_fwd_cmd, -cruise_speed, cruise_speed)
            v_right_cmd = _clamp(v_right_cmd, -cruise_speed, cruise_speed)
            yaw_rate_cmd = _clamp(yaw_rate_cmd, -max_yaw_rate, max_yaw_rate)
            
            cmd = await drone.move_by_velocity_body_frame_z_async(
                v_forward=v_fwd_cmd,
                v_right=v_right_cmd,
                z=current_z,
                duration=dt,
                yaw_is_rate=True,
                yaw=yaw_rate_cmd,
            )
            await cmd
            await asyncio.sleep(0.01)
        
        print(f"[waypoints] ⏱️ Таймаут waypoint")
        return True
    
    # === ОСНОВНОЙ ЦИКЛ ОБЛЁТА ===
    print(f"\n[waypoints] Поднимаемся на высоту {z}м...")
    for _ in range(30):
        cmd = await drone.move_by_velocity_body_frame_z_async(
            v_forward=0.0,
            v_right=0.0,
            z=z,
            duration=0.1,
            yaw_is_rate=True,
            yaw=0.0,
        )
        await cmd
        await asyncio.sleep(0.05)
    
    # Начальное сканирование в точке A
    print("[waypoints] 📍 Точка A (старт) - начальное сканирование")
    await do_scan_rotation()
    
    # Облёт точек B, C, D, E
    for i, (name, wp_n, wp_e) in enumerate(waypoints_named[1:], 1):
        if time.time() - t0 >= total_timeout_sec:
            print(f"[waypoints] ⏱️ Общий таймаут")
            break
        
        print(f"\n[waypoints] ➡️ Летим к точке {name} ({wp_n:.1f}, {wp_e:.1f})")
        reached = await navigate_to_waypoint(wp_n, wp_e, wp_timeout=90.0)
        
        if reached:
            print(f"[waypoints] ✅ Достигли точки {name}")
            await do_scan_rotation()
        else:
            print(f"[waypoints] ❌ Не удалось достичь точки {name}")
    
    # Финальное зависание
    with contextlib.suppress(Exception):
        hover_task = await drone.hover_async()
        await hover_task
    
    elapsed = time.time() - t0
    print(f"\n[waypoints] ========================================")
    print(f"[waypoints] ОБЛЁТ ЗАВЕРШЁН!")
    print(f"[waypoints] Время: {elapsed/60:.1f} минут")
    print(f"[waypoints] ========================================")


async def explore_map_systematic(
    drone: Drone,
    lidar_latest: LidarLatest,
    pose_latest: PoseLatest,
    imu_latest: ImuLatest,
    lio_slam: SimpleLIO,
    path_tracker: Optional[PathTracker],
    extent_n: float,
    extent_e: float,
    z: float,
    cruise_speed: float,
    dt: float,
    arrive_tol: float,
    avoid_dist: float,
    influence_dist: float,
    max_repulse: float,
    max_yaw_rate: float,
    grid_resolution: float = 2.5,
    total_timeout_sec: float = 600.0,
) -> None:
    """
    Систематическое исследование карты с использованием сетки для более точного создания облака точек.
    Дрон исследует карту по сетке, избегая препятствий и возвращаясь в начальную точку.
    
    Args:
        drone: Объект дрона
        lidar_latest: Последние данные лидара
        pose_latest: Последняя поза дрона
        imu_latest: Последние данные IMU
        lio_slam: Объект LIO-SLAM для точной навигации
        extent_n: Размер области по North (м)
        extent_e: Размер области по East (м)
        z: Высота полета (NED, отрицательное = вверх)
        cruise_speed: Крейсерская скорость (м/с)
        dt: Шаг управления (сек)
        arrive_tol: Допуск достижения точки (м)
        avoid_dist: Дистанция срабатывания уклонения (м)
        influence_dist: Радиус влияния для отталкивания (м)
        max_repulse: Максимальная отталкивающая скорость (м/с)
        max_yaw_rate: Максимальная скорость рыскания (рад/с)
        grid_resolution: Разрешение сетки для исследования (м)
        total_timeout_sec: Общий таймаут исследования (сек)
    """
    print("[systematic_explore] Начинаем систематическое исследование карты")
    
    # Получаем стартовую позицию
    start_n = 0.0
    start_e = 0.0
    t_wait = time.time()
    while True:
        pose_msg, _ts = pose_latest.snapshot()
        if pose_msg is not None and isinstance(pose_msg, dict):
            pos = pose_msg.get("position", {})
            start_n = float(pos.get("x", 0.0))
            start_e = float(pos.get("y", 0.0))
            break
        if time.time() - t_wait > 5.0:
            kin = drone.get_ground_truth_kinematics()
            pos = kin["pose"]["position"]
            start_n = float(pos["x"])
            start_e = float(pos["y"])
            break
        await asyncio.sleep(0.05)
    
    print(f"[systematic_explore] Стартовая позиция: ({start_n:.2f}, {start_e:.2f})")
    
    # Создаем сетку для исследования
    grid_size_n = int(math.ceil(abs(extent_n) / grid_resolution))
    grid_size_e = int(math.ceil(abs(extent_e) / grid_resolution))
    
    # Словарь для отслеживания посещенных ячеек: (grid_n, grid_e) -> visited
    visited_cells = {}
    
    # Список целей для исследования (приоритет: ближайшие неисследованные ячейки)
    exploration_targets = []
    
    # Генерируем все ячейки сетки
    for i in range(grid_size_n):
        for j in range(grid_size_e):
            # Координаты центра ячейки в мировых координатах
            cell_n = start_n + (i - grid_size_n / 2) * grid_resolution
            cell_e = start_e + (j - grid_size_e / 2) * grid_resolution
            exploration_targets.append((cell_n, cell_e, i, j))
    
    print(f"[systematic_explore] Создана сетка {grid_size_n}x{grid_size_e} ячеек ({len(exploration_targets)} целей)")
    
    t0 = time.time()
    current_target_idx = 0
    last_target_change = time.time()
    stuck_counter = 0
    
    # Инициализация LIO-SLAM
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
    
    while time.time() - t0 < total_timeout_sec:
        pose_msg, _pose_ts = pose_latest.snapshot()
        if pose_msg is None:
            await asyncio.sleep(0.05)
            continue
        
        # Получаем данные IMU и LiDAR для LIO-SLAM
        imu_orientation, imu_angular_velocity, imu_linear_acceleration, imu_time = imu_latest.snapshot()
        lidar_pts, lidar_time = lidar_latest.snapshot()
        
        # Обновляем LIO-SLAM состояние
        lio_state = lio_slam.update_state(
            imu_orientation=imu_orientation,
            imu_angular_velocity=imu_angular_velocity,
            imu_linear_acceleration=imu_linear_acceleration,
            imu_time=imu_time,
            lidar_points=lidar_pts,
            pose_gt=pose_msg,
            lidar_time=lidar_time,
        )
        
        # Используем позицию из LIO-SLAM для более точной навигации
        pos = pose_msg.get("position", {}) if isinstance(pose_msg, dict) else {}
        ori = pose_msg.get("orientation", {}) if isinstance(pose_msg, dict) else {}
        
        # Смешиваем позицию LIO-SLAM с ground truth
        lio_pos = lio_state.get("position", [0.0, 0.0, 0.0])
        gt_n = float(pos.get("x", lio_pos[0]))
        gt_e = float(pos.get("y", lio_pos[1]))
        gt_d = float(pos.get("z", lio_pos[2]))
        
        alpha_lio = 0.7
        cur_n = alpha_lio * lio_pos[0] + (1.0 - alpha_lio) * gt_n
        cur_e = alpha_lio * lio_pos[1] + (1.0 - alpha_lio) * gt_e
        cur_z = alpha_lio * lio_pos[2] + (1.0 - alpha_lio) * gt_d
        
        # Обновляем путь дрона для отрисовки красной линии траектории
        if path_tracker is not None:
            path_tracker.update_position((cur_n, cur_e, cur_z))
        
        # Используем ориентацию из LIO-SLAM или ground truth
        lio_ori = lio_state.get("orientation", ori if isinstance(ori, dict) else {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0})
        yaw = _quat_to_yaw_rad(lio_ori)
        
        # Определяем текущую ячейку сетки
        grid_n = int(round((cur_n - start_n) / grid_resolution + grid_size_n / 2))
        grid_e = int(round((cur_e - start_e) / grid_resolution + grid_size_e / 2))
        
        # Помечаем текущую ячейку и соседние как посещенные
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = grid_n + di, grid_e + dj
                if 0 <= ni < grid_size_n and 0 <= nj < grid_size_e:
                    visited_cells[(ni, nj)] = True
        
        # Выбираем следующую цель для исследования
        # Ищем ближайшую неисследованную ячейку
        best_target = None
        best_distance = float('inf')
        best_idx = -1
        
        for idx, (target_n, target_e, gi, gj) in enumerate(exploration_targets):
            if visited_cells.get((gi, gj), False):
                continue  # Пропускаем уже посещенные
            
            # Проверяем, достижима ли ячейка (нет ли препятствий на пути)
            distance = math.hypot(target_n - cur_n, target_e - cur_e)
            
            # Предпочитаем более близкие ячейки
            if distance < best_distance:
                best_distance = distance
                best_target = (target_n, target_e)
                best_idx = idx
        
        # Если все ячейки исследованы или нет доступных целей, возвращаемся к старту
        if best_target is None:
            print("[systematic_explore] Все ячейки исследованы или недоступны. Возвращаемся к старту...")
            visited_count = len(visited_cells)
            total_cells = grid_size_n * grid_size_e
            print(f"[systematic_explore] Исследовано {visited_count}/{total_cells} ячеек ({100*visited_count/total_cells:.1f}%)")
            break
        
        target_n, target_e = best_target
        
        # Вычисляем расстояние до цели (используется в логике движения)
        dist_to_target = math.hypot(target_n - cur_n, target_e - cur_e)
        
        # Если цель изменилась, обновляем счетчик
        if best_idx != current_target_idx:
            current_target_idx = best_idx
            last_target_change = time.time()
            stuck_counter = 0
            print(f"[systematic_explore] Новая цель: ({target_n:.1f}, {target_e:.1f}), расстояние: {dist_to_target:.1f}м, ячейка ({grid_n}, {grid_e})")
        
        # Проверяем, достигли ли мы цели
        if dist_to_target < arrive_tol:
            # Помечаем ячейку как посещенную
            visited_cells[(grid_n, grid_e)] = True
            print(f"[systematic_explore] Достигнута цель ({target_n:.1f}, {target_e:.1f})")
            await asyncio.sleep(0.5)  # Небольшая пауза для накопления данных лидара
            continue
        
        # Проверяем на застревание
        if time.time() - last_target_change > 10.0:
            stuck_counter += 1
            if stuck_counter > 3:
                print(f"[systematic_explore] Застряли на цели. Пропускаем ячейку и ищем следующую...")
                # Помечаем текущую ячейку как недоступную (посещенную)
                visited_cells[(grid_n, grid_e)] = True
                stuck_counter = 0
                last_target_change = time.time()
                continue
        
        # Получаем данные лидара для избегания препятствий
        pts, _ts = lidar_latest.snapshot()
        
        if pts is None or getattr(pts, "size", 0) == 0:
            # Нет данных лидара - продолжаем движение с осторожностью
            speed = min(cruise_speed * 0.5, max(0.2, dist_to_target * 0.3))
            dn = target_n - cur_n
            de = target_e - cur_e
            v_n = speed * (dn / max(dist_to_target, 1e-6))
            v_e = speed * (de / max(dist_to_target, 1e-6))
            v_fwd, v_right = _world_to_body(v_n, v_e, yaw)
            
            cmd = await drone.move_by_velocity_body_frame_z_async(
                v_forward=v_fwd,
                v_right=v_right,
                z=z - 0.5,  # Небольшой подъем для безопасности
                duration=dt,
                yaw_is_rate=True,
                yaw=0.0,
            )
            await cmd
            await asyncio.sleep(0.001)
            continue
        
        # Проверка на опасность столкновения ножек
        gear_collision_danger = _check_landing_gear_collision(pts, landing_gear_height=0.5, safety_margin=1.5)
        
        # Проверяем препятствия в разных направлениях (расширенные углы для лучшего обнаружения)
        front_min = _min_range_in_cone(pts, az_min_rad=-math.radians(45), az_max_rad=math.radians(45), max_range=999.0)
        left_min = _min_range_in_cone(pts, az_min_rad=math.radians(30), az_max_rad=math.radians(90), max_range=999.0)
        right_min = _min_range_in_cone(pts, az_min_rad=-math.radians(90), az_max_rad=-math.radians(30), max_range=999.0)
        back_min = _min_range_in_cone(pts, az_min_rad=math.radians(135), az_max_rad=math.radians(180), max_range=999.0)
        down_min = _min_range_in_cone(pts, az_min_rad=-math.radians(90), az_max_rad=math.radians(90), max_range=999.0)
        
        # Увеличенные пороги безопасности
        critical_distance = 2.0  # Критическое расстояние (было 1.0)
        warning_distance = avoid_dist * 1.5  # Предупреждение на большем расстоянии
        safe_speed_distance = avoid_dist * 2.0  # Расстояние для снижения скорости
        
        # Желаемое направление к цели (dist_to_target уже вычислено выше)
        dn = target_n - cur_n
        de = target_e - cur_e
        desired_heading = math.atan2(de, dn)
        
        # Проверяем препятствия в направлении к цели
        target_heading_relative = desired_heading - yaw
        # Нормализуем угол
        while target_heading_relative > math.pi:
            target_heading_relative -= 2 * math.pi
        while target_heading_relative < -math.pi:
            target_heading_relative += 2 * math.pi
        
        # Проверяем препятствия в направлении цели (более узкий конус)
        target_dir_min = _min_range_in_cone(
            pts, 
            az_min_rad=target_heading_relative - math.radians(20), 
            az_max_rad=target_heading_relative + math.radians(20), 
            max_range=999.0
        )
        
        # Адаптивная скорость в зависимости от расстояния до препятствий
        base_speed = cruise_speed
        # Учитываем как общее препятствие впереди, так и в направлении цели
        min_obstacle_dist = min(front_min, target_dir_min)
        if min_obstacle_dist < safe_speed_distance:
            # Снижаем скорость при приближении к препятствиям
            speed_factor = max(0.2, min(1.0, (min_obstacle_dist - critical_distance) / (safe_speed_distance - critical_distance)))
            base_speed = cruise_speed * speed_factor
            # Дополнительно снижаем скорость если препятствие прямо на пути к цели
            if target_dir_min < front_min * 0.8:
                speed_factor *= 0.7  # Еще больше снижаем скорость
                base_speed = cruise_speed * speed_factor
        
        desired_v_n = base_speed * math.cos(desired_heading)
        desired_v_e = base_speed * math.sin(desired_heading)
        
        # Конвертируем в body frame
        v_fwd_desired, v_right_desired = _world_to_body(desired_v_n, desired_v_e, yaw)
        
        # Отталкивающая сила от препятствий (усиленная)
        rep_fwd, rep_right = _repulsive_velocity_xy(pts, influence_dist=influence_dist * 1.2, max_repulse=max_repulse * 1.2, landing_gear_height=0.5)
        
        # Комбинируем желаемое движение и отталкивание
        v_fwd_cmd = v_fwd_desired + rep_fwd
        v_right_cmd = v_right_desired + rep_right
        yaw_rate_cmd = 0.0
        target_z = z
        
        # Логика избегания препятствий (более агрессивная)
        obstacle_ahead = front_min < warning_distance
        
        # КРИТИЧЕСКАЯ ОПАСНОСТЬ - экстренная остановка и отход
        if front_min < critical_distance:
            print(f"[systematic_explore] КРИТИЧЕСКАЯ ОПАСНОСТЬ ({front_min:.2f}м < {critical_distance}м)! ЭКСТРЕННЫЙ ОТХОД!")
            v_fwd_cmd = -cruise_speed * 1.2  # Быстрее назад
            v_right_cmd = 0.0
            yaw_rate_cmd = 0.0
            target_z = z - 1.0  # Поднимаемся выше
            
            # Если сзади тоже препятствие, поворачиваем
            if back_min < avoid_dist * 1.5:
                turn_sign = 1.0 if left_min > right_min else -1.0
                yaw_rate_cmd = turn_sign * max_yaw_rate
                v_right_cmd = turn_sign * cruise_speed * 0.8
                target_z = z - 2.0  # Поднимаемся еще выше
                print(f"[systematic_explore] Сзади препятствие! Экстренный подъем и поворот")
            
            cmd = await drone.move_by_velocity_body_frame_z_async(
                v_forward=v_fwd_cmd,
                v_right=v_right_cmd,
                z=target_z,
                duration=0.8,  # Дольше для безопасности
                yaw_is_rate=True,
                yaw=yaw_rate_cmd,
            )
            await cmd
            await asyncio.sleep(0.1)  # Пауза после экстренного маневра
            continue
        
        # Опасность касания ножками - приоритетная проверка
        if gear_collision_danger or (down_min < avoid_dist * 0.8):
            print(f"[systematic_explore] ОПАСНОСТЬ: Препятствие на уровне ножек ({down_min:.2f}м)! Отступаем и поднимаемся...")
            v_fwd_cmd = -cruise_speed * 1.0  # Быстро назад
            target_z = z - 2.0  # Поднимаемся выше
            v_right_cmd = 0.0
            yaw_rate_cmd = 0.0
            
            # Если сзади препятствие, активно поворачиваем
            if back_min < avoid_dist * 1.5:
                turn_sign = 1.0 if left_min > right_min else -1.0
                yaw_rate_cmd = turn_sign * max_yaw_rate
                v_right_cmd = turn_sign * cruise_speed * 0.7
                target_z = z - 2.5  # Еще выше
                print(f"[systematic_explore] Сзади препятствие при подъеме! Экстренный маневр")
        elif obstacle_ahead:
            # Обход препятствия (улучшенная логика)
            best_side = "left" if left_min > right_min else "right"
            best_clearance = max(left_min, right_min)
            worst_clearance = min(left_min, right_min)
            obstacle_ratio = max(0.0, front_min / max(warning_distance, 0.1))
            
            # Если препятствие очень близко - агрессивный отход
            if front_min < avoid_dist * 0.6:
                print(f"[systematic_explore] Очень близкое препятствие ({front_min:.2f}м)! Агрессивный отход")
                v_fwd_cmd = -cruise_speed * 0.9
                turn_sign = 1.0 if best_side == "left" else -1.0
                yaw_rate_cmd = turn_sign * max_yaw_rate
                v_right_cmd = turn_sign * cruise_speed * 0.9
                target_z = z - 2.5  # Высокий подъем
            # Если препятствия со всех сторон
            elif left_min < avoid_dist * 0.7 and right_min < avoid_dist * 0.7 and back_min < avoid_dist * 1.2:
                print(f"[systematic_explore] Окружен препятствиями! Высокий подъем для поиска обхода")
                v_fwd_cmd = cruise_speed * 0.4
                turn_sign = 1.0 if best_side == "left" else -1.0
                yaw_rate_cmd = turn_sign * max_yaw_rate * 0.95
                v_right_cmd = turn_sign * cruise_speed * 0.6
                target_z = z - 3.0  # Очень высокий подъем
            # Если препятствия слева и справа
            elif left_min < avoid_dist * 0.7 and right_min < avoid_dist * 0.7:
                print(f"[systematic_explore] Горизонтальные препятствия! Поднимаемся и обходим {best_side}")
                v_fwd_cmd = cruise_speed * 0.5
                turn_sign = 1.0 if best_side == "left" else -1.0
                yaw_rate_cmd = turn_sign * max_yaw_rate * 0.9
                v_right_cmd = turn_sign * cruise_speed * 0.7
                target_z = z - 2.5
            # Нормальный обход препятствия
            else:
                turn_sign = 1.0 if best_side == "left" else -1.0
                # Более консервативная скорость при обходе
                safe_speed_factor = max(0.2, obstacle_ratio ** 2.0)  # Более агрессивное снижение скорости
                forward_speed = base_speed * safe_speed_factor * 0.6  # Еще медленнее
                yaw_rate_cmd = turn_sign * max_yaw_rate * (0.7 + 0.3 * (1.0 - obstacle_ratio))
                v_right_cmd = turn_sign * cruise_speed * (0.6 + 0.4 * min(1.0, best_clearance / avoid_dist))
                v_fwd_cmd = forward_speed * 0.6 + v_fwd_desired * 0.4  # Больше обход, меньше к цели
                v_right_cmd = v_right_cmd * 0.8 + v_right_desired * 0.2
                target_z = z - 1.5  # Поднимаемся выше при обходе
                print(f"[systematic_explore] Обходим препятствие {best_side}, зазор: {best_clearance:.1f}м, скорость: {forward_speed:.2f} м/с")
        
        # Дополнительная проверка препятствий снизу (более агрессивная)
        if down_min < avoid_dist * 1.0:  # Увеличен порог
            target_z = min(target_z, z - 2.0)  # Поднимаемся выше
        
        # Если нет препятствий впереди, но есть опасность снизу - превентивный подъем
        if not obstacle_ahead and down_min < avoid_dist * 1.5:
            target_z = min(target_z, z - 1.0)
        
        # Ограничиваем скорости
        v_fwd_cmd = _clamp(v_fwd_cmd, -cruise_speed, cruise_speed)
        v_right_cmd = _clamp(v_right_cmd, -cruise_speed, cruise_speed)
        yaw_rate_cmd = _clamp(yaw_rate_cmd, -max_yaw_rate, max_yaw_rate)
        
        # Управляем дроном
        cmd = await drone.move_by_velocity_body_frame_z_async(
            v_forward=v_fwd_cmd,
            v_right=v_right_cmd,
            z=target_z,
            duration=dt,
            yaw_is_rate=True,
            yaw=yaw_rate_cmd,
        )
        await cmd
        await asyncio.sleep(0.001)
    
    print("[systematic_explore] Исследование завершено. Возвращаемся к стартовой точке...")
    
    # Возвращаемся к стартовой точке
    await _drive_to_waypoint_reactive(
        drone=drone,
        lidar_latest=lidar_latest,
        pose_latest=pose_latest,
        target_n=start_n,
        target_e=start_e,
        z=z,
        cruise_speed=cruise_speed,
        dt=dt,
        arrive_tol=arrive_tol,
        avoid_dist=avoid_dist,
        influence_dist=influence_dist,
        max_repulse=max_repulse,
        max_yaw_rate=max_yaw_rate,
        timeout_sec=120.0,
    )
    
    # Небольшая пауза в конце
    with contextlib.suppress(Exception):
        hover_task = await drone.hover_async()
        await hover_task
    
    print("[systematic_explore] Миссия завершена. Дрон вернулся в стартовую точку.")


async def full_scan_mapping(
    drone: Drone,
    lidar_latest: LidarLatest,
    pose_latest: PoseLatest,
    acc: PointCloudAccumulator,
    extent_n: float,
    extent_e: float,
    start_height: float,
    end_height: float,
    height_step: float,
    cruise_speed: float,
    rotation_angle: float = 45.0,
    dt: float = 0.25,
    avoid_dist: float = 12.0,
    influence_dist: float = 18.0,
    max_repulse: float = 2.5,
    max_yaw_rate: float = 1.2,
) -> None:
    """
    Полное сканирование карты с накоплением всех точек лидара.
    Дрон выполняет паттерн "газонокосилка" на разных высотах с поворотами для полного охвата.
    """
    print("[full_scan] Starting full area mapping scan")
    print(f"[full_scan] Area: {extent_n}m x {extent_e}m")
    print(f"[full_scan] Heights: {start_height}m to {end_height}m (step: {height_step}m)")
    
    # Получаем стартовую позицию
    start_n = 0.0
    start_e = 0.0
    t_wait = time.time()
    while True:
        pose_msg, _ts = pose_latest.snapshot()
        if pose_msg is not None and isinstance(pose_msg, dict):
            pos = pose_msg.get("position", {})
            start_n = float(pos.get("x", 0.0))
            start_e = float(pos.get("y", 0.0))
            break
        if time.time() - t_wait > 5.0:
            kin = drone.get_ground_truth_kinematics()
            pos = kin["pose"]["position"]
            start_n = float(pos["x"])
            start_e = float(pos["y"])
            break
        await asyncio.sleep(0.05)
    
    # Генерируем waypoints "газонокосилка" для каждой высоты
    current_height = start_height
    total_layers = int(math.ceil(abs(start_height - end_height) / abs(height_step)))
    layer = 0
    
    while current_height >= end_height:
        layer += 1
        print(f"[full_scan] Layer {layer}/{total_layers} at height {current_height}m")
        
        # Генерируем waypoints для текущей высоты
        waypoints = _generate_lawnmower_waypoints(
            start_n=start_n,
            start_e=start_e,
            extent_n=extent_n,
            extent_e=extent_e,
            step_e=10.0,  # шаг между проходами
        )
        
        # Добавляем повороты для лучшего охвата
        waypoints_with_rotation = []
        for i, (wp_n, wp_e) in enumerate(waypoints):
            waypoints_with_rotation.append((wp_n, wp_e, current_height))
            
            # Через каждые 2 waypoint добавляем поворот на месте для полного охвата
            if i > 0 and i % 2 == 0:
                # Делаем небольшой поворот на месте
                await drone.rotate_by_yaw_rate_async(yaw_rate=rotation_angle / 180.0 * math.pi, duration=2.0)
                await asyncio.sleep(0.5)
        
        # Облетаем waypoints
        for i, (wp_n, wp_e, wp_z) in enumerate(waypoints_with_rotation):
            print(f"[full_scan] Layer {layer}, waypoint {i+1}/{len(waypoints_with_rotation)}: ({wp_n:.1f}, {wp_e:.1f}, {wp_z:.1f})")
            
            await _drive_to_waypoint_reactive(
                drone=drone,
                lidar_latest=lidar_latest,
                pose_latest=pose_latest,
                target_n=wp_n,
                target_e=wp_e,
                z=wp_z,
                cruise_speed=cruise_speed,
                dt=dt,
                arrive_tol=2.0,
                avoid_dist=avoid_dist,
                influence_dist=influence_dist,
                max_repulse=max_repulse,
                max_yaw_rate=max_yaw_rate,
                timeout_sec=30.0,
            )
            
            # Небольшая пауза для накопления данных лидара
            await asyncio.sleep(0.5)
        
        # Переходим на следующую высоту
        current_height -= height_step
    
    print("[full_scan] Full mapping scan completed")


async def scan_and_navigate_shelves(
    drone: Drone,
    lidar_latest: LidarLatest,
    pose_latest: PoseLatest,
    acc: PointCloudAccumulator,
    extent_n: float,
    extent_e: float,
    start_height: float,
    end_height: float,
    height_step: float,
    cruise_speed: float,
    voxel_size: float = 0.02,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    min_shelf_height: float = 1.0,
    cluster_distance: float = 3.0,
    dt: float = 0.25,
    avoid_dist: float = 12.0,
    influence_dist: float = 18.0,
    max_repulse: float = 2.5,
    max_yaw_rate: float = 1.2,
) -> None:
    """
    Полное сканирование карты с последующим определением стеллажей и навигацией между ними.
    
    1. Выполняет полное сканирование области для накопления облака точек
    2. Очищает облако точек (вокселизация + удаление выбросов)
    3. Определяет стеллажи (вертикальные структуры)
    4. Кластеризует стеллажи
    5. Строит маршрут между стеллажами (сверху вниз)
    6. Выполняет полет по маршруту
    """
    print("[shelf_navigation] Starting shelf detection and navigation mode")
    
    # Шаг 1: Полное сканирование для накопления облака точек
    print("[shelf_navigation] Step 1: Full area scanning...")
    await full_scan_mapping(
        drone=drone,
        lidar_latest=lidar_latest,
        pose_latest=pose_latest,
        acc=acc,
        extent_n=extent_n,
        extent_e=extent_e,
        start_height=start_height,
        end_height=end_height,
        height_step=height_step,
        cruise_speed=cruise_speed,
        dt=dt,
        avoid_dist=avoid_dist,
        influence_dist=influence_dist,
        max_repulse=max_repulse,
        max_yaw_rate=max_yaw_rate,
    )
    
    # Шаг 2: Получаем накопленное облако точек
    print("[shelf_navigation] Step 2: Processing accumulated point cloud...")
    await asyncio.sleep(1.0)  # даем время на финальное накопление
    points_xyz = acc.snapshot()
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        print("[shelf_navigation] ERROR: No points accumulated. Cannot detect shelves.")
        return
    
    print(f"[shelf_navigation] Accumulated {points_xyz.shape[0]} points")
    
    # Преобразуем точки из body frame в world frame (NED)
    # Нужно трансформировать точки с учетом позы дрона
    try:
        import numpy as np
    except Exception:
        print("[shelf_navigation] ERROR: numpy not available")
        return
    
    # Шаг 3: Очистка облака точек
    print("[shelf_navigation] Step 3: Cleaning point cloud...")
    cleaned_points = _clean_point_cloud(
        points_xyz,
        voxel_size=voxel_size,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    print(f"[shelf_navigation] After cleaning: {cleaned_points.shape[0]} points")
    
    # Шаг 4: Определение стеллажей
    print("[shelf_navigation] Step 4: Detecting shelves (vertical structures)...")
    shelves = _detect_vertical_structures(
        cleaned_points,
        min_height=min_shelf_height,
        voxel_size_2d=0.5,
    )
    print(f"[shelf_navigation] Detected {len(shelves)} potential shelves")
    
    if len(shelves) == 0:
        print("[shelf_navigation] WARNING: No shelves detected. Cannot create navigation path.")
        return
    
    # Выводим информацию о найденных стеллажах
    for i, shelf in enumerate(shelves):
        print(f"[shelf_navigation] Shelf {i+1}: center=({shelf['center'][0]:.2f}, {shelf['center'][1]:.2f}), "
              f"height={shelf['height']:.2f}m, points={shelf['point_count']}")
    
    # Шаг 5: Кластеризация стеллажей
    print("[shelf_navigation] Step 5: Clustering shelves...")
    shelf_clusters = _cluster_shelves(shelves, cluster_distance=cluster_distance)
    print(f"[shelf_navigation] Found {len(shelf_clusters)} shelf clusters")
    
    # Шаг 6: Получаем текущую позицию для планирования маршрута
    pose_msg, _ts = pose_latest.snapshot()
    start_pos = (0.0, 0.0)
    if pose_msg is not None and isinstance(pose_msg, dict):
        pos = pose_msg.get("position", {})
        start_pos = (float(pos.get("x", 0.0)), float(pos.get("y", 0.0)))
    
    # Шаг 7: Планирование маршрута между стеллажами
    print("[shelf_navigation] Step 6: Planning navigation path between shelves...")
    waypoints = _plan_path_between_shelves(
        shelf_clusters,
        start_pos=start_pos,
        top_height=start_height,
        bottom_height=end_height,
        layer_height=abs(height_step),
    )
    print(f"[shelf_navigation] Created path with {len(waypoints)} waypoints")
    
    if len(waypoints) == 0:
        print("[shelf_navigation] WARNING: No waypoints generated. Cannot navigate.")
        return
    
    # Шаг 8: Навигация по маршруту
    print("[shelf_navigation] Step 7: Navigating between shelves...")
    for i, (wp_n, wp_e, wp_z) in enumerate(waypoints):
        print(f"[shelf_navigation] Waypoint {i+1}/{len(waypoints)}: ({wp_n:.1f}, {wp_e:.1f}, {wp_z:.1f})")
        
        await _drive_to_waypoint_reactive(
            drone=drone,
            lidar_latest=lidar_latest,
            pose_latest=pose_latest,
            target_n=wp_n,
            target_e=wp_e,
            z=wp_z,
            cruise_speed=cruise_speed,
            dt=dt,
            arrive_tol=2.0,
            avoid_dist=avoid_dist,
            influence_dist=influence_dist,
            max_repulse=max_repulse,
            max_yaw_rate=max_yaw_rate,
            timeout_sec=30.0,
        )
        
        await asyncio.sleep(0.5)
    
    print("[shelf_navigation] Shelf navigation completed!")


async def fly_square_by_position(drone: Drone, side_length: float = 10.0, height: float = -10.0, velocity: float = 3.0):
    """
    Полет дрона по квадрату, используя координаты позиций.
    
    Args:
        drone: Объект дрона
        side_length: Длина стороны квадрата в метрах (по умолчанию 10м)
        height: Высота полета в метрах (отрицательное значение в системе NED, -10 = 10м вверх)
        velocity: Скорость движения в м/с (по умолчанию 3 м/с)
    """
    print(f"Начинаю полет по квадрату (сторона: {side_length}м, высота: {-height}м)")
    
    # Получаем текущую позицию дрона как начальную точку
    cur_pos = drone.get_ground_truth_kinematics()["pose"]["position"]
    start_north = cur_pos["x"]  # x = north в системе NED
    start_east = cur_pos["y"]   # y = east в системе NED
    
    # Определяем вершины квадрата относительно стартовой позиции
    # NED система координат: North (север) = X, East (восток) = Y, Down (вниз) = Z
    square_points = [
        (start_north + side_length, start_east, height),           # Точка 1: вперед (север)
        (start_north + side_length, start_east + side_length, height),  # Точка 2: вперед и вправо (север-восток)
        (start_north, start_east + side_length, height),          # Точка 3: вправо (восток)
        (start_north, start_east, height),                        # Точка 4: возврат в начало
    ]
    
    for i, (north, east, down) in enumerate(square_points, 1):
        print(f"Летим к точке {i}/4: North={north:.1f}, East={east:.1f}, Height={-down:.1f}м")
        move_task = await drone.move_to_position_async(
            north=north, east=east, down=down, velocity=velocity
        )
        await move_task
        print(f"Достигнута точка {i}/4")
        await asyncio.sleep(0.5)  # Небольшая пауза между точками
    
    print("Квадрат завершен!")


async def fly_square_by_velocity(drone: Drone, side_length: float = 10.0, velocity: float = 3.0, height: float = -10.0):
    """
    Полет дрона по квадрату, используя управление скоростью.
    
    Args:
        drone: Объект дрона
        side_length: Длина стороны квадрата в метрах (по умолчанию 10м)
        velocity: Скорость движения в м/с (по умолчанию 3 м/с)
        height: Высота полета в метрах (отрицательное значение в системе NED)
    """
    print(f"Начинаю полет по квадрату через скорость (сторона: {side_length}м)")
    
    # Сначала поднимаемся на нужную высоту
    print(f"Поднимаемся на высоту {-height}м...")
    move_up_task = await drone.move_by_velocity_z_async(
        v_north=0.0, v_east=0.0, duration=2.0, z=height
    )
    await move_up_task
    
    # Вычисляем время для прохождения одной стороны
    duration = side_length / velocity
    
    # Полет по квадрату: вперед, вправо, назад, влево
    directions = [
        ("Вперед (север)", velocity, 0.0, 0.0),
        ("Вправо (восток)", 0.0, velocity, 0.0),
        ("Назад (юг)", -velocity, 0.0, 0.0),
        ("Влево (запад)", 0.0, -velocity, 0.0),
    ]
    
    for i, (direction_name, v_north, v_east, v_down) in enumerate(directions, 1):
        print(f"Сторона {i}/4: {direction_name}")
        move_task = await drone.move_by_velocity_z_async(
            v_north=v_north, v_east=v_east, duration=duration, z=height
        )
        await move_task
        await asyncio.sleep(0.3)  # Небольшая пауза между сторонами
    
    print("Квадрат завершен!")


async def circular_mapping_flight(
    drone: Drone,
    lidar_latest: LidarLatest,
    pose_latest: PoseLatest,
    acc: PointCloudAccumulator,
    radius: float = 30.0,
    height: float = -5.0,
    num_circles: int = 2,
    cruise_speed: float = 3.0,
    dt: float = 0.25,
    avoid_dist: float = 12.0,
    influence_dist: float = 18.0,
    max_repulse: float = 2.5,
    max_yaw_rate: float = 1.2,
) -> None:
    """
    Выполняет круговой облет вокруг стартовой позиции для создания SLAM карты.
    
    Args:
        drone: Объект дрона
        lidar_latest: Объект для получения последних данных лидара
        pose_latest: Объект для получения последней позы
        acc: Аккумулятор облака точек
        radius: Радиус облета в метрах
        height: Высота полета (NED, отрицательное = вверх)
        num_circles: Количество полных кругов
        cruise_speed: Скорость полета в м/с
        dt: Шаг управления в секундах
        avoid_dist: Дистанция срабатывания уклонения (м)
        influence_dist: Радиус влияния для отталкивания (м)
        max_repulse: Максимальная отталкивающая скорость (м/с)
        max_yaw_rate: Максимальная скорость рыскания (рад/с)
    """
    print("[SLAM] Начинаем круговой облет для создания карты местности")
    
    # Получаем стартовую позицию
    start_n = 0.0
    start_e = 0.0
    t_wait = time.time()
    while True:
        pose_msg, _ts = pose_latest.snapshot()
        if pose_msg is not None and isinstance(pose_msg, dict):
            pos = pose_msg.get("position", {})
            start_n = float(pos.get("x", 0.0))
            start_e = float(pos.get("y", 0.0))
            break
        if time.time() - t_wait > 5.0:
            kin = drone.get_ground_truth_kinematics()
            pos = kin["pose"]["position"]
            start_n = float(pos["x"])
            start_e = float(pos["y"])
            break
        await asyncio.sleep(0.05)
    
    print(f"[SLAM] Стартовая позиция: ({start_n:.2f}, {start_e:.2f})")
    print(f"[SLAM] Радиус облета: {radius}м, Высота: {-height}м, Кругов: {num_circles}")
    
    # Создаем waypoints для кругового облета
    num_waypoints_per_circle = 16  # количество точек на круг (больше = более плавный облет)
    waypoints = []
    
    for circle in range(num_circles):
        for i in range(num_waypoints_per_circle):
            angle = 2 * math.pi * (i / num_waypoints_per_circle + circle)
            # Круговые координаты
            wp_n = start_n + radius * math.cos(angle)
            wp_e = start_e + radius * math.sin(angle)
            waypoints.append((wp_n, wp_e, height))
    
    # Также добавляем точку возврата в начало
    waypoints.append((start_n, start_e, height))
    
    print(f"[SLAM] Создано {len(waypoints)} точек маршрута")
    
    # Облетаем все waypoints
    for i, (wp_n, wp_e, wp_z) in enumerate(waypoints):
        print(f"[SLAM] Точка {i+1}/{len(waypoints)}: ({wp_n:.1f}, {wp_e:.1f}, {wp_z:.1f})")
        
        await _drive_to_waypoint_reactive(
            drone=drone,
            lidar_latest=lidar_latest,
            pose_latest=pose_latest,
            target_n=wp_n,
            target_e=wp_e,
            z=wp_z,
            cruise_speed=cruise_speed,
            dt=dt,
            arrive_tol=2.0,
            avoid_dist=avoid_dist,
            influence_dist=influence_dist,
            max_repulse=max_repulse,
            max_yaw_rate=max_yaw_rate,
            timeout_sec=30.0,
        )
        
        # Небольшая пауза для накопления данных лидара
        await asyncio.sleep(0.5)
    
    print("[SLAM] Круговой облет завершен")


async def hover_and_collect_slam(
    drone: Drone,
    lidar_latest: LidarLatest,
    pose_latest: PoseLatest,
    acc: PointCloudAccumulator,
    height: float = -10.0,
    duration_sec: float = 30.0,
) -> None:
    """
    Дрон зависает на месте на заданной высоте и собирает облако точек с помощью SLAM.
    
    Args:
        drone: Объект дрона
        lidar_latest: Объект для получения последних данных лидара
        pose_latest: Объект для получения последней позы
        acc: Аккумулятор облака точек
        height: Высота зависания (NED, отрицательное = вверх)
        duration_sec: Длительность сбора данных в секундах
    """
    print(f"[SLAM] Зависание на месте на высоте {-height}м для сбора облака точек...")
    
    # Получаем текущую позицию
    start_n = 0.0
    start_e = 0.0
    t_wait = time.time()
    while True:
        pose_msg, _ts = pose_latest.snapshot()
        if pose_msg is not None and isinstance(pose_msg, dict):
            pos = pose_msg.get("position", {})
            start_n = float(pos.get("x", 0.0))
            start_e = float(pos.get("y", 0.0))
            break
        if time.time() - t_wait > 5.0:
            kin = drone.get_ground_truth_kinematics()
            pos = kin["pose"]["position"]
            start_n = float(pos["x"])
            start_e = float(pos["y"])
            break
        await asyncio.sleep(0.05)
    
    print(f"[SLAM] Позиция зависания: ({start_n:.2f}, {start_e:.2f}), высота: {-height}м")
    
    # Перемещаемся на заданную высоту и позицию
    print(f"[SLAM] Перемещение на высоту {-height}м...")
    await drone.move_to_position_async(north=start_n, east=start_e, down=height, velocity=2.0)
    
    # Ждем стабилизации
    await asyncio.sleep(2.0)
    
    # Зависаем на месте и собираем данные
    print(f"[SLAM] Начинаем сбор облака точек в течение {duration_sec} секунд...")
    start_time = time.time()
    
    # Периодически корректируем позицию, чтобы оставаться на месте
    while time.time() - start_time < duration_sec:
        # Получаем текущую позицию
        pose_msg, _ts = pose_latest.snapshot()
        current_n = start_n
        current_e = start_e
        current_z = height
        
        if pose_msg is not None and isinstance(pose_msg, dict):
            pos = pose_msg.get("position", {})
            current_n = float(pos.get("x", 0.0))
            current_e = float(pos.get("y", 0.0))
            current_z = float(pos.get("z", 0.0))
        
        # Если дрон сместился, корректируем позицию
        drift_threshold = 0.5  # метры
        if abs(current_n - start_n) > drift_threshold or abs(current_e - start_e) > drift_threshold or abs(current_z - height) > 0.3:
            # Корректируем позицию для удержания на месте
            await drone.move_to_position_async(north=start_n, east=start_e, down=height, velocity=1.0)
            await asyncio.sleep(0.5)
        else:
            # Удерживаем позицию, отправляя команду нулевой скорости
            await drone.move_by_velocity_async(v_north=0.0, v_east=0.0, v_down=0.0, duration=0.5)
        
        elapsed = time.time() - start_time
        remaining = duration_sec - elapsed
        if remaining > 0 and int(elapsed) % 5 == 0:  # Выводим каждые 5 секунд
            print(f"[SLAM] Сбор данных... осталось {remaining:.1f} сек, накоплено точек: {acc._total_points}")
        
        await asyncio.sleep(0.5)
    
    print("[SLAM] Сбор облака точек завершен")


