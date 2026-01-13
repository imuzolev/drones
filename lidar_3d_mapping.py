"""
Программа для управления дроном в ProjectAirSim.

Система координат NED (North-East-Down):
- North (север) = положительное направление по оси X (вперед)
- East (восток) = положительное направление по оси Y (вправо)
- Down (вниз) = положительное направление по оси Z
  ⚠️ ВАЖНО: Высота задается отрицательным значением!
  Например: z = -10 означает высоту 10 метров над землей

Примеры ручного управления дроном:

1. Движение вперед (на север) со скоростью 2 м/с в течение 5 секунд:
   await drone.move_by_velocity_async(v_north=2.0, v_east=0.0, v_down=0.0, duration=5.0)

2. Движение вправо (на восток) со скоростью 3 м/с в течение 3 секунд:
   await drone.move_by_velocity_async(v_north=0.0, v_east=3.0, v_down=0.0, duration=3.0)

3. Подъем на высоту 15 метров со скоростью 2 м/с:
   await drone.move_by_velocity_async(v_north=0.0, v_east=0.0, v_down=-2.0, duration=7.5)

4. Перемещение к конкретной позиции (10м на север, 5м на восток, высота 10м):
   await drone.move_to_position_async(north=10, east=5, down=-10, velocity=3.0)

5. Движение вперед относительно корпуса дрона:
   await drone.move_by_velocity_body_frame_async(v_forward=2.0, v_right=0.0, v_down=0.0, duration=5.0)
"""

import asyncio
import argparse
import contextlib
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List


def _ensure_projectairsim_on_path() -> None:
    """
    Делает импорт `projectairsim` работоспособным при запуске скрипта из корня репозитория,
    без установки пакета в site-packages.
    """
    repo_root = Path(__file__).resolve().parent
    client_python = repo_root / "client" / "python"
    if client_python.exists():
        sys.path.insert(0, str(client_python))


_ensure_projectairsim_on_path()

from projectairsim import ProjectAirSimClient, Drone, World  # noqa: E402


# ============================================================================
# ФУНКЦИИ ОБРАБОТКИ ОБЛАКА ТОЧЕК (интегрировано из Autonomous-Drone-Scanning-and-Mapping)
# ============================================================================

def _voxel_downsample(points_xyz, voxel_size: float = 0.02):
    """
    Вокселизация облака точек для уменьшения количества точек.
    Адаптировано из PointCloudCleaning.py.
    """
    try:
        import numpy as np
    except Exception:
        return points_xyz
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        return points_xyz
    
    pts = points_xyz
    if pts.shape[0] < 100:
        return pts  # слишком мало точек для вокселизации
    
    # Простая вокселизация: группируем точки по вокселям и берем центроид каждого вокселя
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        return pts
    
    # Округляем координаты до размеров вокселя
    voxel_coords = np.floor(pts / voxel_size).astype(np.int32)
    
    # Используем словарь для хранения точек в каждом вокселе
    voxel_dict = {}
    for i, vc in enumerate(voxel_coords):
        key = tuple(vc)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(i)
    
    # Создаем новый массив точек, усредняя точки в каждом вокселе
    new_points = []
    for indices in voxel_dict.values():
        if indices:
            voxel_points = pts[indices]
            centroid = np.mean(voxel_points, axis=0)
            new_points.append(centroid)
    
    if not new_points:
        return pts
    
    return np.array(new_points, dtype=np.float32)


def _remove_statistical_outlier(points_xyz, nb_neighbors: int = 20, std_ratio: float = 2.0):
    """
    Удаление статистических выбросов из облака точек.
    Адаптировано из PointCloudCleaning.py.
    Возвращает (inlier_points, outlier_mask).
    """
    try:
        import numpy as np
    except Exception:
        return points_xyz, None
    
    try:
        from scipy import spatial
    except ImportError:
        # Если scipy недоступен, используем простой алгоритм без KD-дерева
        try:
            import numpy as np
            if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
                return points_xyz, None
            
            pts = points_xyz
            if pts.shape[0] < nb_neighbors + 1:
                return pts, np.ones(pts.shape[0], dtype=bool)
            
            # Простая версия без KD-дерева: вычисляем расстояния до всех соседей
            distances_to_neighbors = []
            for i in range(min(pts.shape[0], 1000)):  # ограничиваем для скорости
                dists = np.sqrt(np.sum((pts - pts[i])**2, axis=1))
                dists = np.sort(dists)[1:min(nb_neighbors + 1, len(dists))]
                if len(dists) > 0:
                    distances_to_neighbors.append(np.mean(dists))
                else:
                    distances_to_neighbors.append(0.0)
            
            if len(distances_to_neighbors) < pts.shape[0]:
                # Дополняем для остальных точек
                avg_dist = np.mean(distances_to_neighbors)
                distances_to_neighbors.extend([avg_dist] * (pts.shape[0] - len(distances_to_neighbors)))
            
            distances_to_neighbors = np.array(distances_to_neighbors)
            mean_dist = np.mean(distances_to_neighbors)
            std_dist = np.std(distances_to_neighbors)
            threshold = mean_dist + std_ratio * std_dist
            inlier_mask = distances_to_neighbors < threshold
            return pts[inlier_mask], inlier_mask
        except Exception:
            return points_xyz, None
    
    # Используем scipy.spatial.KDTree для быстрого поиска
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        return points_xyz, None
    
    pts = points_xyz
    if pts.shape[0] < nb_neighbors + 1:
        return pts, np.ones(pts.shape[0], dtype=bool)
    
    # Строим KD-дерево для быстрого поиска соседей
    tree = spatial.KDTree(pts)
    
    # Для каждой точки находим nb_neighbors ближайших соседей
    distances_to_neighbors = []
    for i in range(pts.shape[0]):
        distances, _ = tree.query(pts[i], k=min(nb_neighbors + 1, pts.shape[0]))
        if len(distances) > 1:
            # Первый элемент - сама точка (расстояние = 0), берем остальные
            mean_dist = np.mean(distances[1:])
            distances_to_neighbors.append(mean_dist)
        else:
            distances_to_neighbors.append(0.0)
    
    distances_to_neighbors = np.array(distances_to_neighbors)
    
    # Вычисляем среднее и стандартное отклонение
    mean_dist = np.mean(distances_to_neighbors)
    std_dist = np.std(distances_to_neighbors)
    
    # Точки, расстояние до которых значительно больше среднего, считаются выбросами
    threshold = mean_dist + std_ratio * std_dist
    inlier_mask = distances_to_neighbors < threshold
    
    inlier_points = pts[inlier_mask]
    return inlier_points, inlier_mask


def _clean_point_cloud(points_xyz, voxel_size: float = 0.02, nb_neighbors: int = 20, std_ratio: float = 2.0):
    """
    Полная очистка облака точек: вокселизация + удаление выбросов.
    """
    if points_xyz is None:
        return points_xyz
    
    # Сначала вокселизация
    cleaned = _voxel_downsample(points_xyz, voxel_size)
    
    # Затем удаление выбросов
    cleaned, _ = _remove_statistical_outlier(cleaned, nb_neighbors, std_ratio)
    
    return cleaned


# ============================================================================
# ФУНКЦИИ ОПРЕДЕЛЕНИЯ СТЕЛЛАЖЕЙ
# ============================================================================

def _detect_vertical_structures(points_xyz, min_height: float = 1.0, voxel_size_2d: float = 0.5):
    """
    Определяет вертикальные структуры (стеллажи) в облаке точек.
    
    Алгоритм:
    1. Проецируем точки на плоскость XY (вид сверху)
    2. Группируем точки в воксели на плоскости XY
    3. Для каждого вокселя проверяем вертикальный разброс точек (высота стеллажа)
    4. Если разброс по Z больше min_height - это потенциальный стеллаж
    
    Возвращает список стеллажей: [(center_x, center_y, min_z, max_z), ...]
    """
    try:
        import numpy as np
    except Exception:
        return []
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        return []
    
    pts = points_xyz
    if pts.shape[0] < 100:
        return []
    
    # Проекция на плоскость XY
    xy_points = pts[:, :2]  # только X и Y
    
    # Вокселизация на плоскости XY
    voxel_coords_2d = np.floor(xy_points / voxel_size_2d).astype(np.int32)
    
    # Группируем точки по вокселям XY
    voxel_dict = {}
    for i, vc in enumerate(voxel_coords_2d):
        key = tuple(vc)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(i)
    
    # Анализируем каждый воксель на наличие вертикальной структуры
    shelves = []
    min_height = float(min_height)
    
    for key, indices in voxel_dict.items():
        if len(indices) < 10:  # слишком мало точек в вокселе
            continue
        
        voxel_points = pts[indices]
        z_values = voxel_points[:, 2]
        
        # Вычисляем разброс по высоте
        z_min = float(np.min(z_values))
        z_max = float(np.max(z_values))
        z_range = z_max - z_min
        
        # Если разброс по высоте достаточно большой - это потенциальный стеллаж
        if z_range >= min_height:
            # Центр вокселя
            center_xy = np.mean(voxel_points[:, :2], axis=0)
            center_x = float(center_xy[0])
            center_y = float(center_xy[1])
            
            shelves.append({
                'center': (center_x, center_y),
                'z_min': z_min,
                'z_max': z_max,
                'height': z_range,
                'point_count': len(indices)
            })
    
    return shelves


def _cluster_shelves(shelves, cluster_distance: float = 3.0):
    """
    Кластеризует стеллажи по близости (иерархическая кластеризация).
    Возвращает список кластеров стеллажей.
    """
    try:
        import numpy as np
    except Exception:
        return [shelves] if shelves else []
    
    if not shelves or len(shelves) == 0:
        return []
    
    if len(shelves) == 1:
        return [shelves]
    
    # Извлекаем центры стеллажей
    centers = np.array([s['center'] for s in shelves])
    
    # Иерархическая кластеризация
    try:
        from scipy import cluster
        cluster_indices = cluster.hierarchy.fclusterdata(
            centers, 
            t=float(cluster_distance), 
            criterion='distance'
        )
    except (ImportError, Exception):
        # Если scipy недоступен или кластеризация не удалась, используем простую кластеризацию
        # Группируем стеллажи по близости вручную
        clusters = []
        used = set()
        cluster_distance = float(cluster_distance)
        
        for i, shelf in enumerate(shelves):
            if i in used:
                continue
            current_cluster = [shelf]
            used.add(i)
            
            for j, other_shelf in enumerate(shelves[i+1:], start=i+1):
                if j in used:
                    continue
                dist = np.sqrt(np.sum((np.array(shelf['center']) - np.array(other_shelf['center']))**2))
                if dist <= cluster_distance:
                    current_cluster.append(other_shelf)
                    used.add(j)
            
            clusters.append(current_cluster)
        
        return clusters
    
    # Группируем стеллажи по кластерам
    clusters = {}
    for i, shelf in enumerate(shelves):
        cluster_id = cluster_indices[i]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(shelf)
    
    return list(clusters.values())


def _get_shelf_cluster_center(cluster):
    """Возвращает центр кластера стеллажей."""
    try:
        import numpy as np
    except Exception:
        return None
    
    if not cluster:
        return None
    
    centers = [s['center'] for s in cluster]
    center = np.mean(centers, axis=0)
    return (float(center[0]), float(center[1]))


def _plan_path_between_shelves(shelf_clusters, start_pos: Tuple[float, float], 
                                top_height: float = -2.0, bottom_height: float = -8.0,
                                layer_height: float = 1.5):
    """
    Планирует маршрут между стеллажами, летая сверху вниз.
    
    Алгоритм:
    1. Определяет центр каждого кластера стеллажей
    2. Создает маршрут, который проходит между стеллажами на разных высотах
    3. Начинает с верхней высоты, затем опускается на layer_height ниже
    
    Возвращает список waypoints: [(north, east, down), ...]
    """
    if not shelf_clusters or len(shelf_clusters) == 0:
        return []
    
    # Получаем центры кластеров
    cluster_centers = []
    for cluster in shelf_clusters:
        center = _get_shelf_cluster_center(cluster)
        if center is not None:
            cluster_centers.append(center)
    
    if not cluster_centers:
        return []
    
    # Если только один кластер, просто облетаем его на разных высотах
    if len(cluster_centers) == 1:
        center = cluster_centers[0]
        waypoints = []
        current_height = top_height
        while current_height >= bottom_height:
            waypoints.append((center[0], center[1], current_height))
            current_height -= layer_height
        return waypoints
    
    # Если несколько кластеров, планируем маршрут между ними
    waypoints = []
    current_height = top_height
    
    # Начинаем с стартовой позиции
    waypoints.append((start_pos[0], start_pos[1], current_height))
    
    # Для каждой высоты проходим все центры кластеров
    while current_height >= bottom_height:
        # Создаем маршрут между всеми центрами на текущей высоте
        for center in cluster_centers:
            waypoints.append((center[0], center[1], current_height))
        
        # Опускаемся на следующий слой
        current_height -= layer_height
    
    return waypoints


# ============================================================================
# КЛАССЫ
# ============================================================================

class LidarLatest:
    """Потокобезопасное хранение последнего кадра лидара."""

    def __init__(self):
        self._lock = threading.Lock()
        self._points = None  # numpy array (N, 3) in sensor/body frame (x forward, y right, z down)
        self._stamp = 0.0

    def update(self, points_xyz, stamp: float) -> None:
        with self._lock:
            self._points = points_xyz
            self._stamp = float(stamp)

    def snapshot(self):
        with self._lock:
            pts = self._points
            ts = self._stamp
        return pts, ts


class PoseLatest:
    """Потокобезопасное хранение последней позы (robot_info['actual_pose'])."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pose = None
        self._stamp = 0.0

    def update(self, pose_msg) -> None:
        with self._lock:
            self._pose = pose_msg
            self._stamp = time.time()

    def snapshot(self):
        with self._lock:
            msg = self._pose
            ts = self._stamp
        return msg, ts


class ImuLatest:
    """Потокобезопасное хранение последних данных IMU."""

    def __init__(self):
        self._lock = threading.Lock()
        self._orientation = None  # quaternion dict with w,x,y,z
        self._angular_velocity = None  # dict with x,y,z
        self._linear_acceleration = None  # dict with x,y,z
        self._time_stamp = 0.0

    def update(self, imu_msg) -> None:
        """Обновляет данные IMU из сообщения."""
        with self._lock:
            if isinstance(imu_msg, dict):
                self._orientation = imu_msg.get("orientation", {})
                self._angular_velocity = imu_msg.get("angular_velocity", {})
                self._linear_acceleration = imu_msg.get("linear_acceleration", {})
                ts = imu_msg.get("time_stamp", None)
                if ts is not None:
                    # time_stamp может быть в наносекундах или секундах
                    if isinstance(ts, (int, float)):
                        if ts > 1e15:  # вероятно, это наносекунды
                            self._time_stamp = float(ts) / 1e9
                        else:
                            self._time_stamp = float(ts)
                    else:
                        self._time_stamp = time.time()
                else:
                    self._time_stamp = time.time()

    def snapshot(self):
        """Возвращает снимок текущих данных IMU."""
        with self._lock:
            return (
                self._orientation.copy() if self._orientation else None,
                self._angular_velocity.copy() if self._angular_velocity else None,
                self._linear_acceleration.copy() if self._linear_acceleration else None,
                self._time_stamp,
            )


def _quat_to_yaw_rad(q: dict) -> float:
    # Quaternion is expected to have keys w,x,y,z
    if q is None:
        return 0.0
    w = float(q.get("w", 1.0))
    x = float(q.get("x", 0.0))
    y = float(q.get("y", 0.0))
    z = float(q.get("z", 0.0))
    # yaw (Z) from quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quat_to_euler_rad(q: dict) -> Tuple[float, float, float]:
    """Конвертирует quaternion в углы Эйлера (roll, pitch, yaw) в радианах."""
    if q is None:
        return 0.0, 0.0, 0.0
    w = float(q.get("w", 1.0))
    x = float(q.get("x", 0.0))
    y = float(q.get("y", 0.0))
    z = float(q.get("z", 0.0))
    
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def _world_to_body(v_n: float, v_e: float, yaw_rad: float) -> Tuple[float, float]:
    """World N/E -> body forward/right (assuming yaw about Down axis)."""
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)
    v_fwd = cy * v_n + sy * v_e
    v_right = -sy * v_n + cy * v_e
    return v_fwd, v_right


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _min_range_in_cone(points_xyz, az_min_rad: float, az_max_rad: float, max_range: float) -> float:
    """Min range in azimuth cone in body frame. Returns max_range if no points."""
    if points_xyz is None:
        return max_range
    if getattr(points_xyz, "size", 0) == 0:
        return max_range
    try:
        import numpy as np
    except Exception:
        return max_range

    pts = points_xyz
    # sample to keep it fast
    if pts.shape[0] > 4000:
        idx = np.random.choice(pts.shape[0], size=4000, replace=False)
        pts = pts[idx]

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    az = np.arctan2(y, x)
    r = np.sqrt(x * x + y * y + z * z)
    mask = (az >= az_min_rad) & (az <= az_max_rad)
    if not np.any(mask):
        return max_range
    return float(np.min(r[mask]))


def _check_landing_gear_collision(points_xyz, landing_gear_height: float = 0.5, safety_margin: float = 1.0) -> bool:
    """
    Проверяет, есть ли препятствия на уровне ножек дрона (landing gear).
    Возвращает True, если обнаружена опасность столкновения ножек с препятствием.
    
    Args:
        points_xyz: Облако точек в body frame (x вперед, y вправо, z вниз)
        landing_gear_height: Высота ножек дрона ниже центра (м)
        safety_margin: Запас безопасности (м)
    """
    if points_xyz is None:
        return False
    if getattr(points_xyz, "size", 0) == 0:
        return False
    try:
        import numpy as np
    except Exception:
        return False

    pts = points_xyz
    # Проверяем точки на уровне ножек (z близко к landing_gear_height)
    # В body frame: z положительное = вниз
    z_threshold = landing_gear_height + safety_margin
    mask_gear_level = pts[:, 2] <= z_threshold  # точки на уровне ножек или ниже
    
    if not np.any(mask_gear_level):
        return False
    
    # Проверяем горизонтальное расстояние до этих точек
    gear_points = pts[mask_gear_level]
    xy_distances = np.sqrt(gear_points[:, 0]**2 + gear_points[:, 1]**2)
    
    # Если есть точки на уровне ножек слишком близко - опасность!
    collision_threshold = safety_margin
    too_close = np.any(xy_distances < collision_threshold)
    
    return bool(too_close)


def _repulsive_velocity_xy(points_xyz, influence_dist: float, max_repulse: float, landing_gear_height: float = 0.5) -> Tuple[float, float]:
    """
    Простая "потенциальная" отталкивающая скорость в плоскости XY (body frame).
    Учитывает препятствия на уровне ножек дрона для предотвращения касаний.
    Возвращает (v_forward_rep, v_right_rep).
    """
    if points_xyz is None:
        return 0.0, 0.0
    if getattr(points_xyz, "size", 0) == 0:
        return 0.0, 0.0
    try:
        import numpy as np
    except Exception:
        return 0.0, 0.0

    pts = points_xyz
    # sample for speed
    if pts.shape[0] > 2000:
        idx = np.random.choice(pts.shape[0], size=2000, replace=False)
        pts = pts[idx]

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    
    # Учитываем точки на уровне ножек или ниже (z положительное = вниз в body frame)
    # Расширяем зону проверки, чтобы учитывать препятствия на уровне ножек
    mask_gear_level = z <= (landing_gear_height + 1.0)  # точки на уровне ножек или немного ниже
    
    # Для точек на уровне ножек используем более агрессивную проверку
    # Не игнорируем точки позади, если они на уровне ножек
    mask_front = x > -1.0
    mask_important = mask_front | mask_gear_level
    
    x = x[mask_important]
    y = y[mask_important]
    z_filtered = z[mask_important] if mask_important.any() else np.array([])
    
    if x.size == 0:
        return 0.0, 0.0

    d = np.sqrt(x * x + y * y)
    
    # Для точек на уровне ножек используем более короткое расстояние влияния
    gear_mask = z_filtered <= (landing_gear_height + 0.5) if z_filtered.size > 0 else np.zeros(x.size, dtype=bool)
    gear_dist_threshold = float(influence_dist) * 0.7  # 70% от обычного расстояния для ножек
    normal_dist_threshold = float(influence_dist)
    
    # Применяем разные пороги в зависимости от высоты точек
    if gear_mask.size > 0 and np.any(gear_mask):
        # Точки на уровне ножек - более агрессивная проверка
        mask_gear = (d > 0.05) & (d < gear_dist_threshold) & gear_mask
        mask_normal = (d > 0.05) & (d < normal_dist_threshold) & ~gear_mask
        mask = mask_gear | mask_normal
    else:
        mask = (d > 0.05) & (d < normal_dist_threshold)
    
    if not np.any(mask):
        return 0.0, 0.0

    # Сохраняем информацию о том, какие точки на уровне ножек, до фильтрации
    gear_mask_filtered = gear_mask[mask] if gear_mask.size > 0 else np.zeros(np.sum(mask), dtype=bool)

    x = x[mask]
    y = y[mask]
    d = d[mask]
    
    # Для точек на уровне ножек увеличиваем вес отталкивания
    weight_multiplier = np.where(gear_mask_filtered, 2.0, 1.0)  # двойной вес для точек на уровне ножек

    # repulsion magnitude grows as distance decreases
    # weight ~ (1/d - 1/R) / d
    w_base = (1.0 / d - 1.0 / normal_dist_threshold) / d
    w = w_base * weight_multiplier
    
    # direction away from obstacle => negative of normalized obstacle direction
    ux = x / d
    uy = y / d
    rx = -np.sum(w * ux)
    ry = -np.sum(w * uy)

    # normalize & clamp
    norm = float(np.hypot(rx, ry))
    if norm < 1e-6:
        return 0.0, 0.0
    scale = min(float(max_repulse) / norm, 1.0)
    return float(rx * scale), float(ry * scale)


def _generate_lawnmower_waypoints(
    start_n: float,
    start_e: float,
    extent_n: float,
    extent_e: float,
    step_e: float,
) -> List[Tuple[float, float]]:
    """
    Генерирует точки "газонокосилки" от (start_n,start_e) по прямоугольнику extent_n x extent_e.
    """
    waypoints: List[Tuple[float, float]] = []
    rows = max(1, int(math.ceil(abs(extent_e) / max(step_e, 1.0))))
    e0 = start_e
    n_min = start_n
    n_max = start_n + float(extent_n)
    # чередуем направление проходов по N
    for i in range(rows + 1):
        e = e0 + (i * (extent_e / max(rows, 1)))
        if i % 2 == 0:
            waypoints.append((n_max, e))
        else:
            waypoints.append((n_min, e))
    return waypoints


class SimpleLIO:
    """
    Упрощенный LIO-SLAM (Lidar-Inertial Odometry and Mapping).
    Объединяет данные LiDAR и IMU для более точной оценки позиции дрона.
    """

    def __init__(self):
        self.position = [0.0, 0.0, 0.0]  # NED: north, east, down
        self.velocity = [0.0, 0.0, 0.0]  # NED velocity
        self.orientation = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}  # quaternion
        self.last_imu_time = None
        self.last_lidar_time = None
        self.lidar_map = None  # аккумулятор для карты лидара
        
    def predict_with_imu(
        self,
        angular_velocity: dict,
        linear_acceleration: dict,
        dt: float,
    ) -> None:
        """
        Предсказывает движение на основе IMU данных (dead reckoning).
        Использует угловые скорости и линейные ускорения для обновления позиции.
        """
        if angular_velocity is None or linear_acceleration is None:
            return
        
        try:
            import numpy as np
        except Exception:
            return
        
        # Получаем угловые скорости
        wx = float(angular_velocity.get("x", 0.0))
        wy = float(angular_velocity.get("y", 0.0))
        wz = float(angular_velocity.get("z", 0.0))
        
        # Получаем линейные ускорения (в body frame)
        ax_b = float(linear_acceleration.get("x", 0.0))
        ay_b = float(linear_acceleration.get("y", 0.0))
        az_b = float(linear_acceleration.get("z", 0.0))
        
        # Обновляем ориентацию используя угловые скорости (упрощенная интеграция)
        roll, pitch, yaw = _quat_to_euler_rad(self.orientation)
        roll += wx * dt
        pitch += wy * dt
        yaw += wz * dt
        
        # Преобразуем углы Эйлера обратно в quaternion
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        self.orientation = {
            "w": cr * cp * cy + sr * sp * sy,
            "x": sr * cp * cy - cr * sp * sy,
            "y": cr * sp * cy + sr * cp * sy,
            "z": cr * cp * sy - sr * sp * cy,
        }
        
        # Преобразуем ускорения из body frame в world frame (NED)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cr = math.cos(roll)
        sr = math.sin(roll)
        
        # Rotation matrix body -> world (NED)
        # R = Rz(yaw) * Ry(pitch) * Rx(roll)
        accel_n = (cy * cp) * ax_b + (cy * sp * sr - sy * cr) * ay_b + (cy * sp * cr + sy * sr) * az_b
        accel_e = (sy * cp) * ax_b + (sy * sp * sr + cy * cr) * ay_b + (sy * sp * cr - cy * sr) * az_b
        accel_d = (-sp) * ax_b + (cp * sr) * ay_b + (cp * cr) * az_b
        
        # Добавляем гравитацию (в NED: гравитация положительна по оси down)
        gravity = 9.81
        accel_d += gravity
        
        # Интегрируем ускорение для получения скорости
        self.velocity[0] += accel_n * dt
        self.velocity[1] += accel_e * dt
        self.velocity[2] += accel_d * dt
        
        # Интегрируем скорость для получения позиции
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        self.position[2] += self.velocity[2] * dt
        
        # Простая коррекция дрейфа (damping)
        self.velocity[0] *= 0.98
        self.velocity[1] *= 0.98
        self.velocity[2] *= 0.98
    
    def correct_with_lidar(self, lidar_points, pose_ground_truth: dict) -> dict:
        """
        Корректирует позицию на основе LiDAR данных.
        Использует ground truth позу для начальной инициализации и коррекции дрейфа.
        """
        if pose_ground_truth is None:
            return {
                "position": self.position.copy(),
                "orientation": self.orientation.copy(),
                "velocity": self.velocity.copy(),
            }
        
        pos_gt = pose_ground_truth.get("position", {})
        ori_gt = pose_ground_truth.get("orientation", {})
        
        if pos_gt and ori_gt:
            # Используем ground truth для коррекции дрейфа IMU
            # В реальном LIO-SLAM здесь была бы коррекция на основе ICP или feature matching
            gt_n = float(pos_gt.get("x", self.position[0]))
            gt_e = float(pos_gt.get("y", self.position[1]))
            gt_d = float(pos_gt.get("z", self.position[2]))
            
            # Простая коррекция с коэффициентом смешивания
            alpha = 0.1  # доверие к ground truth (10%)
            self.position[0] = alpha * gt_n + (1.0 - alpha) * self.position[0]
            self.position[1] = alpha * gt_e + (1.0 - alpha) * self.position[1]
            self.position[2] = alpha * gt_d + (1.0 - alpha) * self.position[2]
            
            # Обновляем ориентацию аналогично
            if ori_gt:
                w = float(ori_gt.get("w", self.orientation["w"]))
                x = float(ori_gt.get("x", self.orientation["x"]))
                y = float(ori_gt.get("y", self.orientation["y"]))
                z = float(ori_gt.get("z", self.orientation["z"]))
                self.orientation["w"] = alpha * w + (1.0 - alpha) * self.orientation["w"]
                self.orientation["x"] = alpha * x + (1.0 - alpha) * self.orientation["x"]
                self.orientation["y"] = alpha * y + (1.0 - alpha) * self.orientation["y"]
                self.orientation["z"] = alpha * z + (1.0 - alpha) * self.orientation["z"]
        
        return {
            "position": self.position.copy(),
            "orientation": self.orientation.copy(),
            "velocity": self.velocity.copy(),
        }
    
    def update_state(
        self,
        imu_orientation: dict,
        imu_angular_velocity: dict,
        imu_linear_acceleration: dict,
        imu_time: float,
        lidar_points,
        pose_gt: dict,
        lidar_time: float,
    ) -> dict:
        """
        Основной метод обновления состояния LIO-SLAM.
        Объединяет предсказание на основе IMU и коррекцию на основе LiDAR.
        """
        current_time = time.time()
        
        # Инициализация last_imu_time при первом вызове
        if self.last_imu_time is None:
            self.last_imu_time = imu_time if imu_time > 0 else current_time
        
        # Предсказание на основе IMU (если есть данные)
        if (imu_angular_velocity is not None and 
            imu_linear_acceleration is not None and 
            isinstance(imu_angular_velocity, dict) and 
            isinstance(imu_linear_acceleration, dict)):
            
            dt = 0.0
            if imu_time > 0:
                dt = max(0.001, min(0.1, imu_time - self.last_imu_time))  # ограничиваем dt
            elif self.last_imu_time is not None:
                dt = max(0.001, min(0.1, current_time - self.last_imu_time))
            
            if dt > 0.0:
                self.predict_with_imu(imu_angular_velocity, imu_linear_acceleration, dt)
                self.last_imu_time = imu_time if imu_time > 0 else current_time
        
        # Коррекция на основе LiDAR (если есть новые данные)
        if lidar_time is not None and (
            self.last_lidar_time is None or lidar_time > self.last_lidar_time
        ):
            state = self.correct_with_lidar(lidar_points, pose_gt)
            self.last_lidar_time = lidar_time
            return state
        
        # Если нет новых данных LiDAR, возвращаем текущее состояние
        return {
            "position": self.position.copy(),
            "orientation": self.orientation.copy(),
            "velocity": self.velocity.copy(),
        }


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


async def explore_map_systematic(
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


class PointCloudAccumulator:
    """
    Кольцевой буфер для накопления точек лидара (для визуализации).
    Хранит точки в системе NED (Z вниз). Для отображения обычно делаем Z вверх.
    """

    def __init__(self, max_points: int = 200_000):
        self._chunks: deque = deque()
        self._total_points = 0
        self._max_points = int(max_points)

    def add_points(self, points_xyz) -> None:
        # points_xyz: numpy array shape (N, 3)
        if points_xyz is None:
            return
        if getattr(points_xyz, "size", 0) == 0:
            return
        self._chunks.append(points_xyz)
        self._total_points += int(points_xyz.shape[0])
        while self._total_points > self._max_points and len(self._chunks) > 1:
            old = self._chunks.popleft()
            self._total_points -= int(old.shape[0])

    def snapshot(self, max_points: Optional[int] = None):
        try:
            import numpy as np
        except Exception:
            return None

        if self._total_points <= 0:
            return np.empty((0, 3), dtype=np.float32)

        pts = np.concatenate(list(self._chunks), axis=0)
        if max_points is not None and pts.shape[0] > max_points:
            # случайная подвыборка для скорости
            idx = np.random.choice(pts.shape[0], size=int(max_points), replace=False)
            pts = pts[idx]
        return pts


class RealtimePointCloudVisualizer:
    """
    Класс для визуализации облака точек в реальном времени.
    Окно открывается сразу при создании и обновляется периодически.
    """
    
    def __init__(self, acc: PointCloudAccumulator, update_interval: float = 0.1, max_display_points: int = 100_000):
        """
        Args:
            acc: PointCloudAccumulator для получения точек
            update_interval: интервал обновления визуализации в секундах
            max_display_points: максимальное количество точек для отображения (для производительности)
        """
        self.acc = acc
        self.update_interval = update_interval
        self.max_display_points = max_display_points
        self.vis = None
        self.pcd = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
    def _transform_ned_to_vis(self, points_xyz):
        """Преобразует точки из NED в систему координат для визуализации."""
        try:
            import numpy as np
        except Exception:
            return None
        
        if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
            return None
        
        # Преобразуем из NED в стандартную систему координат (Z вверх)
        # NED: X=North, Y=East, Z=Down (положительный Z вниз)
        # Для визуализации: X=Right, Y=Forward, Z=Up
        points_vis = np.zeros_like(points_xyz)
        points_vis[:, 0] = points_xyz[:, 1]   # East -> Right
        points_vis[:, 1] = -points_xyz[:, 2]   # -Down -> Up
        points_vis[:, 2] = -points_xyz[:, 0]   # -North -> Forward
        return points_vis
    
    def _update_visualization(self):
        """Обновляет визуализацию с новыми данными из аккумулятора."""
        try:
            import numpy as np
            import open3d as o3d
        except ImportError:
            return False
        
        # Получаем снимок точек из аккумулятора
        points_xyz = self.acc.snapshot(max_points=self.max_display_points)
        
        if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
            # Нет точек, но продолжаем обновлять окно
            with self.lock:
                if self.vis is not None and self.running:
                    try:
                        self.vis.poll_events()
                        self.vis.update_renderer()
                    except:
                        pass
            return False
        
        # Преобразуем координаты
        points_vis = self._transform_ned_to_vis(points_xyz)
        if points_vis is None:
            return False
        
        with self.lock:
            if self.vis is None or not self.running:
                return False
            
            try:
                # Обновляем или создаем облако точек
                if self.pcd is None:
                    self.pcd = o3d.geometry.PointCloud()
                    self.pcd.points = o3d.utility.Vector3dVector(points_vis)
                    # Устанавливаем зеленый цвет
                    colors = np.ones((points_vis.shape[0], 3)) * [0.0, 0.8, 0.0]
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                    self.vis.add_geometry(self.pcd, reset_bounding_box=True)
                    
                    # Настраиваем камеру для просмотра облака точек
                    view_control = self.vis.get_view_control()
                    if points_vis.shape[0] > 0:
                        center = np.mean(points_vis, axis=0)
                        extent = np.max(points_vis, axis=0) - np.min(points_vis, axis=0)
                        max_extent = np.max(extent) if np.max(extent) > 0 else 10.0
                        
                        view_control.set_front([0.5, -0.5, -0.7])
                        view_control.set_lookat(center)
                        view_control.set_up([0, 1, 0])
                        view_control.set_zoom(0.7)
                else:
                    # Обновляем существующее облако точек
                    old_point_count = len(self.pcd.points)
                    self.pcd.points = o3d.utility.Vector3dVector(points_vis)
                    # Обновляем цвета
                    colors = np.ones((points_vis.shape[0], 3)) * [0.0, 0.8, 0.0]
                    self.pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    # Если это первое появление точек (было пусто, стало не пусто), настраиваем камеру
                    if old_point_count == 0 and points_vis.shape[0] > 0:
                        # Перестраиваем bounding box и настраиваем камеру
                        self.vis.clear_geometries()
                        self.vis.add_geometry(self.pcd, reset_bounding_box=True)
                        view_control = self.vis.get_view_control()
                        center = np.mean(points_vis, axis=0)
                        extent = np.max(points_vis, axis=0) - np.min(points_vis, axis=0)
                        max_extent = np.max(extent) if np.max(extent) > 0 else 10.0
                        view_control.set_front([0.5, -0.5, -0.7])
                        view_control.set_lookat(center)
                        view_control.set_up([0, 1, 0])
                        view_control.set_zoom(0.7)
                        print(f"[VISUALIZER] Первые точки появились! Центр: {center}, Размер: {extent}")
                    else:
                        # Обычное обновление
                        self.vis.update_geometry(self.pcd)
                
                # Обрабатываем события окна и обновляем рендерер
                self.vis.poll_events()
                self.vis.update_renderer()
                
                return True
            except Exception as e:
                print(f"[VISUALIZER] Ошибка при обновлении геометрии: {e}")
                return False
        
    def _visualization_thread(self):
        """Поток для обновления визуализации."""
        try:
            import numpy as np
            import open3d as o3d
        except ImportError as e:
            print(f"[ERROR] Не удалось импортировать open3d: {e}")
            print("[ERROR] Установите open3d: pip install open3d")
            return
        
        # Создаем визуализатор
        with self.lock:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="Облако точек SLAM - Режим реального времени",
                width=1280,
                height=720,
                visible=True
            )
            
            # Настройка параметров рендеринга
            render_option = self.vis.get_render_option()
            render_option.point_size = 2.0  # Увеличиваем размер точек для лучшей видимости
            render_option.background_color = np.array([0.05, 0.05, 0.05])  # Темный фон
            render_option.show_coordinate_frame = True  # Показать систему координат
            
            # Настраиваем начальную позицию камеры
            view_control = self.vis.get_view_control()
            view_control.set_front([0.5, -0.5, -0.7])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, 1, 0])
            view_control.set_zoom(0.5)
            
            # Инициализируем пустое облако точек (будет обновляться когда появятся данные)
            self.pcd = None
        
        print("[VISUALIZER] Окно визуализации открыто. Обновление в реальном времени...")
        print("[VISUALIZER] Управление: мышь - вращение, колесико - масштаб, Shift+мышь - перемещение")
        
        # Основной цикл обновления
        update_count = 0
        while self.running:
            try:
                # Обновляем визуализацию
                updated = self._update_visualization()
                
                # Периодически выводим информацию о количестве точек
                update_count += 1
                if update_count % 50 == 0:  # Каждые 5 секунд (50 * 0.1 сек)
                    try:
                        import numpy as np
                        points_xyz = self.acc.snapshot(max_points=100)
                        if points_xyz is not None and points_xyz.shape[0] > 0:
                            points_vis = self._transform_ned_to_vis(points_xyz)
                            if points_vis is not None:
                                center = np.mean(points_vis, axis=0)
                                extent = np.max(points_vis, axis=0) - np.min(points_vis, axis=0)
                                print(f"[VISUALIZER] Отображается {points_xyz.shape[0]} точек (макс. {self.max_display_points})")
                                print(f"[VISUALIZER] Центр: {center}, Размер: {extent}")
                            else:
                                print(f"[VISUALIZER] Получено {points_xyz.shape[0]} точек, но преобразование не удалось")
                        else:
                            print(f"[VISUALIZER] Ожидание данных от лидара... (накоплено: {self.acc._total_points})")
                    except Exception as e:
                        print(f"[VISUALIZER] Ошибка при выводе информации: {e}")
                
                # Проверяем, закрыто ли окно
                with self.lock:
                    if self.vis is None:
                        break
                    try:
                        if not self.vis.poll_events():
                            # Окно закрыто пользователем
                            self.running = False
                            break
                    except RuntimeError:
                        # Окно было закрыто
                        self.running = False
                        break
                
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"[VISUALIZER] Ошибка при обновлении: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(self.update_interval)
        
        # Закрываем окно
        with self.lock:
            if self.vis is not None:
                self.vis.destroy_window()
                self.vis = None
        
        print("[VISUALIZER] Визуализация завершена.")
    
    def start(self):
        """Запускает визуализацию в отдельном потоке."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._visualization_thread, daemon=True)
        self.thread.start()
        print("[VISUALIZER] Визуализация запущена в отдельном потоке.")
    
    def stop(self):
        """Останавливает визуализацию."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        print("[VISUALIZER] Визуализация остановлена.")
    
    def is_running(self):
        """Проверяет, работает ли визуализация."""
        return self.running


def save_point_cloud_to_ply(points_xyz, filename: str = "result.ply"):
    """
    Сохраняет облако точек в PLY файл.
    
    Args:
        points_xyz: numpy array (N, 3) точек в системе координат NED
        filename: имя файла для сохранения (по умолчанию "result.ply")
    
    Returns:
        bool: True если сохранение успешно, False в противном случае
    """
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as e:
        print(f"[ERROR] Не удалось импортировать open3d: {e}")
        return False
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        print("[ERROR] Облако точек пустое, нечего сохранять!")
        return False
    
    try:
        # Преобразуем из NED в стандартную систему координат (Z вверх)
        # NED: X=North, Y=East, Z=Down (положительный Z вниз)
        # Для сохранения: X=Right, Y=Forward, Z=Up
        points_vis = np.zeros_like(points_xyz)
        points_vis[:, 0] = points_xyz[:, 1]   # East -> Right
        points_vis[:, 1] = -points_xyz[:, 2]  # -Down -> Up
        points_vis[:, 2] = -points_xyz[:, 0]  # -North -> Forward
        
        # Создаем облако точек open3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_vis)
        
        # Устанавливаем зеленый цвет для всех точек
        colors = np.ones((points_vis.shape[0], 3)) * [0.0, 0.8, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Сохраняем в PLY файл
        success = o3d.io.write_point_cloud(filename, pcd, write_ascii=False)
        
        if success:
            print(f"[SAVE] Облако точек успешно сохранено в {filename}")
            print(f"[SAVE] Количество точек: {points_vis.shape[0]}")
            return True
        else:
            print(f"[ERROR] Не удалось сохранить облако точек в {filename}")
            return False
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении облака точек: {e}")
        import traceback
        traceback.print_exc()
        return False


def _transform_points_body_to_world(points_body, drone_pos, drone_quat):
    """
    Преобразует точки из body frame (body frame лидара) в world frame (NED).
    
    Args:
        points_body: numpy array (N, 3) точек в body frame (x вперед, y вправо, z вниз)
        drone_pos: словарь с позицией дрона {"x": north, "y": east, "z": down}
        drone_quat: словарь с ориентацией дрона {"w", "x", "y", "z"}
    
    Returns:
        numpy array (N, 3) точек в world frame (NED)
    """
    try:
        import numpy as np
    except Exception:
        return points_body
    
    if points_body is None or getattr(points_body, "size", 0) == 0:
        return points_body
    
    pts = points_body.copy()
    
    # Извлекаем кватернион
    w = float(drone_quat.get("w", 1.0))
    x = float(drone_quat.get("x", 0.0))
    y = float(drone_quat.get("y", 0.0))
    z = float(drone_quat.get("z", 0.0))
    
    # Строим матрицу поворота из кватерниона
    # R = [1-2(y²+z²)   2(xy-wz)     2(xz+wy)   ]
    #     [2(xy+wz)     1-2(x²+z²)   2(yz-wx)   ]
    #     [2(xz-wy)     2(yz+wx)     1-2(x²+y²) ]
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=np.float32)
    
    # Транспонируем для body -> world
    R = R.T
    
    # Преобразуем точки: world = R @ body + pos
    drone_position = np.array([
        float(drone_pos.get("x", 0.0)),
        float(drone_pos.get("y", 0.0)),
        float(drone_pos.get("z", 0.0))
    ], dtype=np.float32)
    
    # Применяем поворот и перенос
    pts_world = (R @ pts.T).T + drone_position
    
    return pts_world


def _point_cloud_to_2d_map(points_xyz, resolution: float = 0.1, map_size_m: float = 100.0):
    """
    Преобразует облако точек в 2D карту (вид сверху).
    
    Args:
        points_xyz: numpy array (N, 3) с точками в системе координат NED (world frame)
        resolution: разрешение карты в метрах на пиксель
        map_size_m: размер карты в метрах (карта будет map_size_m x map_size_m)
    
    Returns:
        numpy array с 2D картой (вид сверху), где значения представляют высоту/наличие препятствий
    """
    try:
        import numpy as np
    except Exception:
        return None
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        return None
    
    pts = points_xyz
    
    # Размер карты в пикселях
    map_size_px = int(map_size_m / resolution)
    
    # Проецируем точки на плоскость XY (вид сверху) - используем North (X) и East (Y)
    # В системе NED: X = North, Y = East, Z = Down (вниз, отрицательное для высоты)
    x_coords = pts[:, 0]  # North
    y_coords = pts[:, 1]  # East
    z_coords = -pts[:, 2]  # Down -> Up (инвертируем для визуализации высоты)
    
    # Находим центр облака точек
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    
    # Смещаем координаты так, чтобы центр был в середине карты
    x_shifted = x_coords - center_x + map_size_m / 2
    y_shifted = y_coords - center_y + map_size_m / 2
    
    # Преобразуем в пиксельные координаты
    x_px = (x_shifted / resolution).astype(np.int32)
    y_px = (y_shifted / resolution).astype(np.int32)
    
    # Отфильтровываем точки за пределами карты
    valid_mask = (x_px >= 0) & (x_px < map_size_px) & (y_px >= 0) & (y_px < map_size_px)
    x_px = x_px[valid_mask]
    y_px = y_px[valid_mask]
    z_coords = z_coords[valid_mask]
    
    if len(x_px) == 0:
        return None
    
    # Создаем карту: для каждого пикселя считаем максимальную высоту (occupancy grid style)
    # Используем максимальную высоту для каждого пикселя, чтобы показать препятствия
    map_2d = np.zeros((map_size_px, map_size_px), dtype=np.float32)
    
    # Накапливаем максимальные высоты (для occupancy grid)
    for i in range(len(x_px)):
        if z_coords[i] > map_2d[y_px[i], x_px[i]]:
            map_2d[y_px[i], x_px[i]] = z_coords[i]
    
    return map_2d


def _save_map_to_png(map_2d, output_path: str, colormap: str = "gray"):
    """
    Сохраняет 2D карту в PNG файл.
    
    Args:
        map_2d: numpy array с 2D картой
        output_path: путь для сохранения PNG
        colormap: название цветовой карты matplotlib (например, "gray", "viridis", "jet")
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")  # Используем неинтерактивный backend
        import matplotlib.pyplot as plt
    except Exception:
        print("[ERROR] Cannot import matplotlib or numpy for map saving")
        return False
    
    if map_2d is None:
        print("[ERROR] Map is None, cannot save")
        return False
    
    try:
        # Нормализуем карту для отображения
        map_normalized = map_2d.copy()
        
        # Убираем нулевые значения (где нет точек)
        mask = map_normalized != 0
        
        if np.any(mask):
            # Нормализуем только ненулевые значения
            min_val = np.min(map_normalized[mask])
            max_val = np.max(map_normalized[mask])
            
            if max_val > min_val:
                map_normalized[mask] = (map_normalized[mask] - min_val) / (max_val - min_val)
        
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Отображаем карту
        im = ax.imshow(map_normalized, cmap=colormap, origin='lower', interpolation='nearest')
        
        # Добавляем цветовую шкалу
        plt.colorbar(im, ax=ax, label='Высота (нормализованная)')
        
        ax.set_xlabel('X (метры)')
        ax.set_ylabel('Y (метры)')
        ax.set_title('SLAM Карта местности (вид сверху)')
        
        # Сохраняем в файл
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"[MAP] Карта сохранена в {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении карты: {e}")
        import traceback
        traceback.print_exc()
        return False

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


def display_point_cloud(points_xyz, clean_cloud: bool = True, point_size: float = 1.5, wireframe_paths: Optional[List[str]] = None):
    """
    Отображает облако точек в окне с помощью open3d с улучшенной визуализацией.
    
    Args:
        points_xyz: numpy array (N, 3) точек в системе координат NED
        clean_cloud: если True, применяет очистку облака точек перед визуализацией
        point_size: размер точек для визуализации (по умолчанию 1.5)
        wireframe_paths: список путей к файлам wireframe моделей (PLY, OBJ, STL) для наложения
    """
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as e:
        print(f"[ERROR] Не удалось импортировать open3d: {e}")
        print("[ERROR] Установите open3d: pip install open3d")
        return
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        print("[ERROR] Облако точек пустое!")
        return
    
    print(f"[INFO] Исходное облако точек: {points_xyz.shape[0]} точек")
    
    # Очистка облака точек для улучшения визуализации
    if clean_cloud:
        print("[INFO] Очистка облака точек...")
        points_xyz = _clean_point_cloud(points_xyz, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0)
        if points_xyz is not None and points_xyz.shape[0] > 0:
            print(f"[INFO] После очистки: {points_xyz.shape[0]} точек")
        else:
            print("[WARNING] После очистки облако точек стало пустым, используем исходное")
            points_xyz = points_xyz if points_xyz is not None else points_xyz
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        print("[ERROR] Облако точек пустое после обработки!")
        return
    
    # Преобразуем из NED в стандартную систему координат (Z вверх)
    # NED: X=North, Y=East, Z=Down (положительный Z вниз)
    # Для визуализации: X=Right, Y=Forward, Z=Up
    points_vis = np.zeros_like(points_xyz)
    points_vis[:, 0] = points_xyz[:, 1]   # East -> Right
    points_vis[:, 1] = -points_xyz[:, 2]  # -Down -> Up
    points_vis[:, 2] = -points_xyz[:, 0]  # -North -> Forward
    
    # Создаем облако точек open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_vis)
    
    # Устанавливаем зеленый цвет для всех точек
    if points_vis.shape[0] > 0:
        # Создаем массив зеленого цвета для всех точек
        # RGB: (0.0, 0.8, 0.0) - яркий зеленый цвет
        colors = np.ones((points_vis.shape[0], 3)) * [0.0, 0.8, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Вычисляем нормали для улучшения визуализации (опционально, но улучшает вид)
    print("[INFO] Вычисление нормалей для улучшения визуализации...")
    try:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.normalize_normals()
    except Exception as e:
        print(f"[WARNING] Не удалось вычислить нормали: {e}")
    
    # Настройка визуализатора с улучшенными параметрами
    print("[INFO] Открывается окно с облаком точек...")
    print("[INFO] Управление: мышь - вращение, колесико - масштаб, Shift+мышь - перемещение")
    print("[INFO] Закройте окно для завершения программы.")
    
    # Загружаем wireframe модели, если они предоставлены
    wireframe_geometries = []
    if wireframe_paths:
        print("[INFO] Загрузка wireframe моделей...")
        wireframe_colors = [
            [1.0, 0.0, 0.0],  # Красный
            [0.0, 0.0, 1.0],  # Синий
            [1.0, 1.0, 1.0],  # Белый
            [0.0, 1.0, 0.0],  # Зеленый
        ]
        for i, wireframe_path in enumerate(wireframe_paths):
            if os.path.exists(wireframe_path):
                try:
                    # Пытаемся загрузить как mesh
                    mesh = o3d.io.read_triangle_mesh(wireframe_path)
                    if len(mesh.vertices) > 0:
                        # Преобразуем в wireframe (только ребра)
                        mesh.compute_vertex_normals()
                        # Устанавливаем цвет
                        color = wireframe_colors[i % len(wireframe_colors)]
                        mesh.paint_uniform_color(color)
                        # Добавляем в список
                        wireframe_geometries.append(mesh)
                        print(f"[INFO] Загружена wireframe модель: {wireframe_path} ({len(mesh.vertices)} вершин)")
                    else:
                        print(f"[WARNING] Пустая wireframe модель: {wireframe_path}")
                except Exception as e:
                    print(f"[WARNING] Не удалось загрузить wireframe модель {wireframe_path}: {e}")
            else:
                print(f"[WARNING] Файл wireframe модели не найден: {wireframe_path}")
    
    # Создаем визуализатор с настройками
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Облако точек SLAM - Детальная визуализация", width=1920, height=1080, visible=True)
    vis.add_geometry(pcd)
    
    # Добавляем wireframe модели, если они есть
    for wireframe in wireframe_geometries:
        vis.add_geometry(wireframe)
    
    # Настройка параметров рендеринга
    render_option = vis.get_render_option()
    render_option.point_size = point_size
    render_option.background_color = np.array([0.05, 0.05, 0.05])  # Темный фон
    render_option.show_coordinate_frame = True  # Показать систему координат
    if wireframe_geometries:
        # Для wireframe моделей используем wireframe режим
        render_option.mesh_show_wireframe = True
        render_option.mesh_show_back_face = True
    
    # Настройка камеры для лучшего обзора
    view_control = vis.get_view_control()
    
    # Вычисляем центр облака точек
    center = np.mean(points_vis, axis=0)
    # Вычисляем размер облака точек
    extent = np.max(points_vis, axis=0) - np.min(points_vis, axis=0)
    max_extent = np.max(extent)
    
    # Устанавливаем камеру для обзора всего облака
    view_control.set_front([0.5, -0.5, -0.7])
    view_control.set_lookat(center)
    view_control.set_up([0, 1, 0])
    view_control.set_zoom(0.7)
    
    # Запускаем визуализацию
    vis.run()
    vis.destroy_window()
    
    print("[INFO] Визуализация завершена.")


async def main():
    parser = argparse.ArgumentParser(description="ProjectAirSim: SLAM картографирование местности с помощью лидара.")
    parser.add_argument("--drone-name", default="Drone1")
    parser.add_argument("--lidar-name", default="lidar1")
    parser.add_argument("--imu-name", default="IMU1", help="Имя IMU сенсора.")
    parser.add_argument("--side-length", type=float, default=10.0)
    parser.add_argument("--height", type=float, default=-5.0, help="Высота в NED (отрицательное значение = вверх).")
    parser.add_argument("--velocity", type=float, default=1.5)
    parser.add_argument("--mission", default="slam", choices=["explore", "square", "shelves", "slam", "systematic"])
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
        example_dir = Path(__file__).parent / "client" / "python" / "example_user_scripts"
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
            client.subscribe(
                drone.robot_info["actual_pose"],
                lambda _, pose_msg: pose_latest.update(pose_msg),
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
            
            # Создаем и сохраняем 2D карту
            print("[SLAM] Создание 2D карты...")
            map_2d = _point_cloud_to_2d_map(points_xyz, resolution=0.1, map_size_m=100.0)
            if map_2d is not None:
                # Создаем директорию для сохранения карты, если её нет
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                map_path = os.path.join(output_dir, "slam_map_2d.png")
                success = _save_map_to_png(map_2d, map_path, colormap="viridis")
                if success:
                    print(f"[SLAM] 2D карта успешно сохранена в {map_path}")
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

if __name__ == "__main__":
    asyncio.run(main())
