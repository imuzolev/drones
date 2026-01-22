"""Thread-safe latest sensor snapshots + math helpers."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Optional, Tuple

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
    
    # ИСПРАВЛЕНИЕ: Игнорируем точки прямо под дроном (в центральной области)
    # Когда дрон стоит на месте, точки пола под ним - это нормально
    # Проверяем только точки, которые находятся сбоку от дрона
    min_radial_distance = 0.8  # Минимальное радиальное расстояние для учета (м)
    # Точки прямо под центром дрона не считаются опасностью
    mask_not_under_center = xy_distances >= min_radial_distance
    
    if not np.any(mask_not_under_center):
        return False  # Все точки находятся прямо под дроном - это нормально
    
    # Проверяем только точки сбоку от дрона
    gear_points_side = gear_points[mask_not_under_center]
    xy_distances_side = xy_distances[mask_not_under_center]
    
    # Если есть точки на уровне ножек слишком близко сбоку - опасность!
    collision_threshold = safety_margin
    too_close = np.any(xy_distances_side < collision_threshold)
    
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
    Создает полный паттерн с промежуточными точками вдоль каждого ряда для полного покрытия.
    """
    waypoints: List[Tuple[float, float]] = []
    
    # Количество рядов (проходов по East) - определяет, сколько раз дрон пройдет по East
    rows = max(1, int(math.ceil(abs(extent_e) / max(step_e, 1.0))))
    
    # Границы области
    n_min = start_n
    n_max = start_n + float(extent_n)
    e_min = start_e
    e_max = start_e + float(extent_e)
    
    # Шаг между точками вдоль каждого ряда (по North) - делаем плотнее для полного покрытия
    # Используем меньший шаг для более плавного движения
    step_n = min(step_e, 3.0)  # Шаг не более 3 метров для плотного покрытия
    
    print(f"[lawnmower] Генерация waypoints: область {extent_n}м x {extent_e}м, шаг между рядами {step_e}м, шаг вдоль ряда {step_n}м")
    print(f"[lawnmower] Границы: N=[{n_min:.1f}, {n_max:.1f}], E=[{e_min:.1f}, {e_max:.1f}], рядов: {rows+1}")
    
    # Генерируем waypoints для каждого ряда
    for i in range(rows + 1):
        # Текущая координата East для этого ряда
        if rows > 0:
            e = e_min + (i * (extent_e / rows))
        else:
            e = e_min
        e = max(e_min, min(e, e_max))  # Ограничиваем границами
        
        # Определяем направление движения по North для этого ряда
        # Четные ряды: от n_min к n_max, нечетные: от n_max к n_min
        if i % 2 == 0:
            # Движение от n_min к n_max
            n_start = n_min
            n_end = n_max
        else:
            # Движение от n_max к n_min
            n_start = n_max
            n_end = n_min
        
        # Количество точек вдоль этого ряда - обеспечиваем плотное покрытие
        n_points = max(2, int(math.ceil(abs(extent_n) / step_n)) + 1)
        
        # Генерируем точки вдоль ряда
        for j in range(n_points):
            # Интерполируем от n_start к n_end
            if n_points > 1:
                alpha = j / (n_points - 1)
            else:
                alpha = 0.0
            n = n_start + alpha * (n_end - n_start)
            waypoints.append((n, e))
    
    print(f"[lawnmower] Создано {len(waypoints)} waypoints")
    return waypoints


