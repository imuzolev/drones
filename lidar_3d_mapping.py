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
from collections import deque
import threading

# Импорты из пакета lidar_mapping
from lidar_mapping.bootstrap import ProjectAirSimClient, Drone, World, REPO_ROOT
from lidar_mapping.config_autofix import _autofix_scene_and_robot_configs
from lidar_mapping.lio import SimpleLIO
from lidar_mapping.mapping import PointCloudAccumulator, save_point_cloud_to_ply, load_point_cloud_from_ply, _transform_points_body_to_world, _point_cloud_to_2d_map, _save_map_to_png
from lidar_mapping.path_tracker import PathTracker
from lidar_mapping.state import LidarLatest, PoseLatest, ImuLatest
from lidar_mapping.visualizer import RealtimePointCloudVisualizer


class FullPointCloudAccumulator:
    """
    Аккумулятор облака точек, который НЕ УДАЛЯЕТ старые данные.
    В отличие от PointCloudAccumulator (кольцевой буфер), этот класс
    накапливает ВСЕ точки со всех сканирований для построения полной карты.
    
    Для экономии памяти и КАЧЕСТВА карты используются:
    1. Воксельная фильтрация (downsampling) - удаление дубликатов
    2. Statistical Outlier Removal (SOR) - удаление шумовых точек
    3. Фильтрация по расстоянию - удаление слишком далеких/близких точек
    """

    def __init__(self, voxel_size: float = 0.10, downsample_threshold: int = 500_000,
                 max_range: float = 30.0, min_range: float = 0.3,
                 sor_neighbors: int = 20, sor_std_ratio: float = 1.5):
        """
        Args:
            voxel_size: размер вокселя для фильтрации дубликатов (в метрах) - увеличен для чистоты
            downsample_threshold: порог точек для применения воксельной фильтрации
            max_range: максимальная дальность точек от дрона (метры)
            min_range: минимальная дальность (фильтр близких отражений)
            sor_neighbors: количество соседей для SOR
            sor_std_ratio: порог стандартного отклонения для SOR
        """
        self._lock = threading.Lock()
        self._chunks: list = []
        self._total_points = 0
        self._voxel_size = voxel_size
        self._downsample_threshold = downsample_threshold
        self._last_downsample_count = 0
        self._max_range = max_range
        self._min_range = min_range
        self._sor_neighbors = sor_neighbors
        self._sor_std_ratio = sor_std_ratio
        self._last_drone_pos = np.array([0.0, 0.0, 0.0])
        self._filter_count = 0

    def set_drone_position(self, pos: np.ndarray):
        """Обновляет позицию дрона для фильтрации по расстоянию."""
        self._last_drone_pos = pos.copy()

    def add_points(self, points_xyz, drone_position=None) -> None:
        """Добавляет точки в аккумулятор с предварительной фильтрацией."""
        if points_xyz is None:
            return
        if getattr(points_xyz, "size", 0) == 0:
            return
        
        pts = points_xyz.copy()
        
        # 1. Фильтрация по расстоянию от дрона (если позиция известна)
        if drone_position is not None:
            drone_pos = np.array(drone_position)
            self._last_drone_pos = drone_pos
            dists = np.linalg.norm(pts - drone_pos, axis=1)
            mask = (dists >= self._min_range) & (dists <= self._max_range)
            pts = pts[mask]
            
        if pts.shape[0] == 0:
            return
        
        # 2. Удаление NaN/Inf значений
        valid_mask = np.all(np.isfinite(pts), axis=1)
        pts = pts[valid_mask]
        
        if pts.shape[0] == 0:
            return
        
        with self._lock:
            self._chunks.append(pts)
            self._total_points += int(pts.shape[0])
            
            # Применяем фильтрацию чаще для поддержания чистоты карты
            if self._total_points > self._downsample_threshold and \
               self._total_points > self._last_downsample_count * 1.3:
                self._apply_full_filter_locked()

    def _apply_voxel_filter(self, pts: np.ndarray) -> np.ndarray:
        """Воксельная фильтрация: оставляет центроид каждого вокселя."""
        if pts.shape[0] == 0:
            return pts
            
        # Вычисляем индексы вокселей
        voxel_indices = np.floor(pts / self._voxel_size).astype(np.int64)
        
        # Создаем уникальный ключ для каждого вокселя
        P1, P2 = 73856093, 19349663
        keys = voxel_indices[:, 0] * P1 + voxel_indices[:, 1] * P2 + voxel_indices[:, 2]
        
        # Находим уникальные ключи
        unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
        
        # Вычисляем центроид каждого вокселя (среднее значение точек в вокселе)
        # Это дает более точную позицию, чем просто первая точка
        n_voxels = len(unique_keys)
        centroids = np.zeros((n_voxels, 3), dtype=np.float32)
        counts = np.zeros(n_voxels, dtype=np.int32)
        
        for i in range(3):
            np.add.at(centroids[:, i], inverse_indices, pts[:, i])
        np.add.at(counts, inverse_indices, 1)
        
        # Делим на количество точек для получения среднего
        centroids /= counts[:, np.newaxis]
        
        return centroids.astype(np.float32)

    def _apply_statistical_outlier_removal(self, pts: np.ndarray) -> np.ndarray:
        """Удаляет выбросы на основе расстояния до соседей (Statistical Outlier Removal)."""
        if pts.shape[0] < self._sor_neighbors + 1:
            return pts
            
        try:
            from scipy.spatial import cKDTree
            
            # Строим KD-дерево
            tree = cKDTree(pts)
            
            # Находим k ближайших соседей для каждой точки
            distances, _ = tree.query(pts, k=self._sor_neighbors + 1)  # +1 потому что точка сама себе сосед
            
            # Средняя дистанция до соседей (исключаем первого - это сама точка)
            mean_dists = np.mean(distances[:, 1:], axis=1)
            
            # Глобальное среднее и стандартное отклонение
            global_mean = np.mean(mean_dists)
            global_std = np.std(mean_dists)
            
            # Порог: mean + std_ratio * std
            threshold = global_mean + self._sor_std_ratio * global_std
            
            # Оставляем только точки с дистанцией ниже порога
            mask = mean_dists < threshold
            return pts[mask]
            
        except ImportError:
            # Если scipy недоступен, пропускаем SOR
            return pts
        except Exception as e:
            print(f"[ACCUMULATOR] SOR ошибка: {e}")
            return pts

    def _apply_radius_outlier_removal(self, pts: np.ndarray, radius: float = 0.3, min_neighbors: int = 5) -> np.ndarray:
        """Удаляет изолированные точки без достаточного количества соседей в радиусе."""
        if pts.shape[0] < min_neighbors:
            return pts
            
        try:
            from scipy.spatial import cKDTree
            
            tree = cKDTree(pts)
            # Считаем количество соседей в радиусе
            neighbors_count = tree.query_ball_point(pts, r=radius, return_length=True)
            
            # Оставляем точки с достаточным количеством соседей
            mask = neighbors_count >= min_neighbors
            return pts[mask]
            
        except ImportError:
            return pts
        except Exception as e:
            print(f"[ACCUMULATOR] Radius filter ошибка: {e}")
            return pts

    def _apply_full_filter_locked(self):
        """Применяет полный набор фильтров (вызывать с lock!)."""
        try:
            if len(self._chunks) == 0:
                return
                
            # Объединяем все чанки
            all_pts = np.concatenate(self._chunks, axis=0)
            old_count = all_pts.shape[0]
            
            self._filter_count += 1
            
            # 1. Воксельная фильтрация (всегда)
            filtered_pts = self._apply_voxel_filter(all_pts)
            after_voxel = filtered_pts.shape[0]
            
            # 2. Statistical Outlier Removal (каждые 3 итерации для производительности)
            if self._filter_count % 3 == 0 and filtered_pts.shape[0] > 1000:
                filtered_pts = self._apply_statistical_outlier_removal(filtered_pts)
                after_sor = filtered_pts.shape[0]
            else:
                after_sor = filtered_pts.shape[0]
            
            # 3. Radius outlier removal (каждые 5 итераций)
            if self._filter_count % 5 == 0 and filtered_pts.shape[0] > 1000:
                filtered_pts = self._apply_radius_outlier_removal(filtered_pts, radius=0.25, min_neighbors=4)
            
            # Заменяем все чанки одним отфильтрованным
            self._chunks = [filtered_pts]
            self._total_points = filtered_pts.shape[0]
            self._last_downsample_count = self._total_points
            
            reduction_pct = 100 * (1 - self._total_points / old_count) if old_count > 0 else 0
            print(f"[ACCUMULATOR] Фильтрация #{self._filter_count}: {old_count} -> voxel:{after_voxel} -> sor:{after_sor} -> final:{self._total_points} (сокращение {reduction_pct:.1f}%)")
            
        except Exception as e:
            print(f"[ACCUMULATOR] Ошибка фильтрации: {e}")
            import traceback
            traceback.print_exc()

    def force_cleanup(self):
        """Принудительная очистка с полным набором фильтров."""
        with self._lock:
            if self._total_points > 0:
                print("[ACCUMULATOR] Принудительная очистка...")
                # Временно устанавливаем более агрессивные параметры
                old_sor_ratio = self._sor_std_ratio
                self._sor_std_ratio = 1.0  # Более агрессивно
                self._filter_count = 14  # Чтобы все фильтры сработали
                self._apply_full_filter_locked()
                self._sor_std_ratio = old_sor_ratio

    def snapshot(self, max_points: int = None, apply_final_filter: bool = False):
        """Возвращает копию всех накопленных точек."""
        with self._lock:
            if self._total_points <= 0 or len(self._chunks) == 0:
                return np.empty((0, 3), dtype=np.float32)

            pts = np.concatenate(self._chunks, axis=0)
            
            # Опционально применяем финальную фильтрацию при сохранении
            if apply_final_filter and pts.shape[0] > 1000:
                # Финальный SOR с агрессивными параметрами
                pts = self._apply_statistical_outlier_removal(pts)
                pts = self._apply_radius_outlier_removal(pts, radius=0.2, min_neighbors=3)
            
            if max_points is not None and pts.shape[0] > max_points:
                # Случайная подвыборка для отображения
                idx = np.random.choice(pts.shape[0], size=int(max_points), replace=False)
                pts = pts[idx]
            
            return pts
    
    def get_total_points(self) -> int:
        """Возвращает общее количество накопленных точек."""
        with self._lock:
            return self._total_points
    
    def get_stats(self) -> dict:
        """Возвращает статистику аккумулятора."""
        with self._lock:
            return {
                "total_points": self._total_points,
                "chunks": len(self._chunks),
                "filter_count": self._filter_count,
                "voxel_size": self._voxel_size
            }

class PoseHistory:
    """
    Хранит историю поз для точной синхронизации с данными лидара по времени.
    Позволяет получать интерполированную позу на заданный момент времени.
    """
    def __init__(self, maxlen=200):
        self._lock = threading.Lock()
        self._history = deque(maxlen=maxlen) # (timestamp, pose_msg)
        self._latest_pose = None
        self._latest_ts = 0.0

    def update(self, pose_msg):
        with self._lock:
            ts = 0.0
            if isinstance(pose_msg, dict):
                # Пробуем получить timestamp из сообщения
                ts_raw = pose_msg.get("time_stamp") or pose_msg.get("timestamp")
                if ts_raw:
                    # AirSim часто шлет наносекунды
                    if isinstance(ts_raw, (int, float)) and ts_raw > 1e12:
                        ts = float(ts_raw) / 1e9
                    else:
                        ts = float(ts_raw)
                else:
                    ts = time.time()
            else:
                ts = time.time()
            
            # Добавляем в историю
            self._history.append((ts, pose_msg))
            self._latest_pose = pose_msg
            self._latest_ts = ts
            
            # Сортировка на случай поступления сообщений не по порядку (редко, но бывает)
            # Если рассинхрон > 0.1с, лучше отсортировать, но для deque это дорого.
            # Обычно данные идут последовательно.
    
    def snapshot(self):
        """Совместимость с PoseLatest.snapshot()"""
        with self._lock:
            return self._latest_pose, self._latest_ts

    def get_interpolated_pose(self, query_ts):
        with self._lock:
            if not self._history:
                return None
            
            # Проверка границ
            if query_ts >= self._history[-1][0]:
                return self._history[-1][1]
            if query_ts <= self._history[0][0]:
                return self._history[0][1]
            
            # Линейный поиск (так как deque небольшой и данные упорядочены)
            # Ищем пару (prev, next), между которыми query_ts
            # Идем с конца, так как обычно запрос близок к текущему времени
            for i in range(len(self._history) - 1, 0, -1):
                t2, p2 = self._history[i]
                t1, p1 = self._history[i-1]
                
                if t1 <= query_ts <= t2:
                    return self._interpolate(p1, p2, t1, t2, query_ts)
            
            return self._latest_pose

    def _interpolate(self, p1, p2, t1, t2, t_query):
        if abs(t2 - t1) < 1e-6:
            return p1
            
        alpha = (t_query - t1) / (t2 - t1)
        
        # Интерполяция позиции (LERP)
        pos1 = p1.get("position", {})
        pos2 = p2.get("position", {})
        
        res_pos = {
            "x": float(pos1.get("x",0)) + alpha*(float(pos2.get("x",0))-float(pos1.get("x",0))),
            "y": float(pos1.get("y",0)) + alpha*(float(pos2.get("y",0))-float(pos1.get("y",0))),
            "z": float(pos1.get("z",0)) + alpha*(float(pos2.get("z",0))-float(pos1.get("z",0))),
        }
        
        # Интерполяция ориентации (SLERP)
        q1 = p1.get("orientation", {})
        q2 = p2.get("orientation", {})
        
        res_ori = self._slerp(q1, q2, alpha)
        
        return {
            "position": res_pos,
            "orientation": res_ori,
            "time_stamp": t_query
        }
        
    def _slerp(self, q1_dict, q2_dict, t):
        # w, x, y, z
        q1 = np.array([float(q1_dict.get("w",1)), float(q1_dict.get("x",0)), float(q1_dict.get("y",0)), float(q1_dict.get("z",0))])
        q2 = np.array([float(q2_dict.get("w",1)), float(q2_dict.get("x",0)), float(q2_dict.get("y",0)), float(q2_dict.get("z",0))])
        
        # Normalize
        norm1 = np.linalg.norm(q1)
        norm2 = np.linalg.norm(q2)
        if norm1 > 0: q1 /= norm1
        if norm2 > 0: q2 /= norm2
        
        dot = np.dot(q1, q2)
        
        # Если скалярное произведение отрицательное, инвертируем один кватернион
        if dot < 0.0:
            q2 = -q2
            dot = -dot
            
        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            # Линейная интерполяция для малых углов
            result = q1 + t * (q2 - q1)
            result /= np.linalg.norm(result)
            return {"w": result[0], "x": result[1], "y": result[2], "z": result[3]}
            
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        
        s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0
        
        result = s1 * q1 + s2 * q2
        return {"w": result[0], "x": result[1], "y": result[2], "z": result[3]}

def _get_repulsive_velocity(
    lidar_points: np.ndarray, 
    influence_dist: float = 3.0, 
    max_repulse: float = 2.0,
    side_influence_dist: float = None
) -> tuple[float, float]:
    """Рассчитывает вектор отталкивания от препятствий в плоскости XY (body frame)"""
    if side_influence_dist is None:
        side_influence_dist = influence_dist

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
    
    # Определяем сектора: |y| > |x| -> Side (Right/Left), иначе Front/Back
    abs_x = np.abs(pts[:, 0])
    abs_y = np.abs(pts[:, 1])
    mask_side = abs_y > abs_x
    mask_front = ~mask_side
    
    fx_total = 0.0
    fy_total = 0.0
    
    # 1. Front/Back Forces (use influence_dist)
    mask_f_active = mask_front & (dists < influence_dist) & (dists > 0.1)
    if np.any(mask_f_active):
        p_f = pts[mask_f_active]
        d_f = dists[mask_f_active]
        w_f = (1.0 / d_f - 1.0 / influence_dist)
        fx_total += np.sum(-p_f[:, 0] / d_f * w_f)
        fy_total += np.sum(-p_f[:, 1] / d_f * w_f)
        
    # 2. Side Forces (use side_influence_dist)
    mask_s_active = mask_side & (dists < side_influence_dist) & (dists > 0.1)
    if np.any(mask_s_active):
        p_s = pts[mask_s_active]
        d_s = dists[mask_s_active]
        w_s = (1.0 / d_s - 1.0 / side_influence_dist)
        fx_total += np.sum(-p_s[:, 0] / d_s * w_s)
        fy_total += np.sum(-p_s[:, 1] / d_s * w_s)
    
    # Ограничиваем максимальную силу
    force_mag = math.hypot(fx_total, fy_total)
    if force_mag > max_repulse:
        scale = max_repulse / force_mag
        fx_total *= scale
        fy_total *= scale
        
    return fx_total, fy_total

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
    influence_dist: float = 3.0,
    side_influence_dist: float = 1.0
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
            rep_fwd, rep_right = _get_repulsive_velocity(
                pts, 
                influence_dist=influence_dist, 
                max_repulse=2.0, 
                side_influence_dist=side_influence_dist
            )
            
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
    duration_sec: float = 5.0,
    rotation_speed: float = 0.5  # радиан/сек (полный оборот ~12.5 сек)
):
    """
    Зависает на текущей позиции и выполняет полное вращение на 360 градусов
    для сканирования всего окружения без мертвых зон.
    """
    print(f"[SCAN] Сканирование с вращением 360° в точке...")
    
    # Получаем текущую позицию для удержания
    start_n, start_e = 0.0, 0.0
    start_yaw = 0.0
    pose_msg, _ = pose_latest.snapshot()
    if pose_msg and isinstance(pose_msg, dict):
        pos = pose_msg.get("position", {})
        start_n = float(pos.get("x", 0.0))
        start_e = float(pos.get("y", 0.0))
        
        # Получаем начальный yaw
        ori = pose_msg.get("orientation", {})
        q0 = float(ori.get("w", 1.0))
        q1 = float(ori.get("x", 0.0))
        q2 = float(ori.get("y", 0.0))
        q3 = float(ori.get("z", 0.0))
        start_yaw = math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
    
    # Вращаемся на 360 градусов (2*pi радиан)
    total_rotation = 2 * math.pi
    rotation_time = total_rotation / rotation_speed  # время для полного оборота
    
    dt = 0.1
    start_time = time.time()
    last_print_time = 0
    
    while time.time() - start_time < rotation_time:
        elapsed = time.time() - start_time
        
        # Вывод прогресса каждые 2 секунды
        if int(elapsed) != last_print_time and int(elapsed) % 2 == 0:
            last_print_time = int(elapsed)
            rotation_degrees = (elapsed / rotation_time) * 360
            total_pts = acc.get_total_points() if hasattr(acc, 'get_total_points') else getattr(acc, '_total_points', 0)
            print(f"[SCAN] Вращение: {rotation_degrees:.0f}°, накоплено точек: {total_pts}")
        
        # Команда: удерживаем позицию, вращаемся с постоянной скоростью
        await drone.move_by_velocity_body_frame_async(
            v_forward=0.0,
            v_right=0.0,
            v_down=0.0,  # Удержание высоты обеспечивается контроллером
            duration=dt,
            yaw_is_rate=True,
            yaw=float(rotation_speed)  # Вращение по часовой стрелке
        )
        
        await asyncio.sleep(dt)
    
    # Небольшая пауза после вращения для стабилизации
    await drone.move_by_velocity_body_frame_async(0, 0, 0, 0.5, yaw_is_rate=True, yaw=0.0)
    await asyncio.sleep(0.3)
    
    total_pts = acc.get_total_points() if hasattr(acc, 'get_total_points') else getattr(acc, '_total_points', 0)
    print(f"[SCAN] Сканирование завершено. Итого точек: {total_pts}")

async def explore_room_perimeter(
    drone: Drone,
    pose_latest: PoseLatest,
    lidar_latest: LidarLatest,
    acc: PointCloudAccumulator,
    height: float,
    velocity: float,
    scan_duration: float,
    target_wall_dist: float = 1.0
):
    """
    Исследует комнату, следуя вдоль стен (Right-Hand Rule) для построения полной карты.
    Алгоритм:
    1. Найти ближайшую стену (сканирование 360).
    2. Долететь до нее на расстояние target_wall_dist.
    3. Следовать вдоль стены, держа её справа.
    4. Завершить при возврате в исходную точку.
    """
    print(f"[MISSION] Начинаю исследование периметра (высота: {-height}м, расстояние до стены: {target_wall_dist}м)...")
    
    # --- Helper: Get Lidar Stats ---
    def get_lidar_stats():
        pts, _ = lidar_latest.snapshot()
        if pts is None or pts.size == 0:
            return 999.0, 999.0, 999.0
            
        # Filter by height (level with drone)
        mask_z = np.abs(pts[:, 2]) < 0.5
        pts_plane = pts[mask_z]
        if pts_plane.size == 0:
            return 999.0, 999.0, 999.0

        dists = np.linalg.norm(pts_plane[:, :2], axis=1)
        angles = np.arctan2(pts_plane[:, 1], pts_plane[:, 0])
        
        # Front (-30..30 deg)
        mask_front = (np.abs(angles) < math.radians(30))
        d_front = np.min(dists[mask_front]) if np.any(mask_front) else 99.0
        
        # Right (60..120 deg) - Y+ is Right
        mask_right = (angles > math.radians(60)) & (angles < math.radians(120))
        d_right = np.min(dists[mask_right]) if np.any(mask_right) else 99.0
        
        # Left (-120..-60 deg)
        mask_left = (angles > math.radians(-120)) & (angles < math.radians(-60))
        d_left = np.min(dists[mask_left]) if np.any(mask_left) else 99.0
        
        return d_front, d_right, d_left

    # --- 1. Get Start Position ---
    start_pos = None
    for _ in range(20):
        pose_msg, _ = pose_latest.snapshot()
        if pose_msg and isinstance(pose_msg, dict):
             p = pose_msg.get("position", {})
             if p:
                 start_pos = np.array([float(p.get("x",0)), float(p.get("y",0))])
                 break
        await asyncio.sleep(0.1)
        
    if start_pos is None:
        print("[ERROR] Не удалось получить начальную позицию!")
        return

    print(f"[MISSION] Старт: {start_pos}")
    
    # --- State Machine ---
    state = "FIND_WALL"
    wall_following_start_pos = None
    wall_following_start_time = 0.0
    total_dist_traveled = 0.0
    last_pos = start_pos.copy()
    
    dt = 0.1
    max_duration = 600.0 # 10 minutes limit
    mission_start = time.time()
    
    kp_dist = 0.8
    kp_yaw = 0.5
    
    while time.time() - mission_start < max_duration:
        d_front, d_right, d_left = get_lidar_stats()
        
        # Get Current Pos
        curr_pos = None
        pose_msg, _ = pose_latest.snapshot()
        cur_z = 0.0
        if pose_msg:
             p = pose_msg.get("position", {})
             curr_pos = np.array([float(p.get("x",0)), float(p.get("y",0))])
             cur_z = float(p.get("z",0))
             
             # Track distance
             step_dist = np.linalg.norm(curr_pos - last_pos)
             total_dist_traveled += step_dist
             last_pos = curr_pos.copy()
             
        v_fwd_cmd = 0.0
        v_right_cmd = 0.0
        yaw_rate_cmd = 0.0
        
        if state == "FIND_WALL":
            # 1. Сначала ищем ближайшую стену сканированием
            print("\n[PHASE 1] Поиск ближайшей стены...")
            wall_dist, wall_angle = await find_nearest_wall_direction(drone, lidar_latest)
            
            # 2. Поворот к стене
            print(f"[PHASE 2] Поворот к стене (угол: {math.degrees(wall_angle):.1f}°)...")
            await turn_to_yaw(drone, wall_angle, pose_latest)
            
            # 3. Летим к стене
            print(f"[PHASE 3] Движение к стене (целевое расстояние: {target_wall_dist}м)...")
            await move_towards_wall(drone, pose_latest, lidar_latest, height, target_wall_dist, velocity)
            
            # 4. Переходим к огибанию
            print("[PHASE 4] Переход к огибанию стены.")
            state = "FOLLOW_WALL"
            wall_following_start_pos = curr_pos
            wall_following_start_time = time.time()
            total_dist_traveled = 0.0
            await drone.move_by_velocity_body_frame_async(0,0,0, 1.0) # Brake
            continue # Переходим к следующей итерации цикла уже в новом стейте
        
        elif state == "FOLLOW_WALL":
            # Loop Closure Check
            if wall_following_start_pos is not None and curr_pos is not None:
                dist_from_start = np.linalg.norm(curr_pos - wall_following_start_pos)
                
                # Check if we returned to start (requires min distance traveled)
                if total_dist_traveled > 20.0 and dist_from_start < 2.0:
                    print(f"[MISSION] Петля замкнута (dist={dist_from_start:.1f}m, total={total_dist_traveled:.1f}m). Завершение.")
                    break
            
            # --- Wall Following Logic (Right Hand) ---
            
            # 1. Blocked Front -> Turn Left
            if d_front < target_wall_dist + 0.5:
                # print(f"[NAV] Блок спереди ({d_front:.1f}) -> Поворот влево")
                v_fwd_cmd = 0.0
                yaw_rate_cmd = -0.5 # Turn Left (CCW)
                
            # 2. Lost Right Wall (Corner?) -> Turn Right to find it
            elif d_right > target_wall_dist * 2.5:
                # print(f"[NAV] Потеря стены справа ({d_right:.1f}) -> Поворот вправо")
                v_fwd_cmd = velocity * 0.4
                yaw_rate_cmd = 0.4 # Turn Right (CW)
                v_right_cmd = 0.3  # Strafe Right slightly
                
            # 3. Too Close to Right Wall -> Strafe Left
            elif d_right < target_wall_dist * 0.5:
                # print(f"[NAV] Слишком близко справа ({d_right:.1f}) -> Отход влево")
                v_fwd_cmd = velocity * 0.5
                v_right_cmd = -0.8 # Strafe Left
                yaw_rate_cmd = -0.1 # Slight turn left
                
            # 4. Normal Following
            else:
                # Keep distance
                err = d_right - target_wall_dist
                # if err > 0 (too far) -> strafe right (+)
                # if err < 0 (too close) -> strafe left (-)
                v_right_cmd = np.clip(err * kp_dist, -1.0, 1.0)
                v_fwd_cmd = velocity
                
                # Align Parallel?
                # If d_front is decreasing, we are turning into wall -> Turn Left
                # Hard to know without derivative.
                # Just simple P-control on distance usually works if speed is low.
        
        # Height Control
        dz = height - cur_z
        vz_cmd = np.clip(dz, -1.0, 1.0)
        
        # Send Command
        await drone.move_by_velocity_body_frame_async(
            v_forward=float(v_fwd_cmd),
            v_right=float(v_right_cmd),
            v_down=float(vz_cmd),
            duration=dt,
            yaw_is_rate=True,
            yaw=float(yaw_rate_cmd)
        )
        
        await asyncio.sleep(dt)

    print("[MISSION] Миссия завершена. Возврат на точку старта...")
    
    # Return to Start Position
    if start_pos is not None:
         await fly_to_point_with_avoidance(
            drone=drone,
            target_n=start_pos[0],
            target_e=start_pos[1],
            height=height,
            pose_latest=pose_latest,
            lidar_latest=lidar_latest,
            stop_dist=0.5
        )
         
    # Smooth Landing
    print("[MISSION] Снижение...")
    # Lower to 1m height first
    current_n, current_e = 0.0, 0.0
    pose_msg, _ = pose_latest.snapshot()
    if pose_msg:
        p = pose_msg.get("position", {})
        current_n = float(p.get("x", 0))
        current_e = float(p.get("y", 0))
        
    await drone.move_to_position_async(current_n, current_e, -1.0, 1.0)
    await asyncio.sleep(1.0)

async def find_nearest_wall_direction(
    drone: Drone,
    lidar_latest: LidarLatest,
    rotation_speed: float = 0.5
) -> tuple:
    """
    Вращается на 360° для определения ближайшей стены.
    Возвращает (min_distance, angle_to_wall) - расстояние и угол к ближайшей стене.
    """
    print("[SCAN] Поиск ближайшей стены (вращение 360°)...")
    
    min_distance = 999.0
    min_angle = 0.0  # Угол к ближайшей стене (в world frame yaw)
    
    total_rotation = 2 * math.pi
    rotation_time = total_rotation / rotation_speed
    
    dt = 0.1
    start_time = time.time()
    
    # Получаем начальный yaw
    start_yaw = 0.0
    
    while time.time() - start_time < rotation_time:
        elapsed = time.time() - start_time
        current_rotation = elapsed * rotation_speed  # Сколько повернулись
        
        # Получаем данные лидара
        pts, _ = lidar_latest.snapshot()
        
        if pts is not None and pts.size > 0:
            # Фильтруем точки по высоте (на уровне дрона ±0.5м)
            mask_z = np.abs(pts[:, 2]) < 0.5
            pts_plane = pts[mask_z]
            
            if pts_plane.size > 0:
                # Расстояния и углы в body frame
                dists = np.linalg.norm(pts_plane[:, :2], axis=1)
                angles = np.arctan2(pts_plane[:, 1], pts_plane[:, 0])
                
                # Смотрим только вперед (±30°)
                mask_front = np.abs(angles) < math.radians(30)
                
                if np.any(mask_front):
                    front_dists = dists[mask_front]
                    min_front_dist = np.min(front_dists)
                    
                    if min_front_dist < min_distance:
                        min_distance = min_front_dist
                        # Угол к стене = начальный yaw + сколько повернулись
                        min_angle = start_yaw + current_rotation
                        
                        # Нормализация угла
                        while min_angle > math.pi: min_angle -= 2*math.pi
                        while min_angle < -math.pi: min_angle += 2*math.pi
        
        # Вращаемся
        await drone.move_by_velocity_body_frame_async(
            v_forward=0.0,
            v_right=0.0,
            v_down=0.0,
            duration=dt,
            yaw_is_rate=True,
            yaw=float(rotation_speed)
        )
        
        await asyncio.sleep(dt)
    
    # Остановка после вращения
    await drone.move_by_velocity_body_frame_async(0, 0, 0, 0.3, yaw_is_rate=True, yaw=0.0)
    await asyncio.sleep(0.2)
    
    print(f"[SCAN] Ближайшая стена: {min_distance:.2f}м под углом {math.degrees(min_angle):.1f}°")
    
    return min_distance, min_angle


async def turn_to_yaw(drone: Drone, target_yaw: float, pose_latest, tolerance: float = 0.1):
    """Поворачивается к заданному углу yaw (в радианах)."""
    dt = 0.1
    timeout = 10.0
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        pose_msg, _ = pose_latest.snapshot()
        if not pose_msg:
            await asyncio.sleep(dt)
            continue
            
        ori = pose_msg.get("orientation", {})
        q0 = float(ori.get("w", 1.0))
        q1 = float(ori.get("x", 0.0))
        q2 = float(ori.get("y", 0.0))
        q3 = float(ori.get("z", 0.0))
        current_yaw = math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        
        yaw_err = target_yaw - current_yaw
        while yaw_err > math.pi: yaw_err -= 2*math.pi
        while yaw_err < -math.pi: yaw_err += 2*math.pi
        
        if abs(yaw_err) < tolerance:
            break
            
        yaw_rate = np.clip(yaw_err * 1.5, -1.0, 1.0)
        
        await drone.move_by_velocity_body_frame_async(
            v_forward=0.0,
            v_right=0.0,
            v_down=0.0,
            duration=dt,
            yaw_is_rate=True,
            yaw=float(yaw_rate)
        )
        await asyncio.sleep(dt)
    
    # Остановка
    await drone.move_by_velocity_body_frame_async(0, 0, 0, 0.2, yaw_is_rate=True, yaw=0.0)


async def get_current_yaw(pose_latest) -> float:
    """Получает текущий yaw дрона."""
    pose_msg, _ = pose_latest.snapshot()
    if not pose_msg:
        return 0.0
    ori = pose_msg.get("orientation", {})
    q0 = float(ori.get("w", 1.0))
    q1 = float(ori.get("x", 0.0))
    q2 = float(ori.get("y", 0.0))
    q3 = float(ori.get("z", 0.0))
    return math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))


async def move_towards_wall(
    drone: Drone,
    pose_latest,
    lidar_latest: LidarLatest,
    height: float,
    target_distance: float = 0.5,
    velocity: float = 1.0
) -> bool:
    """
    Движется вперед к стене, пока расстояние не станет target_distance.
    Возвращает True если достигли стены.
    """
    print(f"[NAV] Движение к стене (целевое расстояние: {target_distance}м)...")
    
    dt = 0.1
    timeout = 60.0
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Получаем расстояние до стены спереди
        pts, _ = lidar_latest.snapshot()
        front_dist = 999.0
        
        if pts is not None and pts.size > 0:
            mask_z = np.abs(pts[:, 2]) < 0.5
            pts_plane = pts[mask_z]
            
            if pts_plane.size > 0:
                dists = np.linalg.norm(pts_plane[:, :2], axis=1)
                angles = np.arctan2(pts_plane[:, 1], pts_plane[:, 0])
                
                # Спереди (±30°)
                mask_front = np.abs(angles) < math.radians(30)
                if np.any(mask_front):
                    front_dist = np.min(dists[mask_front])
        
        # Проверяем достижение цели (с небольшим запасом)
        if front_dist <= target_distance + 0.1:
            print(f"[NAV] Достигли стены ({front_dist:.2f}м)")
            await drone.move_by_velocity_body_frame_async(0, 0, 0, 0.5, yaw_is_rate=True, yaw=0.0)
            await asyncio.sleep(0.3)
            return True
        
        # Скорость движения (замедляемся при приближении - начинаем раньше)
        speed = velocity
        if front_dist < 3.0:  # Начинаем замедление раньше
            speed = velocity * (front_dist - target_distance) / (3.0 - target_distance)
            speed = max(0.2, min(speed, velocity))  # Минимум 0.2 м/с
        
        # Управление высотой
        pose_msg, _ = pose_latest.snapshot()
        cur_z = 0.0
        if pose_msg:
            cur_z = float(pose_msg.get("position", {}).get("z", 0))
        dz = height - cur_z
        vz_cmd = np.clip(dz, -1.0, 1.0)
        
        await drone.move_by_velocity_body_frame_async(
            v_forward=float(speed),
            v_right=0.0,
            v_down=float(vz_cmd),
            duration=dt,
            yaw_is_rate=True,
            yaw=0.0
        )
        
        await asyncio.sleep(dt)
    
    print("[NAV] Таймаут движения к стене")
    return False


async def follow_wall_until_corner(
    drone: Drone,
    pose_latest,
    lidar_latest: LidarLatest,
    height: float,
    wall_distance: float = 0.5,
    velocity: float = 1.0
) -> bool:
    """
    Следует вдоль стены (стена справа) пока не достигнет угла (стена спереди на 0.5м).
    Возвращает True если достигли угла.
    """
    print(f"[NAV] Следование вдоль стены (стена справа на {wall_distance}м)...")
    
    dt = 0.1
    timeout = 120.0
    start_time = time.time()
    
    kp_lateral = 2.0  # Коэффициент для коррекции бокового положения
    
    while time.time() - start_time < timeout:
        # Получаем расстояния
        pts, _ = lidar_latest.snapshot()
        front_dist = 999.0
        right_dist = 999.0
        
        if pts is not None and pts.size > 0:
            mask_z = np.abs(pts[:, 2]) < 0.5
            pts_plane = pts[mask_z]
            
            if pts_plane.size > 0:
                dists = np.linalg.norm(pts_plane[:, :2], axis=1)
                angles = np.arctan2(pts_plane[:, 1], pts_plane[:, 0])
                
                # Спереди (±30°)
                mask_front = np.abs(angles) < math.radians(30)
                if np.any(mask_front):
                    front_dist = np.min(dists[mask_front])
                
                # Справа (60°..120°) - Y+ в body frame = право
                mask_right = (angles > math.radians(60)) & (angles < math.radians(120))
                if np.any(mask_right):
                    right_dist = np.min(dists[mask_right])
        
        # Проверяем достижение угла (стена спереди) - с запасом
        if front_dist <= wall_distance + 0.2:
            print(f"[NAV] Достигли угла (стена спереди: {front_dist:.2f}м)")
            await drone.move_by_velocity_body_frame_async(0, 0, 0, 0.5, yaw_is_rate=True, yaw=0.0)
            await asyncio.sleep(0.3)
            return True
        
        # Управление боковой позицией (держим расстояние до правой стены)
        v_right = 0.0
        if right_dist < 999.0:
            err = wall_distance - right_dist  # Если слишком близко (err > 0), двигаемся влево
            v_right = np.clip(-err * kp_lateral, -0.3, 0.3)  # Уменьшена боковая скорость
        
        # Скорость вперед (замедление при приближении к углу - начинаем раньше)
        v_fwd = velocity
        if front_dist < 4.0:  # Начинаем замедление раньше
            v_fwd = velocity * (front_dist - wall_distance) / (4.0 - wall_distance)
            v_fwd = max(0.2, min(v_fwd, velocity))  # Минимум 0.2 м/с
        
        # Управление высотой
        pose_msg, _ = pose_latest.snapshot()
        cur_z = 0.0
        if pose_msg:
            cur_z = float(pose_msg.get("position", {}).get("z", 0))
        dz = height - cur_z
        vz_cmd = np.clip(dz, -1.0, 1.0)
        
        # Вывод статуса
        if int(time.time() * 2) % 4 == 0:
            print(f"[NAV] Фронт: {front_dist:.1f}м, Право: {right_dist:.1f}м, vR: {v_right:.2f}")
        
        await drone.move_by_velocity_body_frame_async(
            v_forward=float(v_fwd),
            v_right=float(v_right),
            v_down=float(vz_cmd),
            duration=dt,
            yaw_is_rate=True,
            yaw=0.0
        )
        
        await asyncio.sleep(dt)
    
    print("[NAV] Таймаут следования вдоль стены")
    return False


async def fly_wall_following_pattern(
    drone: Drone,
    pose_latest,
    lidar_latest: LidarLatest,
    acc,
    height: float,
    wall_distance: float = 0.5,
    velocity: float = 1.0,
    scan_duration: float = 5.0
):
    """
    Алгоритм облета по стенам:
    1. Вращение на 360° для поиска ближайшей стены
    2. Движение к стене
    3. При достижении 0.5м - поворот направо на 90°
    4. Следование вдоль стены до следующего угла
    5. Повтор до нахождения 4 углов
    6. Возврат в исходную точку и посадка
    В каждом углу - сканирование на 360°
    """
    print(f"[MISSION] Запуск облета по стенам (высота: {-height}м, расстояние до стены: {wall_distance}м)")
    
    # Сохраняем начальную позицию
    start_pos = None
    for _ in range(20):
        pose_msg, _ = pose_latest.snapshot()
        if pose_msg:
            p = pose_msg.get("position", {})
            start_pos = np.array([float(p.get("x", 0)), float(p.get("y", 0))])
            break
        await asyncio.sleep(0.1)
    
    if start_pos is None:
        print("[ERROR] Не удалось получить начальную позицию!")
        return
    
    print(f"[MISSION] Начальная позиция: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")
    
    # 1. Первоначальное сканирование и поиск ближайшей стены
    print("\n[PHASE 1] Поиск ближайшей стены...")
    wall_dist, wall_angle = await find_nearest_wall_direction(drone, lidar_latest)
    
    # 2. Поворот к ближайшей стене
    print(f"\n[PHASE 2] Поворот к стене (угол: {math.degrees(wall_angle):.1f}°)...")
    await turn_to_yaw(drone, wall_angle, pose_latest)
    
    # 3. Движение к стене
    print("\n[PHASE 3] Движение к стене...")
    await move_towards_wall(drone, pose_latest, lidar_latest, height, wall_distance, velocity)
    
    # 4. Сканирование в первом углу
    print("\n[CORNER 0] Сканирование...")
    await hover_and_scan_at_corner(drone, pose_latest, acc, height, scan_duration)
    
    corners_found = 0
    max_corners = 4
    
    while corners_found < max_corners:
        corners_found += 1
        print(f"\n[CORNER {corners_found}] Поворот направо на 90°...")
        
        # Поворот направо на 90°
        current_yaw = await get_current_yaw(pose_latest)
        new_yaw = current_yaw + math.pi / 2  # +90° по часовой стрелке
        while new_yaw > math.pi: new_yaw -= 2*math.pi
        await turn_to_yaw(drone, new_yaw, pose_latest)
        
        # Следование вдоль стены до следующего угла
        print(f"\n[WALL {corners_found}] Следование вдоль стены...")
        corner_reached = await follow_wall_until_corner(
            drone, pose_latest, lidar_latest, height, wall_distance, velocity
        )
        
        if corner_reached:
            # Сканирование в углу
            print(f"\n[CORNER {corners_found}] Сканирование...")
            await hover_and_scan_at_corner(drone, pose_latest, acc, height, scan_duration)
        else:
            print(f"[WARN] Угол {corners_found} не достигнут, продолжаем...")
    
    print("\n[PHASE 5] Возврат в исходную точку...")
    
    # Получаем текущую позицию
    pose_msg, _ = pose_latest.snapshot()
    if pose_msg:
        current_pos = pose_msg.get("position", {})
        current_n = float(current_pos.get("x", 0))
        current_e = float(current_pos.get("y", 0))
        
        # Поворачиваемся к начальной точке
        dn = start_pos[0] - current_n
        de = start_pos[1] - current_e
        target_yaw = math.atan2(de, dn)
        await turn_to_yaw(drone, target_yaw, pose_latest)
    
    # Летим к начальной точке с избеганием препятствий
    await fly_to_point_with_avoidance(
        drone=drone,
        target_n=start_pos[0],
        target_e=start_pos[1],
        height=height,
        pose_latest=pose_latest,
        lidar_latest=lidar_latest,
        cruise_speed=velocity,
        stop_dist=0.5
    )
    
    print("\n[PHASE 6] Плавное снижение...")
    
    # Плавное снижение (на высоту 1м сначала)
    pose_msg, _ = pose_latest.snapshot()
    if pose_msg:
        p = pose_msg.get("position", {})
        current_n = float(p.get("x", 0))
        current_e = float(p.get("y", 0))
        await drone.move_to_position_async(current_n, current_e, -1.0, 1.0)
        await asyncio.sleep(1.0)
    
    print("[MISSION] Облет по стенам завершен!")

class SafePathTracker(PathTracker):
    """
    Исправленная версия PathTracker, которая рисует только новые сегменты пути,
    а не перерисовывает весь путь целиком на каждом шаге.
    Это предотвращает переполнение видеопамяти (VRAM) из-за наслоения тысяч линий.
    """
    def update_position(self, position_ned):
        n, e, d = position_ned
        new_point = [float(n), float(e), float(d)]
        
        # Проверяем расстояние до последней точки
        if self.last_point is not None:
            dist = math.sqrt(
                (new_point[0] - self.last_point[0]) ** 2 +
                (new_point[1] - self.last_point[1]) ** 2 +
                (new_point[2] - self.last_point[2]) ** 2
            )
            if dist < self.min_distance:
                return

        # Добавляем точку
        self.path_points.append(new_point)
        
        # Рисуем ТОЛЬКО новый сегмент
        if self.last_point is not None:
            try:
                segment = [self.last_point, new_point]
                # Красный цвет: [R, G, B, A]
                red_color = [1.0, 0.0, 0.0, 1.0]
                thickness = 3.0
                duration = 0.0  # Persistent
                is_persistent = True
                
                self.world.plot_debug_solid_line(
                    points=segment,
                    color_rgba=red_color,
                    thickness=thickness,
                    duration=duration,
                    is_persistent=is_persistent
                )
            except Exception:
                pass

        self.last_point = new_point

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
    parser.add_argument("--resume", action="store_true", help="Продолжить построение существующей карты (result.ply).")
    parser.add_argument("--enable-visualizer", action="store_true", help="Включить визуализацию облака точек в реальном времени (по умолчанию отключена для экономии видеопамяти).")
    parser.add_argument("--max-visualizer-points", type=int, default=100_000, help="Максимальное количество точек для визуализации (по умолчанию: 100000).")
    
    args = parser.parse_args()

    # --- КОПИЯ SETUP КОДА ИЗ lidar_mapping/cli.py ---
    client = ProjectAirSimClient()
    # Используем FullPointCloudAccumulator для накопления ВСЕХ точек со всех сканирований
    # с агрессивной фильтрацией для чистой карты:
    # - voxel_size=0.10м (укрупненные воксели для удаления дубликатов)
    # - downsample_threshold=300k (частая фильтрация)
    # - max_range=25м (удаление далеких шумовых точек)
    # - sor_std_ratio=1.2 (агрессивное удаление выбросов)
    acc = FullPointCloudAccumulator(
        voxel_size=0.10,             # Увеличен для чистоты карты
        downsample_threshold=300_000, # Чаще применять фильтрацию
        max_range=25.0,              # Максимальная дальность лидара
        min_range=0.3,               # Минимальная (фильтр близких отражений)
        sor_neighbors=15,            # Соседей для SOR
        sor_std_ratio=1.2            # Порог выбросов (ниже = агрессивнее)
    )
    
    # Инициализация для resume
    existing_path_points = []
    
    # Загрузка существующей карты, если запрошено
    if args.resume:
        existing_map, existing_path = load_point_cloud_from_ply("result.ply")
        if existing_map is not None:
            acc.add_points(existing_map)
            print(f"[INIT] Загружена карта: {existing_map.shape[0]} точек.")
        if existing_path is not None:
            # Convert numpy to list of lists for PathTracker
            existing_path_points = existing_path.tolist()
            print(f"[INIT] Загружена траектория: {len(existing_path_points)} точек.")
    
    tmp_cfg_dir: Optional[tempfile.TemporaryDirectory] = None
    lidar_latest = LidarLatest()
    pose_latest = PoseHistory() # Используем PoseHistory вместо PoseLatest
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
        # Используем SafePathTracker для предотвращения утечки памяти
        path_tracker = SafePathTracker(world, min_distance=0.1)
        
        # Восстанавливаем старый путь, если есть
        if existing_path_points:
            path_tracker.path_points = existing_path_points
            if len(existing_path_points) > 0:
                path_tracker.last_point = existing_path_points[-1]
                # Отрисовываем старый путь (опционально, может занять время для большого пути)
                try:
                     world.plot_debug_solid_line(
                        points=existing_path_points,
                        color_rgba=[1.0, 0.0, 0.0, 1.0],
                        thickness=3.0,
                        duration=0.0,
                        is_persistent=True
                    )
                except Exception: pass
            print(f"[INIT] Восстановлен путь в трекере ({len(existing_path_points)} точек)")

        # Запуск визуализатора (отключен по умолчанию для экономии видеопамяти)
        visualizer = None
        if args.enable_visualizer:
            try:
                visualizer = RealtimePointCloudVisualizer(
                    acc=acc, 
                    update_interval=0.1,
                    max_display_points=args.max_visualizer_points
                )
                visualizer.start()
                print(f"[VISUALIZER] Визуализация запущена (макс. точек: {args.max_visualizer_points}).")
            except Exception as e:
                print(f"[WARN] Не удалось запустить визуализацию: {e}")
                visualizer = None
        else:
            print("[VISUALIZER] Визуализация отключена (используйте --enable-visualizer для включения).")

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
                 
                 # Минимальное количество точек для добавления (фильтр неполных сканов)
                 if pts_body.shape[0] < 50:
                     return
                 
                 # Получаем timestamp лидара для точной синхронизации
                 lidar_ts = 0.0
                 ts_raw = lidar_data.get("time_stamp")
                 if ts_raw:
                     # AirSim часто шлет наносекунды
                     if isinstance(ts_raw, (int, float)) and ts_raw > 1e12:
                         lidar_ts = float(ts_raw) / 1e9
                     else:
                         lidar_ts = float(ts_raw)
                 else:
                     lidar_ts = time.time()

                 # Transform с использованием интерполированной позы
                 pose_msg = pose_latest.get_interpolated_pose(lidar_ts)
                 
                 if pose_msg and isinstance(pose_msg, dict):
                     pos = pose_msg.get("position", {})
                     ori = pose_msg.get("orientation", {})
                     if pos and ori:
                         pts_world = _transform_points_body_to_world(pts_body, pos, ori)
                         
                         # Передаем позицию дрона для фильтрации по дальности
                         drone_pos = np.array([
                             float(pos.get("x", 0)),
                             float(pos.get("y", 0)),
                             float(pos.get("z", 0))
                         ])
                         acc.add_points(pts_world, drone_position=drone_pos)
                         lidar_latest.update(pts_body, stamp=lidar_ts)
                     else:
                         # Без позиции не добавляем в карту (только для навигации)
                         lidar_latest.update(pts_body, stamp=lidar_ts)
                 else:
                     # Если позы нет, используем только для навигации
                     lidar_latest.update(pts_body, stamp=lidar_ts)
             except Exception as e:
                 pass

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

        # Выполнение миссии - исследование периметра
        await explore_room_perimeter(
            drone=drone,
            pose_latest=pose_latest,
            lidar_latest=lidar_latest,
            acc=acc,
            height=args.height,
            velocity=1.0,       # Уменьшенная скорость для безопасного облета
            scan_duration=args.scan_duration,
            target_wall_dist=1.0 # Расстояние до стены (было 3.0)
        )

        # --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---
        print("[SAVE] Финальная очистка и сохранение результатов...")
        
        # Принудительная финальная очистка перед сохранением
        acc.force_cleanup()
        
        # Получаем точки с финальной фильтрацией
        final_points = acc.snapshot(apply_final_filter=True)
        
        stats = acc.get_stats()
        print(f"[SAVE] Статистика: {stats['total_points']} точек, {stats['filter_count']} итераций фильтрации")
        
        # Получаем траекторию
        final_path = path_tracker.path_points if path_tracker else []
        
        if (final_points is not None and final_points.shape[0] > 0) or len(final_path) > 0:
            save_point_cloud_to_ply(final_points, trajectory_xyz=final_path, filename="result.ply")
            
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
        try:
            # Пытаемся посадить дрон с таймаутом
            await asyncio.wait_for(drone.land_async(), timeout=5.0)
        except (Exception, asyncio.TimeoutError) as e:
            # Игнорируем ошибки при посадке (соединение может быть уже разорвано)
            pass
            
        try:
            drone.disarm()
            drone.disable_api_control()
        except Exception:
            # Игнорируем ошибки при разоружении
            pass

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Останавливаем визуализатор, если он был запущен
        if visualizer and visualizer.is_running():
            try:
                visualizer.stop()
            except Exception:
                pass
        
        try:
            # Корректное отключение с обработкой ошибок соединения
            await asyncio.sleep(0.1)  # Небольшая задержка для завершения операций
            
            # Проверяем, что клиент еще подключен перед отключением
            if client is not None:
                try:
                    client.disconnect()
                except Exception as e:
                    # Игнорируем ошибки при отключении (соединение может быть уже разорвано)
                    # ConnectionReset и другие ошибки соединения - это нормально при завершении
                    pass
        except Exception as e:
            # Игнорируем все ошибки при завершении (включая ConnectionReset из фоновых задач)
            pass
        
        # Подавляем предупреждения о незавершенных задачах при завершении
        try:
            # Даем время на завершение всех фоновых задач
            await asyncio.sleep(0.2)
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
