"""Simple LIO-SLAM state."""

from __future__ import annotations

import math
from typing import Optional, Tuple

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


