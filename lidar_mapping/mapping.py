"""Point cloud accumulation + saving + 2D map generation."""

from __future__ import annotations

import os
import threading
from collections import deque
from typing import Optional, Tuple

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


def save_point_cloud_to_ply(points_xyz, trajectory_xyz=None, filename: str = "result.ply"):
    """
    Сохраняет облако точек и траекторию в PLY файл.
    
    Args:
        points_xyz: numpy array (N, 3) точек препятствий (NED)
        trajectory_xyz: numpy array (M, 3) точек траектории (NED) или список
        filename: имя файла для сохранения
    """
    try:
        import numpy as np
        import open3d as o3d
    except ImportError as e:
        print(f"[ERROR] Не удалось импортировать open3d: {e}")
        return False
    
    has_points = points_xyz is not None and getattr(points_xyz, "size", 0) > 0
    
    # Обработка траектории
    traj_arr = None
    if trajectory_xyz is not None:
        if isinstance(trajectory_xyz, list):
             if len(trajectory_xyz) > 0:
                 traj_arr = np.array(trajectory_xyz, dtype=np.float32)
        else:
             if getattr(trajectory_xyz, "size", 0) > 0:
                 traj_arr = trajectory_xyz

    has_traj = traj_arr is not None and traj_arr.size > 0

    if not has_points and not has_traj:
        print("[ERROR] Нет данных для сохранения (ни точек, ни траектории)!")
        return False
    
    try:
        # Lists for merging
        all_points = []
        all_colors = []

        # 1. Obstacle Points (Green)
        if has_points:
            # NED -> Visual conversion
            # Visual: X=Right, Y=Up, Z=Forward
            # NED: X=North, Y=East, Z=Down
            # We want: Vis X = NED Y, Vis Y = -NED Z, Vis Z = NED X (Standard ROS/AirSim view usually X=Forward, but Open3D default is Y up)
            
            # Let's stick to the previous transform that worked for the user:
            # Vis X = NED Y
            # Vis Y = -NED Z
            # Vis Z = -NED X (Wait, previously it was -NED X? Let's check original code)
            # Original: vis[:, 2] = -points_xyz[:, 0]
            
            p_vis = np.zeros_like(points_xyz)
            p_vis[:, 0] = points_xyz[:, 1]   # Right = East
            p_vis[:, 1] = -points_xyz[:, 2]  # Up = -Down
            p_vis[:, 2] = -points_xyz[:, 0]  # Forward = -North (This rotates the map 180 degrees? Usually Forward=North)
            # If North is X, and we want Forward to be -Z (OpenGL convention), then North->-Z is correct if we look down Z.
            # But typically: X=North, Y=East, Z=Down.
            # Visual: X=East, Y=-Down(Up), Z=North.
            # Original code: points_vis[:, 2] = -points_xyz[:, 0] -> Z = -North.
            
            all_points.append(p_vis)
            
            # === Color classification: Floor (gray), Walls (blue), Obstacles (green) ===
            vis_x = p_vis[:, 0]  # East/Right
            vis_y = p_vis[:, 1]  # Up (height)
            vis_z = p_vis[:, 2]  # North/Forward
            
            # 1. Floor: lowest points (within 1.5m from minimum height)
            # Increased threshold to capture more floor area
            min_y = np.min(vis_y)
            floor_mask = (vis_y < min_y + 1.5)
            
            # 2. Walls: points near the boundaries of the map (outer perimeter)
            # Calculate bounding box
            min_x, max_x = np.min(vis_x), np.max(vis_x)
            min_z, max_z = np.min(vis_z), np.max(vis_z)
            
            # Wall margin: points within 1.5m from the outer boundary are considered walls
            wall_margin = 1.5
            near_min_x = (vis_x < min_x + wall_margin)
            near_max_x = (vis_x > max_x - wall_margin)
            near_min_z = (vis_z < min_z + wall_margin)
            near_max_z = (vis_z > max_z - wall_margin)
            
            # Wall mask: near any boundary AND not floor
            wall_mask = (near_min_x | near_max_x | near_min_z | near_max_z) & ~floor_mask
            
            # 3. Obstacles (shelves, etc.): everything else that is not floor and not wall
            obstacle_mask = ~floor_mask & ~wall_mask
            
            # Initialize colors array
            colors = np.zeros((p_vis.shape[0], 3))
            
            # Floor color (Gray)
            colors[floor_mask] = [0.4, 0.4, 0.4]
            
            # Wall color (Blue)
            colors[wall_mask] = [0.2, 0.4, 1.0]
            
            # Obstacle color (Green) - shelves, furniture, etc.
            colors[obstacle_mask] = [0.0, 1.0, 0.0]

            all_colors.append(colors)

        # 2. Trajectory Points (Red)
        if has_traj:
            t_vis = np.zeros_like(traj_arr)
            t_vis[:, 0] = traj_arr[:, 1]
            t_vis[:, 1] = -traj_arr[:, 2]
            t_vis[:, 2] = -traj_arr[:, 0]
            
            all_points.append(t_vis)
            # Red color
            all_colors.append(np.ones((t_vis.shape[0], 3)) * [1.0, 0.0, 0.0])

        # Combine
        final_points = np.concatenate(all_points, axis=0)
        final_colors = np.concatenate(all_colors, axis=0)
        
        # Create Cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_points)
        pcd.colors = o3d.utility.Vector3dVector(final_colors)
        
        # Save
        success = o3d.io.write_point_cloud(filename, pcd, write_ascii=False)
        
        if success:
            print(f"[SAVE] Сохранено: {final_points.shape[0]} точек (Map+Path) в {filename}")
            return True
        else:
            return False

    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_point_cloud_from_ply(filename: str = "result.ply"):
    """
    Загружает PLY. Разделяет точки на препятствия (Map) и траекторию (Path) по цвету.
    Зеленый -> Map, Красный -> Path.
    
    Returns:
        tuple: (map_points_ned, path_points_ned)
    """
    try:
        import numpy as np
        import open3d as o3d
    except ImportError:
        return None, None
        
    if not os.path.exists(filename):
        return None, None
        
    try:
        pcd = o3d.io.read_point_cloud(filename)
        if not pcd.has_points():
            return None, None
            
        points_vis = np.asarray(pcd.points)
        
        # Check colors
        map_indices = []
        path_indices = []
        
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            # Simple heuristic: Red channel dominants -> Path, Green channel dominants -> Map
            # Color is RGB [0..1]
            
            # Path (Red): R > 0.8, G < 0.2
            is_red = (colors[:, 0] > 0.5) & (colors[:, 1] < 0.5)
            # Map (Green): G > 0.5
            is_green = (colors[:, 1] > 0.5)
            
            path_indices = np.where(is_red)[0]
            map_indices = np.where(is_green | (~is_red & ~is_green))[0] # Fallback: treat others as map
        else:
            # No colors -> assume all map
            map_indices = np.arange(points_vis.shape[0])
            
        # Extract and Transform back to NED
        # NED X = -Vis Z
        # NED Y = Vis X
        # NED Z = -Vis Y
        
        def to_ned(vis_pts):
            if vis_pts.size == 0: return np.empty((0,3), dtype=np.float32)
            ned = np.zeros_like(vis_pts)
            ned[:, 0] = -vis_pts[:, 2]
            ned[:, 1] = vis_pts[:, 0]
            ned[:, 2] = -vis_pts[:, 1]
            return ned.astype(np.float32)

        map_points = to_ned(points_vis[map_indices]) if len(map_indices) > 0 else None
        path_points = to_ned(points_vis[path_indices]) if len(path_indices) > 0 else None
        
        print(f"[LOAD] Загружено: Map={len(map_indices)}, Path={len(path_indices)}")
        return map_points, path_points
        
    except Exception as e:
        print(f"[ERROR] Load error: {e}")
        return None, None


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
    
    # R - это матрица поворота Body -> World
    # v_world = R @ v_body
    # Для массивов точек (N,3) нам нужно pts @ R.T, что эквивалентно (R @ pts.T).T
    # Поэтому транспонировать R перед умножением НЕ НУЖНО, если мы используем формулу (R @ pts.T).T
    
    # Преобразуем точки: world = R @ body + pos
    drone_position = np.array([
        float(drone_pos.get("x", 0.0)),
        float(drone_pos.get("y", 0.0)),
        float(drone_pos.get("z", 0.0))
    ], dtype=np.float32)
    
    # Применяем поворот и перенос
    pts_world = (R @ pts.T).T + drone_position
    
    return pts_world


def _point_cloud_to_2d_map(
    points_xyz,
    resolution: float = 0.1,
    map_size_m: float = 100.0,
    drone_position: Optional[Tuple[float, float, float]] = None,
    avoid_dist: float = 12.0,
    flight_clearance_m: float = 1.0,
    min_points_per_cell: int = 3,
):
    """
    Преобразует облако точек в 2D карту (вид сверху).
    
    Args:
        points_xyz: numpy array (N, 3) с точками в системе координат NED (world frame)
        resolution: разрешение карты в метрах на пиксель
        map_size_m: размер карты в метрах (карта будет map_size_m x map_size_m)
        drone_position: опциональная позиция дрона (n, e, z) для определения критических препятствий
        avoid_dist: дистанция избегания для определения критических препятствий
    
    Returns:
        tuple: (map_2d, blocking_obstacles_mask, flight_obstacles_mask) где:
            - map_2d: numpy array с 2D картой (вид сверху), значение = макс. высота в ячейке (Up)
            - blocking_obstacles_mask: маска препятствий, критически близких к дрону и мешающих началу движения (или None)
            - flight_obstacles_mask: маска препятствий, которые потенциально мешают полёту по высоте/клиренсу (или None)
    """
    try:
        import numpy as np
    except Exception:
        return None, None
    
    if points_xyz is None or getattr(points_xyz, "size", 0) == 0:
        return None, None
    
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
    x_coords_valid = x_coords[valid_mask]
    y_coords_valid = y_coords[valid_mask]
    
    if len(x_px) == 0:
        return None, None
    
    # Создаем карту: для каждого пикселя считаем максимальную высоту (occupancy grid style)
    # Используем максимальную высоту для каждого пикселя, чтобы показать препятствия
    map_2d = np.zeros((map_size_px, map_size_px), dtype=np.float32)
    count_2d = np.zeros((map_size_px, map_size_px), dtype=np.int32)
    blocking_mask = np.zeros((map_size_px, map_size_px), dtype=bool) if drone_position is not None else None
    flight_obstacles_mask = np.zeros((map_size_px, map_size_px), dtype=bool) if drone_position is not None else None
    
    # Критическое расстояние для препятствий, мешающих началу движения
    critical_distance = max(2.0, avoid_dist * 0.1) if drone_position is not None else None

    # Высотный порог для препятствий, которые могут мешать полёту:
    # если верх препятствия поднимается до уровня полёта дрона минус клиренс — считаем опасным.
    danger_height_threshold = None
    if drone_position is not None:
        try:
            drone_alt_up = -float(drone_position[2])  # NED down -> Up
            danger_height_threshold = drone_alt_up - float(flight_clearance_m)
        except Exception:
            danger_height_threshold = None
    
    # Накапливаем максимальные высоты (для occupancy grid) + счетчик попаданий в ячейку
    for i in range(len(x_px)):
        count_2d[y_px[i], x_px[i]] += 1
        if z_coords[i] > map_2d[y_px[i], x_px[i]]:
            map_2d[y_px[i], x_px[i]] = z_coords[i]
        
        # Определяем критические препятствия, мешающие началу движения
        if blocking_mask is not None and critical_distance is not None:
            # Вычисляем расстояние от дрона до препятствия в горизонтальной плоскости
            dist_2d = np.sqrt((x_coords_valid[i] - drone_position[0])**2 + (y_coords_valid[i] - drone_position[1])**2)
            
            # Если препятствие слишком близко - помечаем как мешающее движению
            if dist_2d < critical_distance:
                # Чтобы не подсвечивать землю рядом с дроном, фильтруем по высоте (если можем)
                if danger_height_threshold is None or z_coords[i] >= danger_height_threshold:
                    blocking_mask[y_px[i], x_px[i]] = True

    # Маска препятствий, потенциально мешающих полёту по высоте/клиренсу
    if flight_obstacles_mask is not None and danger_height_threshold is not None:
        try:
            # Требуем минимум точек в ячейке, чтобы снизить шум от одиночных выбросов
            flight_obstacles_mask[:] = (count_2d >= int(min_points_per_cell)) & (map_2d >= float(danger_height_threshold))
        except Exception:
            pass
    else:
        flight_obstacles_mask = None

    return map_2d, blocking_mask, flight_obstacles_mask


def _save_map_to_png(
    map_2d,
    output_path: str,
    colormap: str = "gray",
    blocking_obstacles_mask=None,
    flight_obstacles_mask=None,
):
    """
    Сохраняет 2D карту в PNG файл.
    
    Args:
        map_2d: numpy array с 2D картой
        output_path: путь для сохранения PNG
        colormap: название цветовой карты matplotlib (например, "gray", "viridis", "jet")
        blocking_obstacles_mask: опциональная маска препятствий, мешающих началу движения (будут отрисованы красным)
        flight_obstacles_mask: опциональная маска препятствий, мешающих полёту по клиренсу (будут отрисованы оранжевым)
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
        
        # Если есть маски препятствий — накладываем их поверх карты
        has_blocking = blocking_obstacles_mask is not None and np.any(blocking_obstacles_mask)
        has_flight = flight_obstacles_mask is not None and np.any(flight_obstacles_mask)
        if has_blocking or has_flight:
            # Создаем RGB изображение из base colormap
            try:
                # Современный способ получения colormap
                cmap_obj = plt.colormaps.get_cmap(colormap)
            except AttributeError:
                # Старый способ для совместимости
                cmap_obj = plt.cm.get_cmap(colormap)

            map_rgb = cmap_obj(map_normalized)[:, :, :3].copy()

            # 1) Препятствия, мешающие полёту по клиренсу — оранжевым (менее "критично", чем красный)
            if has_flight:
                fy, fx = np.where(flight_obstacles_mask)
                map_rgb[fy, fx] = [1.0, 0.55, 0.0]  # Оранжевый
                print(f"[MAP] Обнаружено {int(np.sum(flight_obstacles_mask))} пикселей с препятствиями, мешающими полёту по клиренсу (оранжевым)")

            # 2) Критические (очень близко к дрону) — красным поверх
            if has_blocking:
                by, bx = np.where(blocking_obstacles_mask)
                map_rgb[by, bx] = [1.0, 0.0, 0.0]  # Красный
                print(f"[MAP] Обнаружено {int(np.sum(blocking_obstacles_mask))} пикселей с препятствиями, мешающими началу движения (красным)")

            ax.imshow(map_rgb, origin='lower', interpolation='nearest', alpha=0.9)
        
        ax.set_xlabel('X (метры)')
        ax.set_ylabel('Y (метры)')
        ax.set_title('SLAM Карта местности (вид сверху)\nОранжевым: препятствия по клиренсу, красным: критически близко')
        
        # Сохраняем в файл
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Выводим абсолютный путь к сохраненной карте
        output_path_absolute = os.path.abspath(output_path)
        print(f"[MAP] Карта сохранена в {output_path}")
        print(f"[MAP] Абсолютный путь: {output_path_absolute}")
        return True
    except Exception as e:
        print(f"[ERROR] Ошибка при сохранении карты: {e}")
        import traceback
        traceback.print_exc()
        return False

