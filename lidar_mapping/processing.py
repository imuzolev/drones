"""Point cloud processing + shelf detection."""

from __future__ import annotations

from typing import Tuple, List

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

